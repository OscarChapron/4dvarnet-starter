from copy import deepcopy
from mimetypes import init
import torch
import xarray as xr
import numpy as np
import random
from typing import Sequence, Union
from omegaconf import ListConfig   # only for type hints – optional
import torch.nn.functional as F
import kornia.filters as kfilts
import functools as ft
from collections import namedtuple
from torch import Tensor
from typing import Optional, Tuple, Union

from src.data import AugmentedDataset, BaseDataModule, XrDataset
from src.models import Lit4dVarNet, GradSolver, BaseObsCost, BilinAEPriorCost
from src.utils import get_constant_crop

TrainingItem = namedtuple('TrainingItem', ['input', 'tgt'])
torch.set_float32_matmul_precision('high')

def _rbf_cov(
    x: Tensor,
    y: Tensor,
    length_scale: Union[float, Tensor],
    sigma_f: Union[float, Tensor],
) -> Tensor:
    """Radial‑basis‑function (squared‑exponential) covariance.

    Parameters
    ----------
    x, y : (N, 2) and (M, 2) tensors of *physical* coordinates (same units)
    length_scale : scalar (>0) – decorrelation length.
    sigma_f : scalar – marginal standard deviation (signal).
    """
    # Ensure tensors and broadcastable shapes
    if not torch.is_tensor(length_scale):
        length_scale = torch.as_tensor(length_scale, dtype=x.dtype, device=x.device)
    if not torch.is_tensor(sigma_f):
        sigma_f = torch.as_tensor(sigma_f, dtype=x.dtype, device=x.device)

    sqdist = ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)  # (N, M)
    return sigma_f ** 2 * torch.exp(-0.5 * sqdist / (length_scale ** 2))


@torch.no_grad()
def optimal_interpolation(
    data: Tensor,
    *,
    length_scale: float | Tensor = 15.0,
    sigma_f: float | Tensor = 1.0,
    sigma_n: float | Tensor = 0.1,
    coords: Optional[Tensor] = None,
    return_std: bool = False,
    jitter: float = 1e-6,
) -> Tuple[Tensor, Optional[Tensor]]:
    device, dtype = data.device, data.dtype
    H, W = data.shape

    # ------------------------------------------------------------------
    # Coordinates
    if coords is None:
        ii, jj = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing="ij",
        )
        coords = torch.stack((jj, ii), dim=-1)  # x = col(j), y = row(i) → shape (H, W, 2)
    else:
        if coords.shape != (H, W, 2):
            raise ValueError("coords must have shape (H, W, 2)")
        coords = coords.to(device=device, dtype=dtype)

    observed_mask = ~torch.isnan(data)
    missing_mask = ~observed_mask

    if observed_mask.sum() == 0:
        raise ValueError("No observed points in `data`. Cannot perform OI.")
    if missing_mask.sum() == 0:  # nothing to fill
        return data.clone(), (None if not return_std else torch.zeros_like(data))

    obs_coords = coords[observed_mask].reshape(-1, 2)
    mis_coords = coords[missing_mask].reshape(-1, 2)
    obs_vals = data[observed_mask].to(dtype)

    # ------------------------------------------------------------------
    # Build covariance matrices
    K_oo = _rbf_cov(obs_coords, obs_coords, length_scale, sigma_f)
    if not torch.is_tensor(sigma_n):
        sigma_n = torch.as_tensor(sigma_n, dtype=dtype, device=device)
    K_oo.diagonal().add_(sigma_n ** 2 + jitter)

    K_mo = _rbf_cov(mis_coords, obs_coords, length_scale, sigma_f)

    # ------------------------------------------------------------------
    # Solve K_oo * alpha = obs_vals (via Cholesky) ----------------------
    L = torch.linalg.cholesky(K_oo)
    alpha = torch.cholesky_solve(obs_vals.unsqueeze(-1), L).squeeze(-1)

    # ------------------------------------------------------------------
    # Posterior mean at missing
    mean_mis = K_mo @ alpha

    analysed = data.clone()
    analysed[missing_mask] = mean_mis

    if not return_std:
        return analysed, None

    # ------------------------------------------------------------------
    # Posterior variance diag = K_mm - K_mo @ K_oo^{-1} @ K_om
    # Compute v = L^{-1} K_om^T  (K_om = K_mo.T)
    v = torch.linalg.solve_triangular(L, K_mo.T, upper=False)
    var_mis = _rbf_cov(mis_coords, mis_coords, length_scale, sigma_f).diagonal() - (v ** 2).sum(0)
    std = torch.zeros_like(data)
    std[missing_mask] = var_mis.clamp_min(0.0).sqrt()

    return analysed, std

class BaseDataModule_Fasc(BaseDataModule):
    def __init__(
        self,
        input_da,
        domains,
        xrds_kw,
        dl_kw,
        *,
        norm_type: str = "z_score",
        aug_kw: dict | None = None,
        norm_stats: tuple | None = None,
        **kwargs,
    ):
        self.norm_type = norm_type
        super().__init__(
            input_da,
            domains,
            xrds_kw,
            dl_kw,
            aug_kw=aug_kw,
            norm_stats=norm_stats,
            **kwargs,
        )
        self.train_ds = self.val_ds = self.test_ds = None
        
    def norm_stats(self):
        if self._norm_stats is None:
            if self.norm_type == "z_score":
                self._norm_stats = self.train_mean_std()
            elif self.norm_type == "min_max":
                self._norm_stats = self.min_max_norm()
            else:
                raise ValueError(f"Unknown norm_type: {self.norm_type}")
            print("Norm stats", self._norm_stats)
        return self._norm_stats

    def min_max_norm(self, variable: str = "tgt"):
        train_data = (
            self.input_da.sel(self.xrds_kw.get("domain_limits", {}))
                        .sel(self.domains["train"])
        )
        vmin = train_data.sel(variable=variable).min().values.item()
        vmax = train_data.sel(variable=variable).max().values.item()
        return vmin, vmax

    def post_fn(self):
        if self.norm_type == "z_score":
            m, s = self.norm_stats()
            normalise = lambda x: (x - m) / s
        else:  # "min_max"
            vmin, vmax = self.norm_stats()
            normalise = lambda x: (x - vmin) / (vmax - vmin)

        return ft.partial(
            ft.reduce,
            lambda itm, fn: fn(itm),
            [
                TrainingItem._make,
                lambda itm: itm._replace(tgt=normalise(itm.tgt)),
                lambda itm: itm._replace(input=normalise(itm.input)),
            ],
        )

class TransfertDataModule(BaseDataModule_Fasc):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean_std_domain = kwargs.get('mean_std_domain', 'train')
        self.std_c = kwargs.get('std_c', 1.)
    
    def train_mean_std(self, variable='tgt'):
        train_data = self.input_da.sel(self.xrds_kw.get('domain_limits', {})).sel(self.domains[self.mean_std_domain])
        return train_data.sel(variable=variable).pipe(lambda da: (da.mean().values.item(), self.std_c*da.std().values.item()))
    
    def min_max_norm(self, variable = 'tgt'):
        train_data = self.input_da.sel(self.xrds_kw.get('domain_limits', {})).sel(self.domains[self.mean_std_domain])
        min_value = train_data.sel(variable=variable).min().values.item()
        max_value = train_data.sel(variable=variable).max().values.item()
        return min_value, max_value

    def setup(self, stage='test'):
        post_fn = self.post_fn()
        if stage == 'fit':
            train_data = self.input_da.sel(self.domains['train'])
            train_xrds_kw = deepcopy(self.xrds_kw)
            
            self.train_ds = XrDataset(
                train_data, **train_xrds_kw, postpro_fn=post_fn,
            )
            if self.aug_kw:
                self.train_ds = AugmentedDataset(self.train_ds, **self.aug_kw)

            self.val_ds = XrDataset(
                self.input_da.sel(self.domains['val']),
                **self.xrds_kw,
                postpro_fn=post_fn,
            )
        else:
            self.test_ds = XrDataset(
                self.input_da.sel(self.domains['test']),
                **self.xrds_kw,
                postpro_fn=post_fn,
            )

def cosanneal_lr_adamw(lit_mod, lr, T_max, weight_decay=0.):
    opt = torch.optim.AdamW(
        [
            {'params': lit_mod.solver.grad_mod.parameters(), 'lr': lr},
            {'params': lit_mod.solver.obs_cost.parameters(), 'lr': lr},
            {'params': lit_mod.solver.prior_cost.parameters(), 'lr': lr / 2},
        ], weight_decay=weight_decay
    )
    return {
        'optimizer': opt,
        'lr_scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=T_max,
        ),
    }

def load_and_interpolate(tgt_path, inp_path, tgt_var, inp_var, domain):
    """
    Load ground truth `tgt` and apply the satellites observations `inp`.
    """
    tgt = xr.open_dataset(tgt_path)[tgt_var].sel(domain)
    inp = xr.open_dataset(inp_path)[inp_var].sel(domain)

    return (
        xr.Dataset(
            dict(input=inp*tgt, tgt=(tgt.dims, tgt.values)),
            inp.coords,
        )
        .transpose('time', 'lat', 'lon')
        .to_array()
    )

def load_natl_data(tgt_path, tgt_var, inp_path, inp_var, **kwargs):
    tgt = (
        xr.open_dataset(tgt_path)[tgt_var]
        .sel(kwargs.get('domain', None))
        .sel(kwargs.get('period', None))
        .pipe(threshold_xarray)
    )
    inp = (
        xr.open_dataset(inp_path)[inp_var]
        .sel(kwargs.get('domain', None))
        .sel(kwargs.get('period', None))
        .pipe(threshold_xarray)
        #.pipe(mask)
    )
    print(xr.Dataset(
            dict(input=inp, tgt=(tgt.dims, tgt.values)),
            inp.coords,
        )
        .transpose('time', 'lat', 'lon')
        .to_array())
    return (
        xr.Dataset(
            dict(input=inp, tgt=(tgt.dims, tgt.values)),
            inp.coords,
        )
        .transpose('time', 'lat', 'lon')
        .to_array()
    )

def load_natl_data_pca(tgt_path, tgt_var, inp_path, inp_var, **kwargs):
    tgt = (
        xr.open_dataset(tgt_path)[tgt_var]
        .sel(kwargs.get('domain', None))
        .sel(kwargs.get('period', None))
        .pipe(threshold_xarray)
    )
    inp = (
        xr.open_dataset(inp_path)[inp_var]
        .sel(kwargs.get('domain', None))
        .sel(kwargs.get('period', None))
        .pipe(threshold_xarray)
    )

    ds = xr.Dataset(
        dict(input=inp, tgt=(tgt.dims, tgt.values)),
        inp.coords,
    ).stack(time_component=('time', 'component'))
    
    ds = ds.transpose('time_component', 'lat', 'lon').to_array()
    print(ds)
    
    return ds

def get_dirac_time_wei(patch_dims, offset=0, **crop_kw):
    pw = get_constant_crop(patch_dims, **crop_kw)
    time_size = patch_dims["time"]
    
    # Calculate the center index(es)
    if time_size % 2 == 0:  # Even
        center1 = time_size // 2 - 1
        center_condition = lambda t: (t == center1) * pw
    else:  # Odd
        center1 = time_size // 2
        center_condition = lambda t: (t == center1) * pw
    
    # Use np.fromfunction to create the array
    return np.fromfunction(
        lambda t, *a: center_condition(t).astype(float),
        patch_dims.values(),
    )

def mask(da, sampling_rate = 0.1):
    time_dim = da.time.size
    lat_dim = da.lat.size
    lon_dim = da.lon.size

    random_mask = np.random.choice([0, 1], size=(time_dim, lat_dim, lon_dim), p=[1 - sampling_rate, sampling_rate])
    mask_data_array = xr.DataArray(random_mask, dims=['time', 'lat', 'lon'])
    masked_data_array = da.where(mask_data_array == 1, other=np.nan)
    return masked_data_array

def threshold_xarray(da):
    threshold = 999
    da = xr.where(da > threshold, 0, da)
    da = xr.where(da <= 0, 0, da)
    return da


def _uniform_sr(sampling_rate: Union[float, Sequence[float], ListConfig]) -> float:
    """Return a scalar subsampling rate; handles (min,max) tuples."""
    if isinstance(sampling_rate, (list, tuple, ListConfig)) and len(sampling_rate) == 2:
        return random.uniform(sampling_rate[0], sampling_rate[1])
    return float(sampling_rate)

class Lit4dVarNet_Fasc(Lit4dVarNet):
    def __init__(
        self,
        *args,
        sampling_rate: Union[float, Sequence[float], ListConfig] = 1.0,
        norm_type: str = "z_score",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sampling_rate = sampling_rate
        self.norm_type = norm_type
    
    def weighted_mse_transfert(self, err: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        err:    (B, T*C, H, W)
        weight: (H, W) or (T*C, H, W)
        Returns a scalar loss with correct dtype/device and no shape warnings.
        """
        # Broadcast weight to err
        if weight.ndim == 2:
            w = weight[None, None, ...]          # (1,1,H,W)
        elif weight.ndim == 3:
            w = weight[None, ...]                # (1,T*C,H,W)
        else:
            raise ValueError(f"Unsupported weight.ndim={weight.ndim}")

        # Match dtype/device to err to avoid upcasting to float64
        w = w.to(device=err.device, dtype=err.dtype)

        # Valid mask: finite error and nonzero weight
        sel = torch.isfinite(err) & (w != 0)

        if not sel.any():
            # return a tensor on the right device/dtype that still has grad
            return torch.tensor(1000.0, device=err.device, dtype=err.dtype, requires_grad=True)

        x = (err * w)[sel]
        # Mean of squared residuals (no size mismatch warnings)
        return (x * x).mean()

    @staticmethod
    def weighted_mse_mask(err, weight, mask_nan):
        # keep only valid (non-NaN) positions
        err_valid    = err * mask_nan[None, ...]
        weight_valid = weight * mask_nan[None, ...]
        valid        = err.isfinite() & (weight_valid != 0.)
        if not valid.any():
            return torch.tensor(1_000., device=err.device)
        return F.mse_loss(
            (err_valid * weight_valid)[valid],
            torch.zeros_like(err_valid[valid]),
        )
    
    def _apply_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SR-controlled NaN mask *per sample*."""
        x = x.clone()
        for b in range(x.size(0)):
            sr   = _uniform_sr(self.sampling_rate)
            mask = (torch.rand_like(x[b], dtype=torch.float32) > sr)
            x[b][mask] = float("nan")
        return x
    
    def forward(self, batch):
        # keep parent behaviour except when n_step==0
        if batch.input.ndim == 5:
            shape = batch.input.shape
            new_shape = (shape[0], shape[1] * shape[2], shape[3], shape[4])
            batch = batch._replace(
            input=batch.input.reshape(new_shape),
            tgt=batch.tgt.reshape(new_shape) if batch.tgt.ndim == 5 else batch.tgt
            )
        if self.solver.n_step > 0:
            return super().forward(batch)
        return self.solver.prior_cost.forward_ae(
            batch.input.nan_to_num().detach().requires_grad_(True)
        )
    
    def step(self, batch, phase=""):
        if self.training and batch.tgt.isfinite().float().mean() < 0.9:
            return None, None
        
        had_5d = (batch.input.ndim == 5)
        if had_5d:
            B, T, C, H, W = batch.input.shape
            shape = batch.input.shape
            new_shape = (shape[0], shape[1] * shape[2], shape[3], shape[4])
            batch = batch._replace(
                input=batch.input.reshape(new_shape),
                tgt=batch.tgt.reshape(new_shape) if batch.tgt.ndim == 5 else batch.tgt
            )
        # stochastic masking
        batch = batch._replace(input=self._apply_mask(batch.input))
        # If batch.input has 5 dimensions, repeat rec_weight along the D (component) axis

        # run parent loss

        loss, out   = self.base_step(batch, phase)
        grad_loss   = self.weighted_mse_transfert(
            kfilts.sobel(out) - kfilts.sobel(batch.tgt),
            self.rec_weight
            )
        self.log(f"{phase}_gloss", grad_loss, prog_bar=True, on_step=False, on_epoch=True)

        if had_5d and C > 2:
            out5 = out.view(B, T, C, H, W)
            tgt5 = batch.tgt.view(B, T, C, H, W)

            g_out = out5[:, :, 1:, :, :] - out5[:, :, :-1, :, :]
            g_tgt = tgt5[:, :, 1:, :, :] - tgt5[:, :, :-1, :, :]

            # flatten back to 4D for weighted_mse_transfert
            err_c = (g_out - g_tgt).view(B, (T * (C - 1)), H, W)  # (B, T*(C-1), H, W)

            # adapt weights
            w_c = self.rec_weight.view(T, C, H, W)[:, :-1].reshape(T*(C-1), H, W)
            comp_gloss = self.weighted_mse_transfert(err_c, w_c)
            self.log(f"{phase}_comp_gloss", comp_gloss, prog_bar=True, on_step=False, on_epoch=True)
        else:
            comp_gloss = torch.tensor(0., device=out.device)
        
        if self.solver.n_step > 0:
            prior_cost  = self.solver.prior_cost(
                self.solver.init_state(batch, out)
            )
            self.log(f"{phase}_prior_cost", prior_cost, prog_bar=True, on_step=False, on_epoch=True)
            # Ensure loss terms are tensors and convert to numpy scalars if needed
            composite = 10 * loss + 20 * prior_cost + 5 * grad_loss + 5 * comp_gloss
            return composite, out
        total_loss = 10 * loss + 50 * grad_loss + 1 * comp_gloss
        return total_loss, out

    def base_step(self, batch, phase=""):
        out = self(batch=batch)
        loss = self.weighted_mse_transfert(out - batch.tgt, self.rec_weight)

        with torch.no_grad():
            # if isinstance(self.norm_stats[2], torch.Tensor):
            #     norm_stats_mean = self.norm_stats[2].mean().cpu()
            # else:
            #     norm_stats_mean = self.norm_stats[2]
            self.log(f"{phase}_mse", 10000 *self.norm_stats[2] * loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{phase}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss, out
    
    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.test_data = []

        raw_input = batch.input.clone()
        batch     = batch._replace(input=self._apply_mask(batch.input))
        out       = self(batch=batch)

        if self.norm_type == "z_score":
            m, s = self.norm_stats
            denorm = lambda z: z * s + m
        else:   # min_max
            vmin, vmax = self.norm_stats
            denorm = lambda z: (z - vmin) / (vmax - vmin)

        self.test_data.append(torch.stack(
            [
                denorm(raw_input.cpu()),
                denorm(batch.input.cpu()),
                denorm(batch.tgt.cpu()),
                denorm(out.squeeze(dim=-1).detach().cpu()),
            ],
            dim=1,
        ))

    @property
    def test_quantities(self):
        return ["input", "inp", "tgt", "out"]
    
class GradSolver_Fasc(GradSolver):
    def __init__(
        self,
        prior_cost, obs_cost, grad_mod, n_step,
        weight_obs: float = 1.0,
        weight_prior: float = 1.0,
        init_mode: str = "default",
        **kwargs,
    ):
        super().__init__(prior_cost, obs_cost, grad_mod, n_step, **kwargs)
        # make weights learnable
        self.weight_obs_torch   = torch.nn.Parameter(torch.tensor(weight_obs))
        self.weight_prior_torch = torch.nn.Parameter(torch.tensor(weight_prior))

        self.init_mode = init_mode
        # better initialisation
        self.prior_cost.apply(lambda m: torch.nn.init.kaiming_uniform_(m.weight)
                              if isinstance(m, torch.nn.Conv2d) else None)
        self.grad_mod.apply(lambda m: torch.nn.init.xavier_uniform_(m.weight)
                            if isinstance(m, torch.nn.Conv2d) else None)
        if self.init_mode == 'OI':
            self.oi_kwargs =  dict(length_scale=18.0,     # ← any hyper‑params you want
                                    sigma_f=1.0,
                                    sigma_n=0.05)

    def init_state(self, batch, x_init=None):
        if x_init is not None:
            return x_init
        
        mode = self.init_mode
        if mode == 'OI': 
            # batch.input shape is (B, C, H, W) in the datamodule used by Lit4dVarNet
            B, T, H, W = batch.input.shape
            x_filled = torch.empty_like(batch.input)
            # loop over batch and channels; OI works on *one* 2‑D field
            for b in range(B):
                for t in range(T):
                    x_filled[b, t] = self._oi_fill(
                        batch.input[b, t],           # (H, W) slice
                        **self.oi_kwargs
                    )

                x_init = x_filled
            return x_init.detach().requires_grad_(True)
        
        if mode == 'default':
            return batch.input.nan_to_num().detach().requires_grad_(True)
        
        if mode == 'zeros':
            return torch.zeros_like(batch.input).requires_grad_(True)

    def _oi_fill(self, sample, **oi_kw):
        """
        Run optimal_interpolation on **one** 2‑D (H,W) tensor that may contain NaNs.
        Returns a (H,W) tensor with no missing values.
        """
        analysed, _ = optimal_interpolation(sample, **oi_kw)
        return analysed

class BaseObsCost_Fasc(BaseObsCost):
    def __init__(self, weight1: float = 1.0, **kw):
        super().__init__(w=kw.pop("w", 1))            # keep parent scalar
        self.weight1_torch = torch.nn.Parameter(torch.tensor(weight1))


class BilinAEPriorCost_Fasc(BilinAEPriorCost):
    def __init__(self, *args, weight3: float = 1.0, **kw):
        super().__init__(*args, **kw)
        self.weight3_torch = torch.nn.Parameter(torch.tensor(weight3))

def run(trainer, train_dm, test_dm, lit_mod, ckpt=None):
    """
    Fit and test on two distinct domains.
    """
    print('-*---------------')
    if trainer.logger is not None:
        print()
        print('Logdir:', trainer.logger.log_dir)
        print()

    #trainer.fit(lit_mod, datamodule=train_dm, ckpt_path=ckpt)
    trainer.test(lit_mod, datamodule=test_dm, ckpt_path=ckpt)