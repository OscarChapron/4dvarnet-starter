from copy import deepcopy
import torch
import kornia.filters as kfilts
import xarray as xr
from collections import namedtuple
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities import grad_norm
import pandas as pd
from pathlib import Path
import math
import functools as ft
import random
from omegaconf import ListConfig
import src.data
import src.models
import src.utils

from src.data import AugmentedDataset, BaseDataModule, XrDataset

TrainingItem = namedtuple('TrainingItem', ['input', 'tgt'])

class PositionalEncoding(nn.Module):
    """Adds positional encoding to the normalized input."""
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        d_model = d_model - 1
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, H, W)
        B, T, H, W = x.size()
        pe = self.pe[:, :T].expand(B, -1, -1)  # (B, T, d_model)
        pe = pe.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, H, W)  # (B, T, d_model, H, W)
        x = x.unsqueeze(2)  # (B, T, 1, H, W)
        x = torch.cat([x, pe], dim=2)  # (B, T, d_model+1, H, W)
        return x

class NormPredictor(nn.Module):
    """
    A small network that:
      1. Takes in partial data of shape (B, T, lat, lon).
      2. Optionally uses time positional encoding.
      3. Aggregates across space/time.
      4. Outputs an estimated (mean, std).
    """
    def __init__(self, max_time_window, d_model=16, hidden_size=64):
        super().__init__()
        self.time_pe = PositionalEncoding(max_time_window, d_model)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.fc_agg = nn.Linear(hidden_size, hidden_size)
        
        self.head = nn.Linear(hidden_size, 2)

    def forward(self, x, time_indices=None):
        """
        x: shape (B, T, lat, lon)
        time_indices: shape (B, T) with each entry in [0..max_time_window-1]
          or None if you do not have it.

        Returns: predicted_mu, predicted_sigma
        """
        B, T, H, W = x.shape
        mask = (~torch.isnan(x)).float()  # shape (B, T, H, W)

        x_filled = torch.nan_to_num(x, nan=0.0)

        x_stacked = torch.stack([x_filled, mask], dim=2)  # (B, T, 2, H, W)

        x_reshaped = x_stacked.view(B*T, 2, H, W)  # => (B*T, 2, H, W)

        features_2d = self.conv(x_reshaped)  # => (B*T, hidden_size, H, W)

        features_2d = features_2d.mean(dim=(-1, -2))  # => (B*T, hidden_size)

        features_2d = features_2d.view(B, T, -1)  # => (B, T, hidden_size)

        feat_time = features_2d.mean(dim=1)  # => (B, hidden_size)

        feat_agg = F.relu(self.fc_agg(feat_time))  # => (B, hidden_size)
        out = self.head(feat_agg)  # => (B, 2)

        mu, log_sigma = out.split(1, dim=-1)
        sigma = F.softplus(log_sigma) + 1e-5  # ensures positivity

        return mu.squeeze(-1), sigma.squeeze(-1)

class TransfertDataModule(BaseDataModule):
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
    
    def post_fn(self):
        if self.norm_type == 'z_score':
            m, s = 0,1
            normalize = lambda item: (item - m) / s
        if self.norm_type == 'min_max':
            min_value, max_value = self.norm_stats()
            normalize = lambda item: (item - min_value) / (max_value - min_value)
        return ft.partial(ft.reduce,lambda i, f: f(i), [
            TrainingItem._make,
            lambda item: item._replace(tgt=normalize(item.tgt)),
            lambda item: item._replace(input=normalize(item.input)),
        ])
    
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

def threshold_xarray(da):
    threshold = 999
    da = xr.where(da > threshold, 0, da)
    da = xr.where(da <= 0, 0, da)
    return da

def run(trainer, train_dm, test_dm, lit_mod, ckpt=None):
    """
    Fit and test on two distinct domains.
    """
    print('-*---------------')
    if trainer.logger is not None:
        print()
        print('Logdir:', trainer.logger.log_dir)
        print()

    trainer.fit(lit_mod, datamodule=train_dm, ckpt_path=ckpt)
    #trainer.test(lit_mod, datamodule=test_dm, ckpt_path=ckpt)


class Lit4dVarNet_norm(src.models.Lit4dVarNet):
    def __init__(self, solver, rec_weight, opt_fn, sampling_rate = 1, test_metrics=None, pre_metric_fn=None, norm_stats=None, norm_type ='z_score', norm_predictor=None, persist_rw=True):
        super().__init__(solver=solver, 
              rec_weight=rec_weight,
              opt_fn=opt_fn,)
        self.test_data = None
        self._norm_stats = norm_stats
        self.opt_fn = opt_fn
        self.metrics = test_metrics or {}
        self.pre_metric_fn = pre_metric_fn or (lambda x: x)
        self.sampling_rate = sampling_rate      
        self.norm_type = norm_type
        self.norm_predictor = norm_predictor
        #self.mask = (torch.rand(1, *input_shape) > self.sampling_rate).to('cuda:0')
        print(sampling_rate)
        #self.alphaObs    = solver.obs_cost.weight1_torch
        #self.alphaReg    = solver.prior_cost.weight3_torch
        #self.alphaGrad   = solver.obs_cost.weight2_torch

    @property
    def norm_stats(self):
        if self._norm_stats is not None:
            return self._norm_stats
        elif self.trainer.datamodule is not None:
            return self.trainer.datamodule.norm_stats()
        return (0., 1.)


    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")[0]

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")[0]
    
    def on_before_optimizer_step(self, optimizer, optimizer_idx = None):
        norms = grad_norm(self.solver, norm_type=2)
        self.log_dict(norms)

    def forward(self, batch):
        if self.solver.n_step > 0:
            return self.solver(batch)
        else:
            #print(batch.input)
            return self.solver.prior_cost.forward_ae(batch.input.nan_to_num().detach().requires_grad_(True))
        
    def step(self, batch, phase=""):
        if self.training and batch.tgt.isfinite().float().mean() < 0.9:
            return None, None
        # Create a mask selecting non-NaN values
        # if self.mask_sampling_with_nan is not None:
        # Create a mask selecting non-NaN values for each element in the batch
        masked_input = batch.input.clone()
        for i in range(batch.input.size(0)):  # Assuming the first dimension is the batch size
            sr = self.sampling_rate
            if isinstance(self.sampling_rate, (list, tuple, ListConfig)) and len(self.sampling_rate) == 2:
                sr = random.uniform(self.sampling_rate[0], self.sampling_rate[1])
        
            mask = (torch.rand(batch.input[i].size()) > sr).to(batch.input.device)
            masked_input[i][mask] = float('nan')
    
        batch = batch._replace(input=masked_input)
        hat_mu, hat_std = self.norm_predictor(masked_input)
        hat_mu_4d = hat_mu.view(-1, 1, 1, 1)
        hat_std_4d = hat_std.view(-1, 1, 1, 1)
        masked_input_norm = (masked_input - hat_mu_4d) / (hat_std_4d + 1e-8)
        batch = batch._replace(input=masked_input_norm)

        tgt_mu = batch.tgt.mean()
        tgt_std = batch.tgt.std()
        tgt_norm = (batch.tgt - tgt_mu) / (tgt_std + 1e-5)
        batch = batch._replace(tgt=tgt_norm)

        loss_mu  = F.mse_loss(hat_mu,  tgt_mu.expand_as(hat_mu))
        loss_std = F.mse_loss(hat_std, tgt_std.expand_as(hat_std))
        loss_norm = 0.1*(loss_mu + loss_std)  # weight as you like
        if self.solver.n_step > 0:
            loss, out = self.base_step(batch, phase)
            grad_loss = self.weighted_mse( kfilts.sobel(out) - kfilts.sobel(batch.tgt), self.rec_weight)
            prior_cost = self.solver.prior_cost(self.solver.init_state(batch, out))
        
            self.log( f"{phase}_gloss", grad_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log( f"{phase}_prior_cost", prior_cost, prog_bar=True, on_step=False, on_epoch=True)
            self.log( f"{phase}_loss_norm", loss_norm, prog_bar=True, on_step=False, on_epoch=True)
            #weight_obs = self.solver.obs_cost.weight1_torch
            #weight_prior = self.solver.prior_cost.weight3_torch
            self.log('sampling_rate', sr, on_step=False, on_epoch=True)
            #self.log('weight obs', weight_obs , on_step=False, on_epoch=True)
            #self.log('weight prior', weight_prior,on_step=False, on_epoch=True)

            training_loss = 10 * loss + 20 * prior_cost + 5 * grad_loss + 1.0 * loss_norm
            #training_loss = 50 * loss + 1000 * grad_loss + 1.0 * prior_cost

            return training_loss, out
        
        else:
            loss, out = self.base_step(batch, phase)
            return loss, out

    def base_step(self, batch, phase=""):
        # batch = batch._replace(input = batch.input / torch.bernoulli(torch.full(batch.input.size(), self.sampling_rate)).to('cuda:0'))
        out = self(batch=batch)
        #loss = self.weighted_rel_mse(out - batch.tgt, batch.tgt, self.rec_weight)
        loss = self.weighted_mse(out - batch.tgt, self.rec_weight)
        with torch.no_grad():
            self.log(f"{phase}_mse",  loss * self.norm_stats[1]**2, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{phase}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss, out

    def configure_optimizers(self):
        return self.opt_fn(self)

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.test_data = []
        batch_input_clone = batch.input.clone()
        masked_input = batch.input.clone()
        for i in range(batch.input.size(0)):  # Assuming the first dimension is the batch size
            sr = self.sampling_rate
            if isinstance(self.sampling_rate, (list, tuple, ListConfig)) and len(self.sampling_rate) == 2:
                sr = random.uniform(self.sampling_rate[0], self.sampling_rate[1])
        
            mask = (torch.rand(batch.input[i].size()) > sr).to(batch.input.device)
            masked_input[i][mask] = float('nan')
    
        batch = batch._replace(input=masked_input)
    
        out = self(batch=batch)

        if self.norm_type == 'z_score':
            m, s = self.norm_stats
            self.test_data.append(torch.stack(
                [   batch_input_clone.cpu() * s + m,
                    batch.input.cpu() * s + m,
                    batch.tgt.cpu() * s + m,
                    out.squeeze(dim=-1).detach().cpu() * s + m,
                ],
                dim=1,
            ))

        if self.norm_type == 'min_max':
            min_value, max_value = self.norm_stats
            self.test_data.append(torch.stack(
                [   (batch_input_clone.cpu()  - min_value) / (max_value - min_value),
                    (batch.input.cpu()  - min_value) / (max_value - min_value),
                    (batch.tgt.cpu()  - min_value) / (max_value - min_value),
                    (out.squeeze(dim=-1).detach().cpu()  - min_value) / (max_value - min_value),
                ],
                dim=1,
            ))

    @property
    def test_quantities(self):
        return ['input', 'inp', 'tgt', 'out']

    def on_test_epoch_end(self):
        rec_da = self.trainer.test_dataloaders.dataset.reconstruct(
            self.test_data, self.rec_weight.cpu().numpy()
        )

        if isinstance(rec_da, list):
            rec_da = rec_da[0]

        self.test_data = rec_da.assign_coords(
            dict(v0=self.test_quantities)
        ).to_dataset(dim='v0')

        metric_data = self.test_data.pipe(self.pre_metric_fn)
        metrics = pd.Series({
            metric_n: metric_fn(metric_data) 
            for metric_n, metric_fn in self.metrics.items()
        })

        print(metrics.to_frame(name="Metrics").to_markdown())
        if self.logger:
            self.test_data.to_netcdf(Path(self.logger.log_dir) / 'test_data.nc')
            print(Path(self.trainer.log_dir) / 'test_data.nc')
            self.logger.log_metrics(metrics.to_dict())
