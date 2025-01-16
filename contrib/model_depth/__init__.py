import xarray as xr
import numpy as np
import functools as ft
import einops
import torch
import torch.nn as nn
import collections
import src.data
import src.models
import src.utils
import kornia.filters as kfilts
import random
from omegaconf import ListConfig
from copy import deepcopy
import torch
import xarray as xr
from pathlib import Path
import pandas as pd
from src.data import AugmentedDataset, BaseDataModule, XrDataset
from collections import namedtuple

TrainingItem = namedtuple('TrainingItem', ['input', 'tgt'])

class TransfertXrDataset(XrDataset):
    def reconstruct_from_items(self, items, weight=None):
        print("items[0].shape =", items[0].shape)
        print("weight.shape =", weight.shape)
        if weight is None:
            weight = np.ones(list(self.patch_dims.values()))
        weight = weight.reshape(list(self.patch_dims.values()))
        w = xr.DataArray(weight, dims=list(self.patch_dims.keys()))

        coords = self.get_coords()

        new_dims = [f'v{i}' for i in range(len(items[0].shape) - len(coords[0].dims))]
        dims = new_dims + list(coords[0].dims)

        das = [xr.DataArray(it.numpy(), dims=dims, coords=co.coords)
               for  it, co in zip(items, coords)]

        da_shape = dict(zip(coords[0].dims, self.da.shape[-len(coords[0].dims):]))
        new_shape = dict(zip(new_dims, items[0].shape[:len(new_dims)]))

        rec_da = xr.DataArray(
                np.zeros([*new_shape.values(), *da_shape.values()]),
                dims=dims,
                coords={d: self.da[d] for d in self.patch_dims} 
        )
        count_da = xr.zeros_like(rec_da)

        for da in das:
            rec_da.loc[da.coords] = rec_da.sel(da.coords) + da * w
            count_da.loc[da.coords] = count_da.sel(da.coords) + w

        return rec_da / count_da

class TransfertDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean_std_domain = kwargs.get('mean_std_domain', 'train')
        self.std_c = kwargs.get('std_c', 1.)
    
    def train_mean_std(self, variable='tgt'):
        train_data = (
            self.input_da
            .sel(self.xrds_kw.get('domain_limits', {}))
            .sel(self.domains[self.mean_std_domain])
            .sel(variable=variable)
        )
        # shape: (component,) after mean over time, lat, lon
        means = train_data.mean(dim=('time', 'lat', 'lon'))
        stds = train_data.std(dim=('time', 'lat', 'lon')) * self.std_c

        # Return them as NumPy arrays for convenience
        return means.values, stds.values

    def min_max_norm(self, variable='tgt'):
        train_data = (
            self.input_da
            .sel(self.xrds_kw.get('domain_limits', {}))
            .sel(self.domains[self.mean_std_domain])
            .sel(variable=variable)
        )
        min_vals = train_data.min(dim=('time', 'lat', 'lon'))
        max_vals = train_data.max(dim=('time', 'lat', 'lon'))

        return min_vals.values, max_vals.values

    def post_fn(self):
        """
        Applies per-component normalization to `item.tgt` and `item.input`.
        We assume that each of these is shaped like (component, ..., ...).
        """
        if self.norm_type == 'z_score':
            means, stds = self.norm_stats()  # shape: (component,)

            def normalize(values):
                return (values - means[:, None, None]) / stds[:, None, None]

        elif self.norm_type == 'min_max':
            min_vals, max_vals = self.norm_stats()  # shape: (component,)

            def normalize(values):
                return (values - min_vals[:, None, None]) / (max_vals[:, None, None] - min_vals[:, None, None])

        else:
            def normalize(values):
                return values

        # Now return a partial function that applies your transformations
        return ft.partial(
            ft.reduce,
            lambda i, f: f(i),
            [
                TrainingItem._make,
                # normalize item.tgt
                lambda item: item._replace(tgt=normalize(item.tgt).astype(np.float32)),
                # normalize item.input
                lambda item: item._replace(input=normalize(item.input).astype(np.float32)),
            ]
        )
    
    def setup(self, stage='test'):
        post_fn = self.post_fn()
        if stage == 'fit':
            train_data = self.input_da.sel(self.domains['train'])
            train_xrds_kw = deepcopy(self.xrds_kw)
            
            self.train_ds = TransfertXrDataset(
                train_data, **train_xrds_kw, postpro_fn=post_fn,
            )
            if self.aug_kw:
                self.train_ds = AugmentedDataset(self.train_ds, **self.aug_kw)

            self.val_ds = TransfertXrDataset(
                self.input_da.sel(self.domains['val']),
                **self.xrds_kw,
                postpro_fn=post_fn,
            )
        else:
            self.test_ds = TransfertXrDataset(
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
    ).transpose('time', 'component', 'lat', 'lon').to_array()
    print(ds.shape)
    
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
    if trainer.logger is not None:
        print()
        print('Logdir:', trainer.logger.log_dir)
        print()

    trainer.fit(lit_mod, datamodule=train_dm, ckpt_path=ckpt)
    #trainer.test(lit_mod, datamodule=test_dm, ckpt_path=ckpt)
    
class Lit4dVarNet_depth(src.models.Lit4dVarNet):
    def step(self, batch, phase=""):
        if self.training and batch.tgt.isfinite().float().mean() < 0.9:
            return None, None
        batch = batch._replace(
            input=batch.input.view(batch.input.size(0), -1, batch.input.size(-2), batch.input.size(-1)),
            tgt=batch.tgt.view(batch.tgt.size(0), -1, batch.tgt.size(-2), batch.tgt.size(-1))
        )
        masked_input = batch.input.clone()
        for i in range(batch.input.size(0)):  # Assuming the first dimension is the batch size
            sr = self.sampling_rate
            if isinstance(self.sampling_rate, (list, tuple, ListConfig)) and len(self.sampling_rate) == 2:
                sr = random.uniform(self.sampling_rate[0], self.sampling_rate[1])
        
            mask = (torch.rand(batch.input[i].size()) > sr).to(batch.input.device)
            masked_input[i][mask] = float('nan')
    
        batch = batch._replace(input=masked_input)
        if self.solver.n_step > 0:

            loss, out = self.base_step(batch, phase)
            grad_loss = self.weighted_mse( kfilts.sobel(out) - kfilts.sobel(batch.tgt), self.rec_weight)
            prior_cost = self.solver.prior_cost(self.solver.init_state(batch, out))
        
            self.log( f"{phase}_gloss", grad_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log( f"{phase}_prior_cost", prior_cost, prog_bar=True, on_step=False, on_epoch=True)
            #weight_obs = self.solver.obs_cost.weight1_torch
            #weight_prior = self.solver.prior_cost.weight3_torch
            self.log('sampling_rate', sr, on_step=False, on_epoch=True)
            #self.log('weight obs', weight_obs , on_step=False, on_epoch=True)
            #self.log('weight prior', weight_prior,on_step=False, on_epoch=True)

            training_loss = 10 * loss + 20 * prior_cost + 5 * grad_loss
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
            self.log(f"{phase}_mse",  loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{phase}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss, out
    
    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.test_data = []
        batch = batch._replace(
            input=batch.input.view(batch.input.size(0), -1, batch.input.size(-2), batch.input.size(-1)),
            tgt=batch.tgt.view(batch.tgt.size(0), -1, batch.tgt.size(-2), batch.tgt.size(-1))
        )
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
            batch_input_clone_unnorm = self._denormalize_zscore(batch_input_clone, self.norm_stats)
            print(batch_input_clone_unnorm.shape)
            masked_input_unnorm      = self._denormalize_zscore(masked_input,       self.norm_stats)
            batch_tgt_unnorm        = self._denormalize_zscore(batch.tgt,          self.norm_stats)
            out_unnorm              = self._denormalize_zscore(out,                self.norm_stats)
            self.test_data.append(torch.stack(
                [   batch_input_clone_unnorm.cpu(),
                    masked_input_unnorm.cpu(),
                    batch_tgt_unnorm.cpu(),
                    out_unnorm.squeeze(dim=-1).detach().cpu(),
                ],
                dim=1,
            ))
        
        if self.norm_type == 'min_max':
            batch_input_clone_unnorm = self._denormalize_minmax(batch_input_clone, self.norm_stats)
            masked_input_unnorm      = self._denormalize_minmax(masked_input,       self.norm_stats)
            batch_tgt_unnorm        = self._denormalize_minmax(batch.tgt,          self.norm_stats)
            out_unnorm              = self._denormalize_minmax(out,                self.norm_stats)
            self.test_data.append(torch.stack(
                [   batch_input_clone_unnorm.cpu(),
                    masked_input_unnorm.cpu(),
                    batch_tgt_unnorm.cpu(),
                    out_unnorm.squeeze(dim=-1).detach().cpu(),
                ],
                dim=1,
            ))
    def _denormalize_minmax(self, tensor_4d, minmax_list):
        """
        Denormalize a 4D tensor (B, T*C, H, W) given a list of (min_c, max_c)
        for each component c.

        Args:
            tensor_4d (torch.Tensor): shape (batch, T*C, height, width)
            minmax_list (List[Tuple[float, float]]): 
                list of (min_value, max_value) for each component c

        Returns:
            denorm_5d (torch.Tensor): shape (batch, T, C, height, width)
                where each component c is mapped back to [min_c, max_c].
        """
        B, TC, H, W = tensor_4d.shape
        C = 5  # number of components
        T = TC // C           # number of timesteps
        # Reshape to (B, T, C, H, W)
        reshaped = tensor_4d.view(B, T, C, H, W)

        # For each component c, apply x * (max_c - min_c) + min_c
        for c_idx, (min_c, max_c) in enumerate(zip(minmax_list[0], minmax_list[1])):
            reshaped[:, :, c_idx, :, :] = reshaped[:, :, c_idx, :, :] * (max_c - min_c) + min_c

        return reshaped
    
    def _denormalize_zscore(self, tensor_4d, norm_list):
        """
        tensor_4d: shape (B, T*C, H, W)
        norm_list: list of (mean_c, std_c) for each component c
        Returns shape (B, T, C, H, W) with each component unnormalized.
        """
        B, TC, H, W = tensor_4d.shape
        C = 5       # e.g. 4 components
        T = TC // C              # e.g. 60 // 4 = 15
        # Reshape
        reshaped = tensor_4d.view(B, T, C, H, W)
        print(norm_list[0])
        # Loop over each component c
        for c_idx, (mean_c, std_c) in enumerate(zip(norm_list[0], norm_list[1])):
            reshaped[:, :, c_idx, :, :] = reshaped[:, :, c_idx, :, :] * std_c + mean_c
        # Reshape back to the original shape (B, T*C, H, W)
        output = reshaped#.view(B, TC, H, W)
        print('-----')
        print(output.shape)
        return output
    
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
 