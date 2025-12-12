from copy import deepcopy
import xarray as xr
import einops
import functools as ft
import torch
import torch.nn as nn
import src.data
import src.models
import src.utils
import kornia.filters as kfilts
import pandas as pd
import tqdm
import numpy as np
import torch.nn.functional as F
from src.data import AugmentedDataset, BaseDataModule, XrDataset

class NormSeasonDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean_std_domain = kwargs.get('mean_std_domain', 'train')
        self.std_c = kwargs.get('std_c', 1.)
        
        self.periods = {
            'allyear': [(None, None), (None, None)],
            'midautumn': [(f'{year}-10-21', f'{year}-11-30') for year in ('2009', '2012')],
            'midwinter': [(f'{year}-01-02', f'{year}-03-13') for year in ('2010', '2013')],
            'midspring': [(f'{year}-04-30', f'{year}-06-09') for year in ('2010', '2013')],
            'midsummer': [(f'{year}-07-11', f'{year}-08-20') for year in ('2009', '2013')],
        }

    def norm_stats(self):
        if self._norm_stats is None:
            if self.norm_type == 'z_score':
                self._norm_stats = self.train_mean_std()
                print("Norm stats", self._norm_stats)
            if self.norm_type == 'min_max':
                self._norm_stats = self.min_max_norm()
                print("Norm stats", self._norm_stats)
        return self._norm_stats
    
    def norm_stats_for_period(self, period):
        if period not in self.periods:
            raise ValueError(f"Invalid period: {period}")

        norm_stats_list = []
        for period_range in self.periods[period]:
            self._norm_stats = self.norm_stats()  # replace this with the filtered data
            norm_stats_list.append(self._norm_stats)
            print("Norm stats for period", period_range, self._norm_stats)
        return norm_stats_list

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

    #trainer.fit(lit_mod, datamodule=train_dm, ckpt_path=ckpt)
    trainer.test(lit_mod, datamodule=test_dm, ckpt_path=ckpt)

class Lit4dVarNet_Norm(src.models.Lit4dVarNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(240*240*15*self.sampling_rate, 1)
        )

    @staticmethod
    def weighted_mse_mask(err, weight, mask_nan):
        err_valid = err * mask_nan[None, ...]
        # Calculate the number of valid elements
        num_valid = mask_nan.sum()
    
        if num_valid == 0:
            return torch.tensor(1000.0, device=err.device, requires_grad=True)
        weight_valid = weight * mask_nan[None, ...]

        err_w = err_valid * weight_valid[None, ...]
        non_zeros = (torch.ones_like(err) * weight[None, ...]) == 0.0
        err_num = err.isfinite() & ~non_zeros
        if err_num.sum() == 0:
            return torch.scalar_tensor(1000.0, device=err_num.device).requires_grad_()
        err_w_res = err_w.reshape(err_num.size())
        loss = F.mse_loss(err_w_res[err_num], torch.zeros_like(err_w_res[err_num]))
        return loss
    
    def step(self, batch, phase=""):
        if self.training and batch.tgt.isfinite().float().mean() < 0.9:
            return None, None
        mask = (torch.rand(batch.input.size()) > self.sampling_rate).to('cuda:0')
        # Apply the mask to the input data, setting selected values to NaN
        masked_input = batch.input.clone()
        masked_input[mask] = float('nan')
        batch = batch._replace(input = masked_input)

        mask_tgt = torch.where(batch.tgt!= 0, torch.tensor(1.), torch.tensor(0.)).to('cuda:0')

        if self.solver.n_step > 0:
            
            loss, out = self.base_step(batch, phase)

            m, s = self.norm_stats
            ecs_class_O = torch.where(batch.tgt!= 0, torch.ones_like(batch.tgt), torch.zeros_like(batch.tgt)).squeeze(dim=1).float()
            out_class_O = torch.where(out * s + m > 1, torch.ones_like(out), torch.zeros_like(out)).squeeze(dim=1).float()
            #binary_cross_entropy_loss = F.binary_cross_entropy_with_logits(ecs_class_O, out_classif_O)
            #out_class_O = self.classifier(out.view(out.size(0), -1)).view_as(out)
            binary_cross_entropy_loss = F.binary_cross_entropy_with_logits(ecs_class_O, out_class_O)

            grad_loss = self.weighted_mse_mask( kfilts.sobel(out) - kfilts.sobel(batch.tgt), self.rec_weight, mask_tgt)
            prior_cost = self.solver.prior_cost(self.solver.init_state(batch, out))
        
            self.log( f"{phase}_gloss", grad_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log( f"{phase}_prior_cost", prior_cost, prog_bar=True, on_step=False, on_epoch=True)
            self.log( f"{phase}_class_loss", binary_cross_entropy_loss, prog_bar=True, on_step=False, on_epoch=True)
            
            training_loss = 10 * loss + 20 * prior_cost + 5 * grad_loss + binary_cross_entropy_loss
            
            self.log('sampling_rate', self.sampling_rate, on_step=False, on_epoch=True)
            self.log('weight loss', 10., on_step=False, on_epoch=True)
            self.log('prior cost', 20.,on_step=False, on_epoch=True)
            self.log('grad loss', 5., on_step=False, on_epoch=True)
            #training_loss = 50 * loss + 1000 * grad_loss + 1.0 * prior_cost
            return training_loss, out
        
        else:
            loss, out = self.base_step(batch, phase)
            return loss, out
        
    def base_step(self, batch, phase=""):
        # batch = batch._replace(input = batch.input / torch.bernoulli(torch.full(batch.input.size(), self.sampling_rate)).to('cuda:0'))
        out = self(batch=batch)
        mask = torch.where(batch.tgt!= 0, torch.tensor(1.), torch.tensor(0.)).to('cuda:0')

        #loss = self.weighted_rel_mse(out - batch.tgt, batch.tgt, self.rec_weight)
        loss = self.weighted_mse_mask(out - batch.tgt, self.rec_weight, mask)
        with torch.no_grad():
            self.log(f"{phase}_mse",  loss * self.norm_stats[1]**2, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{phase}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss, out
       
    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.test_data = []
            
        batch_input_clone = batch.input.clone()
        mask = (torch.rand(batch.input.size()) > self.sampling_rate).to('cuda:0')
        # Apply the mask to the input data, setting selected values to NaN
        masked_input = batch.input.clone()
        masked_input[mask] = float('nan')
        batch = batch._replace(input = masked_input)
        out = self(batch=batch)
        class0 = torch.where(out!= 0, torch.ones_like(out), torch.zeros_like(out))
        if self.norm_type == 'z_score':
            m, s = self.norm_stats
            self.test_data.append(torch.stack(
                [   batch_input_clone.cpu() * s + m,
                    batch.input.cpu() * s + m,
                    batch.tgt.cpu() * s + m,
                    out.squeeze(dim=-1).detach().cpu() * s + m,
                    class0.squeeze(dim=-1).detach().cpu()
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

class Lit4dVarNet_NormSeason(src.models.Lit4dVarNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm_period_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(240*240*15*self.sampling_rate, 4),
            nn.Softmax(dim=1)
        )

    @staticmethod
    def weighted_mse_mask(err, weight, mask_nan):
        err_valid = err * mask_nan[None, ...]
        # Calculate the number of valid elements
        num_valid = mask_nan.sum()
    
        if num_valid == 0:
            return torch.tensor(1000.0, device=err.device, requires_grad=True)
        weight_valid = weight * mask_nan[None, ...]

        err_w = err_valid * weight_valid[None, ...]
        non_zeros = (torch.ones_like(err) * weight[None, ...]) == 0.0
        err_num = err.isfinite() & ~non_zeros
        if err_num.sum() == 0:
            return torch.scalar_tensor(1000.0, device=err_num.device).requires_grad_()
        err_w_res = err_w.reshape(err_num.size())
        loss = F.mse_loss(err_w_res[err_num], torch.zeros_like(err_w_res[err_num]))
        return loss
    
    def step(self, batch, phase=""):
        if self.training and batch.tgt.isfinite().float().mean() < 0.9:
            return None, None
        mask = (torch.rand(batch.input.size()) > self.sampling_rate).to('cuda:0')
        # Apply the mask to the input data, setting selected values to NaN
        masked_input = batch.input.clone()
        masked_input[mask] = float('nan')
        batch = batch._replace(input = masked_input)

        mask_tgt = torch.where(batch.tgt!= 0, torch.tensor(1.), torch.tensor(0.)).to('cuda:0')

        if self.solver.n_step > 0:
            
            loss, out = self.base_step(batch, phase)

            m, s = self.norm_stats
            ecs_class_O = torch.where(batch.tgt!= 0, torch.ones_like(batch.tgt), torch.zeros_like(batch.tgt)).squeeze(dim=1).float()
            out_class_O = torch.where(out * s + m > 1, torch.ones_like(out), torch.zeros_like(out)).squeeze(dim=1).float()
            #binary_cross_entropy_loss = F.binary_cross_entropy_with_logits(ecs_class_O, out_classif_O)
            #out_class_O = self.classifier(out.view(out.size(0), -1)).view_as(out)
            binary_cross_entropy_loss = F.binary_cross_entropy_with_logits(ecs_class_O, out_class_O)

            grad_loss = self.weighted_mse_mask( kfilts.sobel(out) - kfilts.sobel(batch.tgt), self.rec_weight, mask_tgt)
            prior_cost = self.solver.prior_cost(self.solver.init_state(batch, out))
        
            self.log( f"{phase}_gloss", grad_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log( f"{phase}_prior_cost", prior_cost, prog_bar=True, on_step=False, on_epoch=True)
            self.log( f"{phase}_class_loss", binary_cross_entropy_loss, prog_bar=True, on_step=False, on_epoch=True)
            
            training_loss = 10 * loss + 20 * prior_cost + 5 * grad_loss + binary_cross_entropy_loss
            
            self.log('sampling_rate', self.sampling_rate, on_step=False, on_epoch=True)
            self.log('weight loss', 10., on_step=False, on_epoch=True)
            self.log('prior cost', 20.,on_step=False, on_epoch=True)
            self.log('grad loss', 5., on_step=False, on_epoch=True)
            #training_loss = 50 * loss + 1000 * grad_loss + 1.0 * prior_cost
            return training_loss, out
        
        else:
            loss, out = self.base_step(batch, phase)
            return loss, out
        
    def base_step(self, batch, phase=""):
        # batch = batch._replace(input = batch.input / torch.bernoulli(torch.full(batch.input.size(), self.sampling_rate)).to('cuda:0'))
        out = self(batch=batch)
        mask = torch.where(batch.tgt!= 0, torch.tensor(1.), torch.tensor(0.)).to('cuda:0')

        #loss = self.weighted_rel_mse(out - batch.tgt, batch.tgt, self.rec_weight)
        loss = self.weighted_mse_mask(out - batch.tgt, self.rec_weight, mask)
        with torch.no_grad():
            self.log(f"{phase}_mse",  loss * self.norm_stats[1]**2, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{phase}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss, out
       
    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.test_data = []
            
        batch_input_clone = batch.input.clone()
        mask = (torch.rand(batch.input.size()) > self.sampling_rate).to('cuda:0')
        # Apply the mask to the input data, setting selected values to NaN
        masked_input = batch.input.clone()
        masked_input[mask] = float('nan')
        batch = batch._replace(input = masked_input)
        out = self(batch=batch)
        class0 = torch.where(out!= 0, torch.ones_like(out), torch.zeros_like(out))
        if self.norm_type == 'z_score':
            m, s = self.norm_stats
            self.test_data.append(torch.stack(
                [   batch_input_clone.cpu() * s + m,
                    batch.input.cpu() * s + m,
                    batch.tgt.cpu() * s + m,
                    out.squeeze(dim=-1).detach().cpu() * s + m,
                    class0.squeeze(dim=-1).detach().cpu()
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