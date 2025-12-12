import xarray as xr
import einops
import functools as ft
import torch
from torch.distributions import Categorical
import torch.nn as nn
import collections
import src.data
import src.models
import src.utils
import numpy as np
import kornia.filters as kfilts
import pytorch_lightning as pl


MultiModalPosTrainingItem = collections.namedtuple(
    "MultiModalPosTrainingItem", ["input", "tgt", "pos"]
)

def threshold_xarray(da):
    threshold = 10**3
    da = xr.where(da > threshold, 0, da)
    da = xr.where(da <= 0, 0, da)
    return da

def load_natl_data_pos(tgt_path, tgt_var, inp_path, inp_var, pos_path, pos_var, **kwargs):
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
    pos = (
        xr.open_dataset(pos_path)[pos_var]
        .sel(kwargs.get('domain', None))
        .sel(kwargs.get('period', None))
        #.pipe(mask)
    )
    print(xr.Dataset(
            dict(input=inp, tgt=(tgt.dims, tgt.values), pos = (pos.dims, pos.values)),
            inp.coords,
        )
        .transpose('time', 'lat', 'lon')
        .to_array())
    return (
        xr.Dataset(
            dict(input=inp, tgt=(tgt.dims, tgt.values), pos = (pos.dims, pos.values)),
            inp.coords,
        )
        .transpose('time', 'lat', 'lon')
        .to_array()
    )

def cosanneal_lr_adamw(lit_mod, lr, T_max, weight_decay=0.):
    opt = torch.optim.AdamW(
        [
            {'params': lit_mod.batch.pos.parameters(), 'lr': lr},
        ], weight_decay=weight_decay
    )
    return {
        'optimizer': opt,
        'lr_scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=T_max,
        ),
    }

class MultiModalPosDataModule(src.data.BaseDataModule):
    def post_fn(self):        
        normalize_ecs = lambda item: (item - self.norm_stats()[0]) / self.norm_stats()[1]
        return ft.partial(
            ft.reduce,
            lambda i, f: f(i),
            [
                MultiModalPosTrainingItem._make,
                lambda item: item._replace(tgt=normalize_ecs(item.tgt)),
                lambda item: item._replace(input=normalize_ecs(item.input)),
                lambda item: item._replace(pos=(item.pos)),
                #lambda item: item._replace(pos=torch.nn.functional.gumbel_softmax(Categorical(prob= item.pos), tau=1, hard=False, dim=-1)),

            ],
        )
    
class GumbelSoftmaxObs(pl.LightningModule):
    def __init__(self, pos, sampling_rate, model = None):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.pos = pos
        self.model = model

    def forward(self, input, tgt):
        # Ensure that `pos` is the same shape as `input`.
        assert input.shape == self.pos.shape, "Input and pos must be the same shape"
        # Create a mask of the same size as the input tensor, filled with values
        # sampled from a Bernoulli distribution with probabilities given by `self.pos`.
        mask = torch.bernoulli(self.pos).bool()
        # Create an observation tensor of the same size as the input tensor, filled with NaNs.
        obs = torch.full(input.shape, np.nan)
        # Place the values from the input tensor that were selected by the mask into the observation tensor.
        obs[mask] = input[mask]
        if self.model is not None:
            out = self.model(obs)
        rmse_loss = torch.sqrt(torch.mean((out - tgt) ** 2))

        return obs, rmse_loss
    
#class GumbelSoftmaxObs(pl.LightningModule):
#    def __init__(self, pos, tau=1.0):
#        super().__init__()
#        self.pos = pos
#        self.tau = tau
#
#    def forward(self, input):
#        assert input.shape == self.pos.shape, "Input and pos must be the same shape"
#        # Sample from Gumbel(0, 1)
#        gumbel_noise = -torch.empty_like(self.pos).exponential_().log()
#        # Compute the Gumbel-Softmax sample
#        gumbel_softmax_sample = F.softmax((self.pos.log() + gumbel_noise) / self.tau, dim=-1)
#        # Create an observation tensor of the same size as the input tensor, filled with NaNs.
#        obs = torch.full(input.shape, np.nan)
#        # Place the values from the input tensor that were selected by the mask into the observation tensor.
#        obs[gumbel_softmax_sample > 0.5] = input[gumbel_softmax_sample > 0.5]
#        return obs

class Lit4dVarNetPos(src.models.Lit4dVarNet):
    def __init__(self, solver, rec_weight, opt_fn,  sampling_rate = 1, test_metrics=None, pre_metric_fn=None, norm_stats=None, norm_type ='z_score', persist_rw=True):
        super().__init__(solver, rec_weight, opt_fn, sampling_rate, test_metrics, pre_metric_fn, norm_stats, norm_type, persist_rw)
        #self.temperature = nn.parameter.Parameter(torch.tensor(1.0))

    def step(self, batch, phase=""):
        if self.training and batch.tgt.isfinite().float().mean() < 0.9:
            return None, None
        
        probabilities = torch.nn.functional.softmax(batch.pos, dim=-1)
        num_samples = int(self.sampling_rate * probabilities.numel())
        indices = torch.multinomial(probabilities.view(-1), num_samples)
        mask = torch.zeros_like(probabilities, dtype=torch.bool).view(-1)
        mask[indices] = True
        mask = mask.view(probabilities.shape)
        masked_input = torch.where(mask, torch.tensor(float('nan')), batch.input)
        batch = batch._replace(input = masked_input)


        if self.solver.n_step > 0:

            loss, out = self.base_step(batch, phase)
            grad_loss = self.weighted_mse( kfilts.sobel(out) - kfilts.sobel(batch.tgt), self.rec_weight)
            prior_cost = self.solver.prior_cost(self.solver.init_state(batch, out))
        
            self.log( f"{phase}_gloss", grad_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log( f"{phase}_prior_cost", prior_cost, prog_bar=True, on_step=False, on_epoch=True)
            training_loss = 10 * loss + 20 * prior_cost + 5 * grad_loss
            
            self.log('sampling_rate', self.sampling_rate, on_step=False, on_epoch=True)
            self.log('weight loss', 10., on_step=False, on_epoch=True)
            self.log('prior cost', 20.,on_step=False, on_epoch=True)
            self.log('grad loss', 5., on_step=False, on_epoch=True)
            #training_loss = 50 * loss + 1000 * grad_loss + 1.0 * prior_cost
            return training_loss, out
        
        else:
            loss, out = self.base_step(batch, phase)
            return loss, out
        
    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.test_data = []
            
        probabilities = torch.nn.functional.softmax(batch.pos, dim=-1)
        num_samples = int(self.sampling_rate * probabilities.numel())
        indices = torch.multinomial(probabilities.view(-1), num_samples)
        mask = torch.zeros_like(probabilities, dtype=torch.bool).view(-1)
        mask[indices] = True
        mask = mask.view(probabilities.shape)
        masked_input = torch.where(mask, torch.tensor(float('nan')), batch.input)
        batch = batch._replace(input = masked_input)

        out = self(batch=batch)

        if self.norm_type == 'z_score':
            m, s = self.norm_stats
            self.test_data.append(torch.stack(
                [   
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
