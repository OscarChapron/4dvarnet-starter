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
from omegaconf import ListConfig, DictConfig
from copy import deepcopy
from pathlib import Path
import pandas as pd
import torch.nn.functional as F
from src.data import AugmentedDataset, BaseDataModule, XrDataset
from src.utils import get_constant_crop
from collections import namedtuple
from contrib import transfert
from typing import Optional
torch.set_float32_matmul_precision('high')
TrainingItem = namedtuple('TrainingItem', ['input', 'tgt'])

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

# ---------------------------------------------------------------------
# class Lit4dVarNetCoarse(transfert.Lit4dVarNet_Fasc):
#     """
#     Combines the stochastic-masking variant (Lit4dVarNet_Fasc) with a
#     coarse-grid reconstruction loss.  All keyword arguments understood
#     by Lit4dVarNet_Fasc –– sampling_rate, norm_type, etc. –– can still
#     be passed transparently.
#     """

#     def __init__(
#         self,
#         *args,
#         down_factor: int = 4,            # 1/20° ➜ 1/5°  (set 1 to disable)
#         pool_mode:   str = "avg",        # 'avg' | 'bilinear'
#         **kwargs,                        # <- sampling_rate, norm_type, …
#     ):
#         super().__init__(*args, **kwargs)
#         self.down_factor = int(max(down_factor, 1))
#         if self.down_factor == 1:
#             self.down = lambda x: x
#         elif pool_mode == "avg":
#             f = self.down_factor
#             self.down = lambda x: F.avg_pool2d(x, f, f)
#         elif pool_mode == "bilinear":
#             scale = 1.0 / self.down_factor
#             self.down = lambda x: F.interpolate(
#                 x, scale_factor=scale, mode="bilinear", align_corners=False
#             )
#         else:
#             raise ValueError(f"Unknown pool_mode={pool_mode!r}")

#     def step(self, batch, phase=""):
#         if self.training and batch.tgt.isfinite().float().mean() < 0.9:
#             return None, None

#         # stochastic masking
#         batch = batch._replace(input=self._apply_mask(batch.input))

#         # run parent loss
#         if self.solver.n_step > 0:
#             total, out   = self.base_step(batch, phase)
#             return total, out

#         return self.base_step(batch, phase)
    
#     def base_step(self, batch, phase=""):
#         out_full = self(batch=batch)      
#         tgt_full = batch.tgt
#         rw_full  = self.rec_weight

#         out = self.down(out_full)
#         tgt = self.down(tgt_full)
#         rw  = self.down(rw_full)

#         rec_loss  = self.weighted_mse(out - tgt, rw)
#         grad_loss = self.weighted_mse(
#             kfilts.sobel(out) - kfilts.sobel(tgt),
#             rw,
#         )
        
#         if self.solver.n_step > 0:          # cost irrelevant for AE mode
#             prior_cost = self.solver.prior_cost(
#                 self.solver.init_state(batch, out_full)
#             )
#         else:
#             prior_cost = torch.tensor(0., device=out_full.device)
#         # metrics at coarse grid
#         self.log(f"{phase}_loss",
#                  rec_loss,
#                  prog_bar=True, on_step=False, on_epoch=True)
#         self.log(f"{phase}_mse",
#                  1e4 * rec_loss * self.norm_stats[1] ** 2,
#                  prog_bar=True, on_step=False, on_epoch=True)
#         self.log(f"{phase}_gloss",
#                  grad_loss,
#                  prog_bar=True, on_step=False, on_epoch=True)
#         self.log(f"{phase}_prior_cost",
#                  prior_cost,
#                  prog_bar=True, on_step=False, on_epoch=True)
#         total = 10 * rec_loss + 5 * grad_loss + 20 * prior_cost
#         return total, out_full   # <-- keep full-res field for caller


# class Lit4dVarNetCoarse(transfert.Lit4dVarNet_Fasc):
#     def __init__(
#         self,
#         *args,
#         down_factor: int = 16,
#         **kwargs
#     ):
#         super().__init__(*args, **kwargs)
#         self.down_factor = down_factor
#         self.in_channels = 15

#         # Trainable downsampling layer
#         self.down = nn.Conv2d(
#             self.in_channels, self.in_channels,
#             kernel_size=3, stride=down_factor, padding=1, bias=False
#         )

#         # Trainable upsampling layer
#         self.up = nn.ConvTranspose2d(
#             self.in_channels, self.in_channels,
#             kernel_size=4, stride=down_factor, padding=1, bias=False
#         )

#     def forward(self, batch, phase=""):
#         """
#         Override forward to insert downsample → model → upsample logic.
#         """
#         x = batch.input
#         tgt = batch.tgt
#         x_coarse = self.down(x)
#         tgt_coarse = self.down(tgt)
#         batch_coarse = batch._replace(input=x_coarse, tgt=tgt_coarse)  # shape: (B, C, H//f, W//f)
#         pred_coarse = self.solver(batch_coarse)      # (B, C, H//f, W//f)
#         pred_full = self.up(pred_coarse)        # (B, C, H, W)
#         return pred_full

#     def step(self, batch, phase=""):
#         if self.training and batch.tgt.isfinite().float().mean() < 0.9:
#              return None, None
#         # Apply masking if needed (defined in parent class)
#         #batch = batch._replace(input=self._apply_mask(batch.input))

#         # Forward pass with downsampled inference and upsampled output
#         out_full = self(batch=batch)               # Up to full resolution
#         tgt_full = batch.tgt
#         rw_full  = self.rec_weight

#         # Compute full-res loss
#         rec_loss  = self.weighted_mse(out_full - tgt_full, rw_full)
#         grad_loss = self.weighted_mse(
#             kfilts.sobel(out_full) - kfilts.sobel(tgt_full),
#             rw_full,
#         )

#         if self.solver.n_step > 0:
#             prior_cost = self.solver.prior_cost(
#                 self.solver.init_state(batch, out_full)
#             )
#         else:
#             prior_cost = torch.tensor(0., device=out_full.device)

#         # Logging (inherited from parent)
#         self.log(f"{phase}_loss", rec_loss, prog_bar=True)
#         self.log(f"{phase}_mse", 1e4 * rec_loss * self.norm_stats[1]**2, prog_bar=True)
#         self.log(f"{phase}_gloss", grad_loss, prog_bar=True)
#         self.log(f"{phase}_prior_cost", prior_cost, prog_bar=True)

#         total_loss = 10 * rec_loss + 5 * grad_loss + 20 * prior_cost
#         return total_loss, out_full


class LatentDecoderMR(torch.nn.Module):
    def __init__(self, dim_state, dim_latent, channel_dims,scale_factor,interp_mode='linear',w_dx = None):
        super().__init__()
        self.dim_state    = dim_state
        self.dim_latent   = dim_latent
        self.channel_dims = channel_dims
        self.scale_factor = scale_factor
        self.interp_mode  = interp_mode
        if w_dx is not None :  self.w_dx =  w_dx 
        else: 
            self.w_dx = 1.

        self.decode_residual = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=dim_state+dim_latent,
                out_channels=channel_dims,
                padding="same",
                kernel_size=1,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=channel_dims,
                out_channels=dim_state,
                padding="same",
                kernel_size=1,
            ),
        )

    def forward(self, x):
        x = x.nan_to_num()

        x_up = torch.nn.functional.interpolate(x,scale_factor=self.scale_factor,mode=self.interp_mode)
        dx   = self.decode_residual(x_up)

        return x_up[:,:self.dim_state,:,:] + self.w_dx * dx 


class LatentEncoderMR(torch.nn.Module):
    def __init__(self, dim_state, dim_latent, channel_dims,scale_factor):
        super().__init__()
        self.dim_state    = dim_state
        self.dim_latent   = dim_latent
        self.channel_dims = channel_dims
        self.scale_factor = scale_factor

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=dim_state,
                out_channels=2*channel_dims,
                padding="same",
                kernel_size=3,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=2*channel_dims,
                out_channels=channel_dims,
                padding="same",
                kernel_size=3,
            ),
            torch.nn.AvgPool2d((scale_factor,scale_factor)),
            torch.nn.Conv2d(
                in_channels=channel_dims,
                out_channels=2*channel_dims,
                padding="same",
                kernel_size=3,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=2*channel_dims,
                out_channels=dim_latent,
                padding="same",
                kernel_size=3,
            ),
        )

    def forward(self, x):
        x = x.nan_to_num()
        
        dx_latent = self.encoder(x)
        x_latent  = torch.nn.functional.avg_pool2d(x,self.scale_factor)

        return torch.cat((x_latent,dx_latent),dim=1) 


class GradSolverWithLatent(transfert.GradSolver_Fasc):
    """
    A gradient-based solver for optimization in 4D-VarNet.

    Attributes:
        prior_cost (nn.Module): The prior cost function.
        obs_cost (nn.Module): The observation cost function.
        grad_mod (nn.Module): The gradient modulation model.
        n_step (int): Number of optimization steps.
        lr_grad (float): Learning rate for gradient updates.
        lbd (float): Regularization parameter.
    """

    def __init__(self, prior_cost, obs_cost, grad_mod, latent_decoder, latent_encoder, n_step, lr_grad=0.2, lbd=1.0, std_latent_init=0., **kwargs):
        """
        Initialize the GradSolver.

        Args:
            prior_cost (nn.Module): The prior cost function.
            obs_cost (nn.Module): The observation cost function.
            grad_mod (nn.Module): The gradient modulation model.
            n_step (int): Number of optimization steps.
            lr_grad (float, optional): Learning rate for gradient updates. Defaults to 0.2.
            lbd (float, optional): Regularization parameter. Defaults to 1.0.
        """
        super().__init__(prior_cost, obs_cost, grad_mod, n_step, lr_grad,**kwargs)

        self.latent_decoder  = latent_decoder
        self.latent_encoder  = latent_encoder
        self.std_latent_init = torch.nn.Parameter(torch.Tensor([std_latent_init]),requires_grad=True)
        self.lbd = lbd 
    def init_latent_from_state(self,x):
        # initialization using average-pooled obs inputs
        # for the coarse-scale component
        x = x.nan_to_num().detach()
        m = 1. - torch.isnan( x ).float()
        
        x = torch.nn.functional.avg_pool2d(x,self.latent_decoder.scale_factor)
        m = torch.nn.functional.avg_pool2d(m.float(),self.latent_decoder.scale_factor)
        x = x / (m + 1e-8)

        # random initialisation for the latent representation
        size = [x.shape[0], self.latent_decoder.dim_latent, *x.shape[-2:]]
        latent_state_init = self.std_latent_init * torch.randn(size,device=x.device)

        return torch.cat( (x,latent_state_init) , dim = 1)

    def init_state(self, batch, x_init=None):
        """
        Initialize the state for optimization.

        Args:
            batch (dict): Input batch containing data.
            x_init (torch.Tensor, optional): Initial state. Defaults to None.

        Returns:
            torch.Tensor: Initialized state.
        """
        if x_init is not None:
            return x_init

        # initialization using average-pooled obs inputs
        # for the coarse-scale component
        x_init_ = self.init_latent_from_state( batch.input)

        return x_init_.detach().requires_grad_(True)

    def solver_step(self, state, batch, step):
        """
        Perform a single optimization step.

        Args:
            state (torch.Tensor): Current state.
            batch (dict): Input batch containing data.
            step (int): Current optimization step.

        Returns:
            torch.Tensor: Updated state.
        """

        var_cost = self.prior_cost(state) + self.lbd**2 * self.obs_cost(self.latent_decoder(state), batch)
        grad = torch.autograd.grad(var_cost, state, create_graph=True)[0]

        gmod = self.grad_mod(grad)
        state_update = (
            1 / (step + 1) * gmod
            + self.lr_grad * (step + 1) / self.n_step * grad
        )

        return state - state_update

    def forward(self, batch):
        """
        Perform the forward pass of the solver.

        Args:
            batch (dict): Input batch containing data.

        Returns:
            torch.Tensor: Final optimized state.
        """
        with torch.set_grad_enabled(True):
            state = self.init_state(batch)
            self.grad_mod.reset_state(state) #batch.input)

            for step in range(self.n_step):
                state = self.solver_step(state, batch, step=step)
                if not self.training:
                    state = state.detach().requires_grad_(True)

            #if not self.training:
            #    state = self.prior_cost.forward_ae(state)

        #print(self.latent_decoder(state).shape)

        return self.latent_decoder(state),state # apply decoder from latent representation

class Lit4dVarNetIgnoreNaNLatent(transfert.Lit4dVarNet_Fasc):
    def __init__(self,  
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

    def loss_mse(self,batch,out,phase):
        loss =  self.weighted_mse(out - batch.tgt,
            self.rec_weight,
        )

        grad_loss =  self.weighted_mse(
            kfilts.sobel(out) - kfilts.sobel(batch.tgt),
            self.rec_weight,
        )

        return loss, grad_loss

    def step(self, batch, phase):
        if self.training and batch.tgt.isfinite().float().mean() < 0.5:
            return None, None
        
        # stochastic masking
        batch = batch._replace(input=self._apply_mask(batch.input))
        
        loss, out = self.base_step(batch, phase)
        grad_loss = self.weighted_mse(
            kfilts.sobel(out) - kfilts.sobel(batch.tgt),
            self.rec_weight,
        )

        self.log(
            f"{phase}_gloss",
            grad_loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,  # sync_dist=True,
        )

        loss_mse = self.loss_mse(batch,out,phase)
        training_loss = 10 * loss_mse[0] + 20 * loss_mse[1] 

        # log
        self.log(
            f"{phase}_gloss",
            loss_mse[1],
            prog_bar=False,
            on_step=False,
            on_epoch=True,  # sync_dist=True,
        )

        return training_loss, out

    def base_step(self, batch, phase):
        out = self(batch=batch)
        if isinstance(out, tuple):
            out, _ = out
        loss = self.weighted_mse(out - batch.tgt, self.rec_weight)

        with torch.no_grad():
            self.log(
                f"{phase}_mse",
                10000 * loss * self.norm_stats[1] ** 2,
                prog_bar=True,
                on_step=False,
                on_epoch=True,  # sync_dist=True,
            )
            self.log(
                f"{phase}_loss",
                loss,
                prog_bar=False,
                on_step=False,
                on_epoch=True,  # sync_dist=True,
            )
        return loss, out
