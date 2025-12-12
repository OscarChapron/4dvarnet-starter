import xarray as xr
import numpy as np
import functools as ft
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
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

TrainingItemLRGT = namedtuple("TrainingItemLRGT", ["input", "tgt_lr"])

class Lit4dVarNet_MR_LRGT_HRObs(transfert.Lit4dVarNet_Fasc):
    """
    Multi-resolution training head:
      - LR anchor with dense LR GT (tgt_lr) to stabilize large scales
      - HR observation consistency (value + gradient) at observed pixels only
      - Optional latent prior via GradSolverWithLatent

    Expect self.solver to be a GradSolverWithLatent returning (out_hr, latent_state).
    """

    def __init__(self,  # HR obs losses (on observed pixels only)
        w_hr_obs_mse: float = 1.0,
        w_hr_obs_grad: float = 0.2,
        # LR anchor losses (dense LR GT)
        w_lr_mse: float = 1.0,
        w_lr_grad: float = 0.2,
        # latent prior
        w_prior: float = 0.1,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.w_hr_obs_mse = w_hr_obs_mse
        self.w_hr_obs_grad = w_hr_obs_grad
        self.w_lr_mse = w_lr_mse
        self.w_lr_grad = w_lr_grad
        self.w_prior = w_prior
        self.lr_scale = int(self.solver.latent_decoder.scale_factor)

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
        super().__init__(prior_cost, obs_cost, grad_mod, n_step, lr_grad, lbd,**kwargs)

        self.latent_decoder  = latent_decoder
        self.latent_encoder  = latent_encoder
        self.std_latent_init = torch.nn.Parameter(torch.Tensor([std_latent_init]),requires_grad=True)

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

        print(self.latent_decoder(state).shape)

        return self.latent_decoder(state),state # apply decoder from latent representation
    
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