import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
import kornia.filters as kfilts
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities import grad_norm

class GradUnetSolver(nn.Module):
    def __init__(self, prior_cost, obs_cost, grad_mod, n_step, weight_obs = 1., weight_prior = 1., lr_grad=0.2, **kwargs):
        super().__init__()
        self.prior_cost = prior_cost
        self.obs_cost = obs_cost
        self.grad_mod = grad_mod

        self.n_step = n_step
        self.lr_grad = lr_grad

        self._grad_norm = None
    
        self.weight_obs_torch = torch.nn.Parameter(torch.tensor(weight_obs), requires_grad = True)
        self.weight_prior_torch = torch.nn.Parameter(torch.tensor(weight_prior), requires_grad = True)

        def _apply_kaiming(module):
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(module.weight)

        def _apply_xavier(module):
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)

        self.prior_cost.apply(_apply_kaiming)
        self.grad_mod.apply(_apply_xavier)

    def init_state(self, batch, x_init=None):
        if x_init is not None:
            return x_init

        return batch.input.nan_to_num().detach().requires_grad_(True)
    
    def solver_step(self, state, batch, step):
        var_cost = self.prior_cost.weight3_torch * self.prior_cost(state) +  self.obs_cost.weight1_torch * self.obs_cost(state, batch)
        grad = torch.autograd.grad(var_cost, state, create_graph=True)[0]
        gmod = self.grad_mod(grad)
        state_update = (
            1 / (step + 1) * gmod
                + self.lr_grad * (step + 1) / self.n_step * grad
        )
        return state - state_update

class GradSolver(nn.Module):
    def __init__(self, prior_cost, obs_cost, grad_mod, n_step, weight_obs = 1., weight_prior = 1., lr_grad=0.2, **kwargs):
        super().__init__()
        self.prior_cost = prior_cost
        self.obs_cost = obs_cost
        self.grad_mod = grad_mod

        self.n_step = n_step
        self.lr_grad = lr_grad

        self._grad_norm = None
    
        self.weight_obs_torch = torch.nn.Parameter(torch.tensor(weight_obs), requires_grad = True)
        self.weight_prior_torch = torch.nn.Parameter(torch.tensor(weight_prior), requires_grad = True)

        def _apply_kaiming(module):
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(module.weight)

        def _apply_xavier(module):
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)

        self.prior_cost.apply(_apply_kaiming)
        self.grad_mod.apply(_apply_xavier)
            
    def init_state(self, batch, x_init=None):
        if x_init is not None:
            return x_init

        return batch.input.nan_to_num().detach().requires_grad_(True)
    
    def solver_step(self, state, batch, step):
        var_cost = self.prior_cost.weight3_torch * self.prior_cost(state) +  self.obs_cost.weight1_torch * self.obs_cost(state, batch)
        grad = torch.autograd.grad(var_cost, state, create_graph=True)[0]
        gmod = self.grad_mod(grad)
        state_update = (
            1 / (step + 1) * gmod
                + self.lr_grad * (step + 1) / self.n_step * grad
        )
        return state - state_update

    def forward(self, batch):
        with torch.set_grad_enabled(True):
            state = self.init_state(batch)
            self.grad_mod.reset_state(batch.input)

            for step in range(self.n_step):
                state = self.solver_step(state, batch, step=step)
                if not self.training:
                    state = state.detach().requires_grad_(True)

            if not self.training:
                state = self.prior_cost.forward_ae(state)
        return state


class ConvLstmGradModel(nn.Module):
    def __init__(self, dim_in, dim_hidden, kernel_size=3, dropout=0.1, downsamp=None):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.gates = torch.nn.Conv2d(
            dim_in + dim_hidden,
            4 * dim_hidden,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.conv_out = torch.nn.Conv2d(
            dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.dropout = torch.nn.Dropout(dropout)
        self._state = []
        self.down = nn.AvgPool2d(downsamp) if downsamp is not None else nn.Identity()
        self.up = (
            nn.UpsamplingBilinear2d(scale_factor=downsamp)
            if downsamp is not None
            else nn.Identity()
        )

    def reset_state(self, inp):
        size = [inp.shape[0], self.dim_hidden, *inp.shape[-2:]]
        self._grad_norm = None
        self._state = [
            self.down(torch.zeros(size, device=inp.device)),
            self.down(torch.zeros(size, device=inp.device)),
        ]

    def forward(self, x):
        if self._grad_norm is None:
            self._grad_norm = (x**2).mean().sqrt()
        x =  x / self._grad_norm
        hidden, cell = self._state
        x = self.dropout(x)
        x = self.down(x)
        gates = self.gates(torch.cat((x, hidden), 1))

        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        in_gate, remember_gate, out_gate = map(
            torch.sigmoid, [in_gate, remember_gate, out_gate]
        )
        cell_gate = torch.tanh(cell_gate)

        cell = (remember_gate * cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        self._state = hidden, cell
        out = self.conv_out(hidden)
        out = self.up(out)
        return out


class UNet_LSTM(nn.Module):
    def __init__(self, dim_in, dim_hidden, output_dim, kernel_size=3, dropout=0.1, bidirectional=True):
        super(UNet_LSTM, self).__init__()
        self.hidden_dim = dim_hidden

        # UNet Encoder
        self.enc1 = self.conv_block(dim_in, dim_hidden)
        self.enc2 = self.conv_block(dim_hidden, dim_hidden * 2)
        self.enc3 = self.conv_block(dim_hidden * 2, dim_hidden * 4)

        # LSTM layer for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=dim_hidden * 4, hidden_size=dim_hidden, batch_first=True, bidirectional=bidirectional
        )

        # UNet Decoder
        self.dec3 = self.conv_block(dim_hidden * 4, dim_hidden * 2)
        self.dec2 = self.conv_block(dim_hidden * 2, dim_hidden)
        self.dec1 = nn.Conv2d(dim_hidden, output_dim, kernel_size=1)

        self.dropout = nn.Dropout(dropout)
        self._state = None

    def conv_block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # UNet Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(nn.MaxPool2d(2)(enc1))
        enc3 = self.enc3(nn.MaxPool2d(2)(enc2))

        # Flatten and reshape for LSTM input
        batch_size, C, H, W = enc3.size()
        enc3_flattened = enc3.view(batch_size, C, -1).permute(0, 2, 1)

        # LSTM layer
        lstm_out, _ = self.lstm(enc3_flattened)

        # Reshape back to spatial dimensions
        lstm_out = lstm_out.permute(0, 2, 1).view(batch_size, -1, H, W)

        # UNet Decoder
        dec3 = self.dec3(torch.cat([nn.Upsample(scale_factor=2)(lstm_out), enc2], dim=1))
        dec2 = self.dec2(torch.cat([nn.Upsample(scale_factor=2)(dec3), enc1], dim=1))
        dec1 = self.dec1(dec2)

        return dec1
    