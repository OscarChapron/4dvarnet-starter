import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import xarray as xr
import numpy as np
import functools as ft
import einops
import collections
import src.data
import src.models
import src.utils
import src.unet
import kornia.filters as kfilts
import random
from omegaconf import ListConfig, DictConfig
from copy import deepcopy
from pathlib import Path
import pandas as pd
from src.data import AugmentedDataset, BaseDataModule, XrDataset
from src.utils import get_constant_crop
from collections import namedtuple
from contrib import transfert
from types import SimpleNamespace
from typing import Sequence, Union
from dataclasses import dataclass, field
torch.set_float32_matmul_precision('high')
import itertools  
TrainingItem = namedtuple('TrainingItem', ['input', 'tgt'])

class DepthUNet(nn.Module):
    """
    UNet for combining reconstructions across depth levels.
    Input: (B, D*T*C, H, W) where D is depth levels
    Output: (B, T*C, H, W) - final reconstruction
    """
    def __init__(self, in_channels: int, out_channels: int, depth_levels: int):
        super().__init__()
        self.depth_levels = depth_levels
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)
        
        # Decoder
        self.dec4 = self._upconv_block(1024, 512)
        self.dec3 = self._upconv_block(512, 256)
        self.dec2 = self._upconv_block(256, 128)
        self.dec1 = self._upconv_block(128, 64)
        
        # Final layer
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def _upconv_block(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        
        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))
        
        # Decoder with skip connections
        d4 = self.dec4(b) + e4
        d3 = self.dec3(d4) + e3
        d2 = self.dec2(d3) + e2
        d1 = self.dec1(d2) + e1
        
        return self.final(d1)
    
class Lit4dVarNet_MultiDepth(transfert.Lit4dVarNet_Fasc):
    """
    Enhanced 4DVarNet that processes each depth level separately 
    then combines them using a UNet
    """

    def __init__(self, depth_levels: int = 3, use_unet: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.depth_levels = depth_levels
        self.use_unet = use_unet
        
        if self.use_unet:
            # UNet to combine depth reconstructions
            # Input: all depth reconstructions concatenated
            # Output: final combined reconstruction
            in_channels = depth_levels * kwargs.get('time_steps', 15) 
            out_channels = depth_levels * kwargs.get('time_steps', 15) 

            self.depth_unet = DepthUNet(
                in_channels=in_channels,
                out_channels=out_channels, 
                depth_levels=depth_levels
            )
    def _mask_input(self, x5d):
        """Apply spatial masking with sampling_rate; x5d: (B, D, T, H, W)."""
        B, D, T, H, W = x5d.shape
        device = x5d.device

        # draw per-sample sampling rates if a range is provided
        sr = self.sampling_rate
        if isinstance(sr, (list, tuple)) and len(sr) == 2:
            sr_vals = torch.empty(B, device=device).uniform_(float(sr[0]), float(sr[1]))
        else:
            sr_vals = torch.full((B,), float(sr), device=device)

        # vectorized spatial mask (True == masked to NaN)
        rnd = torch.rand(B, 1, 1, H, W, device=device)
        mask2d = rnd > sr_vals.view(B, 1, 1, 1, 1)  # shape (B,1,1,H,W)
        mask = mask2d.expand(B, D, T, H, W)
        x = x5d.clone()
        x[mask] = float("nan")
        return x
    
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
        means, stds = minmax_list
        C = len(means)                
        assert TC % C == 0, "TC is not divisible by C"
        T = TC // C
        # Reshape to (B, T, C, H, W)
        reshaped = tensor_4d.view(B, T, C, H, W)

        # For each component c, apply x * (max_c - min_c) + min_c
        for c_idx, (min_c, max_c) in enumerate(zip(minmax_list[0], minmax_list[1])):
            reshaped[:, :, c_idx, :, :] = reshaped[:, :, c_idx, :, :] * (max_c - min_c) + min_c

        return reshaped
    
    def _denormalize_zscore(self, tensor_5d, norm_list):
        """
        tensor_4d:  (B, T*C, H, W)
        norm_list: (means, stds)  length = C
        returns    (B, T, C, H, W)
        """
        means, stds = norm_list
        for c_idx, (mean_c, std_c) in enumerate(zip(means, stds)):
            tensor_5d[:, :, c_idx, :, :] = tensor_5d[:, :, c_idx, :, :] * std_c + mean_c
        return tensor_5d
    
    def forward(self, batch):
        """
        Process each depth level through 4DVarNet, then combine with UNet
        
        batch.input shape: (B, D, T, H, W) where D is depth levels
        batch.tgt shape: (B, D, T, H, W)
        """
        B, T, D, H, W = batch.input.shape
        
        # Process each depth level separately
        depth_reconstructions = []
        
        for d in range(D):
            # Extract data for this depth level
            depth_batch = batch._replace(
                input=batch.input[:, :, d, :, :],  # (B, D, T, H, W)
                tgt=batch.tgt[:, :, d, :, :] # (B, D, T, H, W)
            )
            # Apply 4DVarNet to this depth level
            depth_recon = super().forward(depth_batch)  # (B, T, H, W)
            depth_reconstructions.append(depth_recon)
        
        if not self.use_unet:
            # Simple averaging across depth levels
            return torch.stack(depth_reconstructions, dim=2) # (B, D, T, H, W)
        
        # Combine depth reconstructions using UNet
        # Stack all depth reconstructions: (B, D*T, H, W)
        combined_input = torch.cat(depth_reconstructions, dim=2)

        # Apply UNet to get final reconstruction
        final_recon = self.depth_unet(combined_input)  # (B, D, T, H, W)
        final_recon_unet = final_recon.view(B, T, D, H, W)  # Reshape to (B, D, T, H, W)
        return final_recon_unet

    def step(self, batch, phase=""):
        """Modified step function to handle depth dimension"""
        if self.training and batch.tgt.isfinite().float().mean() < 0.9:
            return None, None
            
        # Ensure proper shape handling for depth dimension
        if batch.input.dim() == 5:  # (B, D, T, H, W)
            B, T, D, H, W = batch.input.shape
            # Keep depth dimension for processing
            pass
        else:  # (B, D, T, H, W) - add depth dimension
            B, TD, H, W = batch.input.shape
            D = len(self.norm_stats[0])
            T = TD // D
            batch = batch._replace(
                input=batch.input.contiguous().view(B, T, D, H, W),  # (B, D, T, H, W)
                tgt=batch.tgt.contiguous().view(B, T, D, H, W)  # (B, D, T, H, W)
            )
        
        # Apply masking to each depth level
        masked_input = batch.input.clone()
        
        for i in range(B):
            sr = self.sampling_rate
            if isinstance(sr, (list, tuple)) and len(sr) == 2:
                sr = random.uniform(sr[0], sr[1])
            
            # Create spatial mask and apply to all depth levels
            spatial_mask = (torch.rand((H, W), device=batch.input.device) > sr)
            for d in range(D):
                for t in range(T):
                    masked_input[i, t, d][spatial_mask] = float("nan")
        
        batch = batch._replace(input=masked_input)
        
        # Forward pass
        out = self(batch)  # (B, T, D, H, W)
        target = batch.tgt  # (B, T, D, H, W)
        
        out_4d = out.view(B, T * D, H, W)
        target_4d = target.view(B, T * D, H, W)

        loss = self.weighted_mse(err=out_4d - target_4d, weight=self.rec_weight)
        self.log(f"{phase}_mse", 10000 * loss.cpu() , prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{phase}_loss", loss.cpu(), prog_bar=True, on_step=False, on_epoch=True)
        
        if self.solver.n_step > 0:
            grad_loss = self.weighted_mse(
                err=kfilts.sobel(out_4d) - kfilts.sobel(target_4d), 
                weight=self.rec_weight
            )
            prior_cost = 0.0
            pcs = []
            for d in range(D):
                depth_out = out[:, :, d, :, :].view(B, T, H, W)
                depth_batch = batch._replace(
                    input=batch.input[:, :, d, :, :],
                    tgt=batch.tgt[:, :, d, :, :]
                )
                pcs.append(self.solver.prior_cost(self.solver.init_state(depth_batch, depth_out)))
            prior_cost = torch.stack(pcs).mean()

            
            self.log(f"{phase}_gloss", grad_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{phase}_prior_cost", prior_cost, prog_bar=True, on_step=False, on_epoch=True)
            
            training_loss = 20 * loss +  prior_cost + 20 * grad_loss
            return training_loss, out
        return loss, out

    def test_step(self, batch, batch_idx):
        # reset buffer on first batch
        if batch_idx == 0:
            self.test_data = [] 

        raw_input = batch.input.clone() 
        masked_input = self._mask_input(raw_input)
        masked_batch = batch._replace(input=masked_input)
        out = self(masked_batch)
        
        # denorm to (B, T, C, H, W)
        if self.norm_type == "z_score":
            raw_input_den = self._denormalize_zscore(raw_input,      self.norm_stats)
            masked_den    = self._denormalize_zscore(masked_batch.input, self.norm_stats)
            tgt_den       = self._denormalize_zscore(batch.tgt,      self.norm_stats)
            out_den       = self._denormalize_zscore(out,          self.norm_stats)
        else:
            raw_input_den = self._denormalize_minmax(raw_input,      self.norm_stats)
            masked_den    = self._denormalize_minmax(masked_batch.input, self.norm_stats)
            tgt_den       = self._denormalize_minmax(batch.tgt,      self.norm_stats)
            out_den       = self._denormalize_minmax(out,          self.norm_stats)

        self.test_data.append(torch.stack(
            [raw_input_den.cpu(), masked_den.cpu(), tgt_den.cpu(), out_den.detach().cpu()],
            dim=1,
        ))
    
    @property
    def test_quantities(self):
        # keep your ordering & names
        return ['input', 'inp', 'tgt', 'out']
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


# class TransfertXrDataset_LazyDepth(TransfertXrDataset):
#     """
#     Dataset that use k=lazy loading for the depth dimension.
#     """
#     pass