import inspect
import random
from collections.abc import Sequence as SequenceABC
from pathlib import Path
from typing import Optional, Sequence, Union

import kornia.filters as kfilts
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn


def _uniform_rate(rate: Union[float, Sequence[float]]) -> float:
    if isinstance(rate, SequenceABC) and not isinstance(rate, (str, bytes)) and len(rate) == 2:
        return random.uniform(float(rate[0]), float(rate[1]))
    return float(rate)


def get_triang_time_weight(patch_dims, offset=0, crop=None):
    dims = list(patch_dims.keys())
    crop = crop or {}
    patch_weight = np.zeros([patch_dims[d] for d in dims], dtype="float32")
    mask = tuple(
        slice(crop[d], -crop[d]) if crop.get(d, 0) > 0 else slice(None)
        for d in dims
    )
    patch_weight[mask] = 1.0

    time_weight = np.fromfunction(
        lambda t: 1.0 - np.abs(offset + 2 * t - patch_dims["time"]) / patch_dims["time"],
        (patch_dims["time"],),
        dtype=np.float32,
    ).astype("float32")
    time_shape = [1] * len(dims)
    time_shape[dims.index("time")] = patch_dims["time"]
    return patch_weight * time_weight.reshape(time_shape)


def cosanneal_lr_adamw(lit_mod, lr, T_max=100, weight_decay=0.0):
    opt = torch.optim.AdamW(lit_mod.parameters(), lr=lr, weight_decay=weight_decay)
    return {
        "optimizer": opt,
        "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max),
    }


class ConditionalVelocityWrapper(nn.Module):
    """
    Adapts a backbone to the CFM velocity signature `(x_t, t, condition, extra)`.

    `concat_extra` is intended for the guided-diffusion UNets in `contrib.Unet_fdv`:
    they receive `x_t`, `timesteps`, and an `extra` dict containing
    `concat_conditioning`.
    """

    def __init__(self, net: nn.Module, condition_mode: str = "concat_extra"):
        super().__init__()
        valid_modes = {"concat_extra", "concat_input", "extra_only", "none"}
        if condition_mode not in valid_modes:
            raise ValueError(f"condition_mode must be one of {sorted(valid_modes)}")
        self.net = net
        self.condition_mode = condition_mode

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        extra: Optional[dict] = None,
    ) -> torch.Tensor:
        extra = dict(extra or {})

        if self.condition_mode == "concat_input" and condition is not None:
            x_t = torch.cat([x_t, condition], dim=1)
        elif self.condition_mode == "concat_extra" and condition is not None:
            extra["concat_conditioning"] = condition
        elif self.condition_mode == "extra_only" and condition is not None:
            extra["conditioning"] = condition

        return self._call_net(x_t, t, extra)

    def _call_net(self, x: torch.Tensor, t: torch.Tensor, extra: dict) -> torch.Tensor:
        signature = inspect.signature(self.net.forward)
        params = list(signature.parameters.values())
        has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
        positional = [
            p for p in params
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]

        if has_varargs or len(positional) >= 3:
            return self.net(x, t, extra)
        if len(positional) == 2:
            return self.net(x, t)
        return self.net(x)


class LitConditionalFlowMatching(pl.LightningModule):
    """
    Conditional flow matching trainer for DA/reconstruction batches.

    The training path follows the stochastic interpolant used in the preprint:

        x_tau = (1 - tau) x0 + tau x1,  x0 ~ N(0, I)

    and trains a conditional velocity field v_theta(x_tau, tau, y) against
    dx_tau / d_tau = x1 - x0. It also logs the implied conditional expectation
    estimate `x1_hat = x_tau + (1 - tau) v_theta`, matching the paper's
    `E[x1 | x_tau, y]` objective.
    """

    def __init__(
        self,
        velocity_net: nn.Module,
        opt_fn,
        rec_weight=None,
        sampling_rate: Union[float, Sequence[float], None] = None,
        norm_type: str = "z_score",
        norm_stats=None,
        test_metrics=None,
        pre_metric_fn=None,
        include_obs_mask: bool = True,
        noise_std: float = 1.0,
        t_eps: float = 1e-5,
        velocity_loss_weight: float = 1.0,
        x1_loss_weight: float = 1.0,
        grad_loss_weight: float = 0.0,
        sample_steps: int = 20,
        persist_rw: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["velocity_net", "opt_fn", "rec_weight", "test_metrics", "pre_metric_fn", "kwargs"])
        self.velocity_net = velocity_net
        self.opt_fn = opt_fn
        self.sampling_rate = sampling_rate
        self.norm_type = norm_type
        self._norm_stats = norm_stats
        self.metrics = test_metrics or {}
        self.pre_metric_fn = pre_metric_fn or (lambda x: x)
        self.include_obs_mask = include_obs_mask
        self.noise_std = noise_std
        self.t_eps = t_eps
        self.velocity_loss_weight = velocity_loss_weight
        self.x1_loss_weight = x1_loss_weight
        self.grad_loss_weight = grad_loss_weight
        self.sample_steps = sample_steps
        self.test_data = None

        if rec_weight is not None:
            self.register_buffer(
                "rec_weight",
                torch.as_tensor(rec_weight, dtype=torch.float32),
                persistent=persist_rw,
            )
        else:
            self.rec_weight = None

    @property
    def norm_stats(self):
        if self._norm_stats is not None:
            return self._norm_stats
        trainer = getattr(self, "_trainer", None)
        if trainer is not None and trainer.datamodule is not None:
            return trainer.datamodule.norm_stats()
        return (0.0, 1.0)

    def configure_optimizers(self):
        return self.opt_fn(self)

    def forward(self, batch, sample_steps: Optional[int] = None, x0: Optional[torch.Tensor] = None):
        return self.sample(batch, sample_steps=sample_steps, x0=x0)

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def step(self, batch, phase: str):
        x1, _ = self._flatten_state(batch.tgt)
        valid_target = torch.isfinite(x1)
        x1 = x1.nan_to_num()

        condition_input = self._apply_mask(batch.input)
        condition = self._condition_from_input(condition_input)
        extra = self._extra_from_batch(batch, x1.device)

        x0 = self.noise_std * torch.randn_like(x1)
        t = self._sample_t(x1)
        x_t = (1.0 - t) * x0 + t * x1
        target_velocity = x1 - x0

        pred_velocity = self.velocity_net(x_t, t.flatten(), condition, extra)
        weight = self._channel_weight(x1)

        velocity_loss = self._weighted_mse(pred_velocity - target_velocity, weight, valid_target)
        x1_hat = x_t + (1.0 - t) * pred_velocity
        x1_loss = self._weighted_mse(x1_hat - x1, weight, valid_target)

        if self.grad_loss_weight > 0.0:
            grad_valid = valid_target.all(dim=1, keepdim=True).expand_as(x1)
            grad_loss = self._weighted_mse(
                kfilts.sobel(x1_hat) - kfilts.sobel(x1),
                weight,
                grad_valid,
            )
        else:
            grad_loss = torch.zeros((), device=x1.device, dtype=x1.dtype)

        loss = (
            self.velocity_loss_weight * velocity_loss
            + self.x1_loss_weight * x1_loss
            + self.grad_loss_weight * grad_loss
        )

        self.log(f"{phase}_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=x1.size(0))
        self.log(f"{phase}_cfm_loss", velocity_loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=x1.size(0))
        self.log(f"{phase}_x1_loss", x1_loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=x1.size(0))
        self.log(f"{phase}_mse", 10000.0 * x1_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=x1.size(0))
        if self.grad_loss_weight > 0.0:
            self.log(f"{phase}_gloss", grad_loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=x1.size(0))

        return loss

    @torch.no_grad()
    def sample(self, batch, sample_steps: Optional[int] = None, x0: Optional[torch.Tensor] = None):
        sample_steps = int(sample_steps or self.sample_steps)
        ref = batch.tgt if hasattr(batch, "tgt") else batch.input
        ref_channels, original_shape = self._flatten_state(ref)
        condition = self._condition_from_input(batch.input)
        extra = self._extra_from_batch(batch, ref_channels.device)

        x = self.noise_std * torch.randn_like(ref_channels) if x0 is None else self._flatten_state(x0)[0]
        dt = 1.0 / sample_steps
        for step in range(sample_steps):
            t = torch.full((x.size(0),), step * dt, device=x.device, dtype=x.dtype)
            x = x + dt * self.velocity_net(x, t, condition, extra)

        return self._unflatten_state(x, original_shape)

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.test_data = []

        out = self(batch=batch)
        denorm = self._denorm_fn(batch.tgt, getattr(batch, "comp_idx", None))
        self.test_data.append(torch.stack(
            [
                denorm(batch.input).detach().cpu(),
                denorm(batch.tgt).detach().cpu(),
                denorm(out).detach().cpu(),
            ],
            dim=1,
        ))

    @property
    def test_quantities(self):
        return ["inp", "tgt", "out"]

    def on_test_epoch_end(self):
        rec_da = self.trainer.test_dataloaders.dataset.reconstruct(
            self.test_data,
            self._weight_for_reconstruct(),
        )

        if isinstance(rec_da, list):
            rec_da = rec_da[0]

        self.test_data = rec_da.assign_coords(dict(v0=self.test_quantities)).to_dataset(dim="v0")
        metric_data = self.test_data.pipe(self.pre_metric_fn)
        metrics = pd.Series({
            metric_n: metric_fn(metric_data)
            for metric_n, metric_fn in self.metrics.items()
        })

        print(metrics.to_frame(name="Metrics").to_markdown())
        if self.logger:
            save_path = Path(self.logger.log_dir) / "test_data.nc"
            self.test_data.to_netcdf(save_path)
            print(save_path)
            self.logger.log_metrics(metrics.to_dict())

    def _apply_mask(self, x: torch.Tensor) -> torch.Tensor:
        if self.sampling_rate is None:
            return x

        x = x.clone()
        for b in range(x.size(0)):
            sr = _uniform_rate(self.sampling_rate)
            keep = torch.rand_like(x[b], dtype=torch.float32) < sr
            x[b][~keep] = float("nan")
        return x

    def _sample_t(self, ref: torch.Tensor) -> torch.Tensor:
        t = torch.rand((ref.size(0), 1, 1, 1), device=ref.device, dtype=ref.dtype)
        if self.t_eps > 0.0:
            t = self.t_eps + (1.0 - 2.0 * self.t_eps) * t
        return t

    def _condition_from_input(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self._flatten_state(x)
        obs_mask = torch.isfinite(x).to(dtype=x.dtype)
        x = x.nan_to_num()
        if self.include_obs_mask:
            return torch.cat([x, obs_mask], dim=1)
        return x

    def _extra_from_batch(self, batch, device) -> dict:
        extra = {}
        depth = None
        for name in ("depth_idx", "z_idx", "comp_idx"):
            if hasattr(batch, name):
                depth = getattr(batch, name)
                break

        if depth is not None:
            depth = torch.as_tensor(depth, device=device, dtype=torch.float32)
            if depth.ndim > 1:
                depth = depth.flatten(start_dim=1).mean(dim=1)
            extra["depth_idx"] = depth
        return extra

    @staticmethod
    def _flatten_state(x: torch.Tensor):
        if x.ndim == 5:
            b, t, c, h, w = x.shape
            return x.reshape(b, t * c, h, w), x.shape
        if x.ndim == 4:
            return x, None
        raise ValueError(f"Expected a 4D or 5D state tensor, got shape {tuple(x.shape)}")

    @staticmethod
    def _unflatten_state(x: torch.Tensor, original_shape):
        if original_shape is None:
            return x
        b, t, c, h, w = original_shape
        return x.reshape(b, t, c, h, w)

    def _channel_weight(self, ref: torch.Tensor):
        if self.rec_weight is None:
            return None

        weight = self.rec_weight.to(device=ref.device, dtype=ref.dtype)
        if weight.ndim == 4:
            weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2], weight.shape[3])
        return weight

    def _weight_for_reconstruct(self):
        if self.rec_weight is None:
            return None
        return self.rec_weight.detach().cpu().numpy()

    @staticmethod
    def _weighted_mse(err: torch.Tensor, weight: Optional[torch.Tensor] = None, valid_mask: Optional[torch.Tensor] = None):
        if weight is None:
            weighted = err
            finite = torch.isfinite(weighted)
        else:
            if weight.ndim == 2:
                weight = weight[None, None, ...]
            elif weight.ndim == 3:
                weight = weight[None, ...]
            elif weight.ndim != err.ndim:
                raise ValueError(f"Unsupported weight shape {tuple(weight.shape)} for error shape {tuple(err.shape)}")
            weighted = err * weight
            finite = torch.isfinite(weighted) & (weight.expand_as(err) != 0.0)

        if valid_mask is not None:
            finite = finite & valid_mask

        if not finite.any():
            return torch.tensor(1000.0, device=err.device, dtype=err.dtype, requires_grad=True)
        return (weighted[finite] ** 2).mean()

    def _denorm_fn(self, ref_tensor: torch.Tensor, comp_idx: Optional[torch.Tensor]):
        stats = self.norm_stats
        if stats is None:
            return lambda z: z

        if self.norm_type == "z_score":
            mean, std = stats
            mean_t = torch.as_tensor(mean, device=ref_tensor.device, dtype=ref_tensor.dtype)
            std_t = torch.as_tensor(std, device=ref_tensor.device, dtype=ref_tensor.dtype)

            def denorm(z):
                m = self._expand_stat(mean_t, z, comp_idx)
                s = self._expand_stat(std_t, z, comp_idx)
                return z * s + m

            return denorm

        vmin, vmax = stats
        vmin_t = torch.as_tensor(vmin, device=ref_tensor.device, dtype=ref_tensor.dtype)
        vmax_t = torch.as_tensor(vmax, device=ref_tensor.device, dtype=ref_tensor.dtype)

        def denorm(z):
            mn = self._expand_stat(vmin_t, z, comp_idx)
            mx = self._expand_stat(vmax_t, z, comp_idx)
            return z * (mx - mn) + mn

        return denorm

    @staticmethod
    def _expand_stat(stat: torch.Tensor, ref: torch.Tensor, comp_idx: Optional[torch.Tensor]):
        if stat.ndim == 0:
            return stat

        if ref.ndim == 5 and comp_idx is not None:
            b, _, c, _, _ = ref.shape
            comp_idx = comp_idx.to(device=ref.device, dtype=torch.long)
            if comp_idx.ndim == 1:
                comp_idx = comp_idx.view(b, 1).expand(b, c)
            elif comp_idx.shape == (b, 1) and c > 1:
                comp_idx = comp_idx.expand(b, c)
            comp_idx = comp_idx.clamp(0, stat.numel() - 1)
            return stat[comp_idx].view(b, 1, c, 1, 1)

        return stat.mean().view(*([1] * ref.ndim))
