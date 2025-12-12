from ast import Return
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
import math
from omegaconf import ListConfig, DictConfig
from copy import deepcopy
from pathlib import Path
import pandas as pd
from src.data import AugmentedDataset, BaseDataModule, XrDataset
from src.utils import get_constant_crop
from collections import namedtuple
from contrib import transfert
from types import SimpleNamespace
from typing import Sequence, Tuple, Union, Dict, Any
from dataclasses import dataclass, field
from typing import Optional
torch.set_float32_matmul_precision('medium')  # Use TF32 for better memory efficiency
import itertools 

TrainingItem = namedtuple('TrainingItem', ['input', 'tgt', 'comp_idx'])
TrainingItemwithZ = namedtuple('TrainingItemwithZ', ['input', 'tgt', 'z_idx'])

def rmse_based_scores_from_ds_slice(ds, ref_variable='tgt', study_variable='out', depth_dim='component'):
    try:
        return rmse_slicewise_mean_scores(ds[study_variable], ds[ref_variable], depth_dim)
    except:
        return [np.nan, np.nan]

def psd_based_scores_from_ds_slice(ds, ref_variable='tgt', study_variable='out', depth_dim='component'):
    try:
        return psd_slicewise_based_scores(ds[study_variable], ds[ref_variable], depth_dim)
    except:
        return [np.nan, np.nan]

def rmse_slicewise_mean_scores(da_rec, da_ref, depth_dim='component'):
    if depth_dim not in da_ref.dims:
        raise ValueError(f"depth_dim '{depth_dim}' not found in dims {da_ref.dims}")
    print((((da_rec - da_ref) ** 2).mean()) ** 0.5)
    vals_leader = []
    vals_stab   = []
    for dep in da_ref['component'].values:
        ref_slice = da_ref.sel({depth_dim: dep})
        rec_slice = da_rec.sel({depth_dim: dep})
        try:
            _, _, leader_scalar, stab_scalar = rmse_based_scores_std(rec_slice, ref_slice)
            if np.isfinite(leader_scalar): vals_leader.append(leader_scalar)
            if np.isfinite(stab_scalar):   vals_stab.append(stab_scalar)
        except Exception as e:
            # skip slice if it fails (NaNs, insufficient dims, etc.)
            pass
    mean_leader = float(np.nan) if len(vals_leader) == 0 else float(np.mean(vals_leader))
    mean_stab   = float(np.nan) if len(vals_stab)   == 0 else float(np.mean(vals_stab))
    return (mean_leader, mean_stab)

def psd_slicewise_based_scores(da_rec, da_ref, depth_dim='component'):
    if depth_dim not in da_ref.dims:
        raise ValueError(f"depth_dim '{depth_dim}' not found in dims {da_ref.dims}")
    
    vals_psd_x  = []
    vals_psd_t  = []
    for dep in da_ref['component'].values:
        ref_slice = da_ref.sel({depth_dim: dep})
        rec_slice = da_rec.sel({depth_dim: dep})
        # PSD-based (take the two scalar thresholds)
        try:
            _, shortest_x, shortest_t = src.utils.psd_based_scores(rec_slice, ref_slice)
            if np.isfinite(shortest_x): vals_psd_x.append(shortest_x)
            if np.isfinite(shortest_t): vals_psd_t.append(shortest_t)
        except Exception as e:
            pass
    mean_psd_x  = float(np.nan) if len(vals_psd_x)  == 0 else float(np.mean(vals_psd_x))
    mean_psd_t  = float(np.nan) if len(vals_psd_t)  == 0 else float(np.mean(vals_psd_t))
    return (mean_psd_x, mean_psd_t)

def rmse_based_scores_std(da_rec, da_ref):
    rmse_t = (
        1.0
        - (((da_rec - da_ref) ** 2).mean(dim=("lon", "lat"))) ** 0.5
        / (((da_ref) ** 2).mean(dim=("lon", "lat"))) ** 0.5
    )
    rmse_xy = (((da_rec - da_ref) ** 2).mean(dim=("time"))) ** 0.5
    rmse_t = rmse_t.rename("rmse_t")
    rmse_xy = rmse_xy.rename("rmse_xy")
    reconstruction_error_stability_metric = rmse_t.std().values
    leaderboard_rmse = (
        1.0 - (((da_rec - da_ref) ** 2).mean()) ** 0.5 / da_ref.std()
    )
    return (
        rmse_t,
        rmse_xy,
        np.round(leaderboard_rmse.values, 5).item(),
        np.round(reconstruction_error_stability_metric, 5).item(),
    )

def tstats(x, name):
    x = x.detach()
    finite = torch.isfinite(x)
    nz = finite.sum().item()
    return {
        "name": name,
        "shape": tuple(x.shape),
        "min": float(x[finite].min().item()) if nz else float("nan"),
        "max": float(x[finite].max().item()) if nz else float("nan"),
        "mean": float(x[finite].mean().item()) if nz else float("nan"),
        "std": float(x[finite].std().item()) if nz else float("nan"),
        "finite_frac": float(nz) / x.numel(),
    }

def is_bad(x):
    # robust & readable
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    return not torch.isfinite(x).all().item()

def grad_norm(model, p=2.0):
    tot = 0.0
    for n, p_ in model.named_parameters():
        if p_.grad is None: 
            continue
        g = p_.grad.detach()
        if torch.any(~torch.isfinite(g)):
            return float("inf")
        tot += g.norm(p).item() ** 2
    return float(tot) ** 0.5

class LazyXrDataset(torch.utils.data.Dataset):
    """
    Lazily iterate over windowed patches of an xarray DataArray or Dataset.

    - pass `var="pca_test"` if `ds` is a Dataset (we'll select that DataArray)
    - 'full-extent' per-dim via None / -1 in patch_dims
    - keeps xarray.DataArray all the way to postpro_fn (for safe coord-aware ops)
    """
    def __init__(
        self,
        ds,
        patch_dims,                 # dict: {dim: size or None/-1 for full}
        domain_limits=None,
        strides=None,               # dict: {dim: stride}; default 1
        postpro_fn=None,
        *,
        var: str | None = None,     # <- NEW: select variable if ds is a Dataset
        depth_dim: str | None = None,  # <- NEW: used by reconstruct_*
        **kwargs,
    ):
        super().__init__()
        self.return_coords = False
        self.postpro_fn = postpro_fn
        self.depth_dim = depth_dim  # may be None; reconstruct_* checks it
        self.mask = None

        # 1) crop
        self.ds = ds.sel(**(domain_limits or {}))

        # 2) if Dataset, optionally pick one var (strongly recommended)
        if isinstance(self.ds, xr.Dataset):
            if var is not None:
                if var not in self.ds.data_vars:
                    raise KeyError(f"`var='{var}'` not in dataset variables: {list(self.ds.data_vars)}")
                self.ds = self.ds[var]  # now DataArray
            else:
                # if no var provided, keep Dataset but warn via assertive guard in __getitem__
                pass

        # 3) store dims/strides
        self.patch_dims = dict(patch_dims)
        self.strides = dict(strides or {})

        # 4) sizes and early guards
        #    If self.ds is a Dataset here, it means user intentionally wants to pass a Dataset
        #    (e.g., will use mask branch building a stacked 'variable' dim). That also works.
        self._sizes = {dim: self.ds.sizes[dim] for dim in self.ds.dims}
        missing = [k for k in self.patch_dims if k not in self._sizes]
        if missing:
            raise KeyError(
                f"patch_dims keys not found in dataset dims: {missing}. "
                f"Available dims: {list(self._sizes.keys())}"
            )

        # 5) normalize 'full extent'
        for dim in list(self.patch_dims.keys()):
            val = self.patch_dims[dim]
            if val is None or val == -1:
                self.patch_dims[dim] = self._sizes[dim]
            else:
                self.patch_dims[dim] = int(val)

        # 6) default stride 1
        for dim in self.patch_dims:
            self.strides.setdefault(dim, 1)

        # 7) number of windows per dim
        self.ds_size = {}
        for dim, psize in self.patch_dims.items():
            full = (psize == self._sizes[dim])
            if full:
                self.ds_size[dim] = 1
            else:
                stride = self.strides.get(dim, 1)
                if psize > self._sizes[dim]:
                    raise ValueError(f"patch_dims[{dim}]={psize} exceeds dataset size {self._sizes[dim]}")
                self.ds_size[dim] = max((self._sizes[dim] - psize) // stride + 1, 0)

        # 8) unravel helpers
        self._scan_dims = tuple(self.ds_size.keys())
        self._scan_counts = tuple(self.ds_size[d] for d in self._scan_dims)
        self._num = int(np.prod(self._scan_counts)) if len(self._scan_counts) else 1

    def __len__(self):
        return self._num

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_coords(self):
        """Return the coords (as xarray Datasets) corresponding to each lazy patch."""
        self.return_coords = True
        coords = []
        try:
            for i in range(len(self)):
                coords.append(self[i])
        finally:
            self.return_coords = False
        return coords

    def _build_slices(self, flat_idx):
        if len(self._scan_counts) == 0:
            return {dim: slice(None) for dim in self.ds.dims}
        multi_idx = np.unravel_index(flat_idx, self._scan_counts)
        sl = {}
        for dim, count, idx in zip(self._scan_dims, self._scan_counts, multi_idx):
            psize = self.patch_dims[dim]
            stride = self.strides.get(dim, 1)
            if psize == self._sizes[dim]:
                sl[dim] = slice(None)
            else:
                start = stride * int(idx)
                stop  = start + psize
                sl[dim] = slice(start, stop)
        for dim in self.ds.dims:
            if dim not in sl:
                sl[dim] = slice(None)
        return sl

    def __getitem__(self, item):
        sl = self._build_slices(item)

        # build masked two-channel (tgt/input) if a mask is provided
        if self.mask is not None:
            # time-wrap mask if needed
            if "time" in sl and self.mask is not None:
                start = sl["time"].start or 0
                stop  = sl["time"].stop or self._sizes["time"]
                start_mod, stop_mod = start % 365, stop % 365
                if start_mod > stop_mod:
                    start_mod -= stop_mod
                    stop_mod = None
                sl_mask = dict(sl)
                sl_mask["time"] = slice(start_mod, stop_mod)
            else:
                sl_mask = sl

            da = self.ds.isel(**sl)
            # ensure we’re working with a DataArray
            if isinstance(da, xr.Dataset):
                # pick 'tgt' if present, else first var (to be consistent)
                main_name = "tgt" if "tgt" in da.data_vars else next(iter(da.data_vars))
                da_main = da[main_name]
            else:
                da_main = da

            ds_stack = da_main.to_dataset(name="tgt")
            inp = da_main.where(self.mask.isel(**sl_mask).values) if self.mask is not None else da_main
            ds_stack = ds_stack.assign(input=inp)
            item_xr = ds_stack.to_array()  # dims: ('variable', ...) with ['tgt','input'] if names exist

            if self.return_coords:
                return item_xr.coords.to_dataset()[list(self.patch_dims)]

            x = item_xr.astype(np.float32)  # keep xarray to allow coord-safe postpro
            return self.postpro_fn(x) if self.postpro_fn is not None else x

        # no mask: just slice
        item_xr = self.ds.isel(**sl)

        # If still a Dataset here (no var provided at init), pick the first data_var deterministically
        if isinstance(item_xr, xr.Dataset):
            pick = next(iter(item_xr.data_vars))
            item_xr = item_xr[pick]

        if self.return_coords:
            return item_xr.coords.to_dataset()[list(self.patch_dims)]

        x = item_xr.astype(np.float32)  # keep xarray to allow coord-safe postpro
        return self.postpro_fn(x) if self.postpro_fn is not None else x

    # ---------------- reconstruction (unchanged except: self.depth_dim may be None) ----------------
    def reconstruct(self, batches, weight=None):
        items = list(itertools.chain(*batches))
        return self.reconstruct_from_items(items, weight)

    def reconstruct_from_items(self, items, weight=None):
        # normalize items to numpy and collect coords as before
        def _to_numpy(a):
            if torch.is_tensor(a):
                return a.detach().cpu().numpy()
            if isinstance(a, xr.DataArray):
                return a.values
            return np.asarray(a)
    
        coords = self.get_coords()
        sample = items[0]
        item_np = _to_numpy(sample)
        coord_dims = list(coords[0].dims)
    
        # leading dims (v0, v1, ...) if any
        n_lead = item_np.ndim - len(coord_dims)
        new_dims = [f'v{i}' for i in range(n_lead)]
        dims = new_dims + coord_dims
    
        das = [
            xr.DataArray(_to_numpy(it), dims=dims, coords=co.coords)
            for it, co in zip(items, coords)
        ]
    
        # full output shape & coords
        full_shape = {}
        for dim in coord_dims:
            if dim in self.ds.dims:
                full_shape[dim] = self.ds.sizes[dim]
            else:
                full_shape[dim] = max(co[dim].size for co in coords)
    
        for i, dim in enumerate(new_dims):
            full_shape[dim] = item_np.shape[i]
    
        full_coords = {}
        for dim in coord_dims:
            if dim in self.ds.coords:
                full_coords[dim] = self.ds[dim]
            else:
                full_coords[dim] = np.arange(full_shape[dim])
    
        rec_da = xr.DataArray(
            np.zeros([full_shape[dim] for dim in dims], dtype=np.float32),
            dims=dims,
            coords=full_coords,
        )
        count_da = xr.zeros_like(rec_da)
    
        # ---------- robust weight handling ----------
        # keys we "slide" on (exclude depth_dim if provided)
        if getattr(self, "depth_dim", None) is not None:
            spatial_temporal_keys = [k for k in self.patch_dims if k != self.depth_dim]
        else:
            spatial_temporal_keys = list(self.patch_dims.keys())
    
        # Make an xr.DataArray 'w' with whatever dims user supplied; broadcast later.
        def _make_weight_da(weight):
            if weight is None:
                # default to ones over all non-depth patch dims
                shp = [self.patch_dims[k] for k in spatial_temporal_keys]
                return xr.DataArray(np.ones(shp, dtype=np.float32), dims=spatial_temporal_keys)
    
            if isinstance(weight, xr.DataArray):
                return weight.astype(np.float32)
    
            w_np = np.asarray(weight)
            # Try to assign dims heuristically by matching known sizes
            known_sizes = {k: self.patch_dims[k] for k in self.patch_dims}
            # candidates in preferred order
            pref = ['time', 'lat', 'lon', 'component']
            # build a dims list that matches w_np.shape
            dims_guess = []
            sizes_left = dict(known_sizes)
            shape = list(w_np.shape)
    
            # Simple cases first
            if w_np.ndim == 0:
                return xr.DataArray(float(w_np))
            if w_np.ndim == 1:
                # try to match a single known dim by size
                for k in pref:
                    if k in sizes_left and sizes_left[k] == shape[0]:
                        return xr.DataArray(w_np.astype(np.float32), dims=[k])
                # fallback: unnamed 1D (broadcast later)
                return xr.DataArray(w_np.astype(np.float32))
    
            # Multi-dim: greedily match from pref list
            used = [False] * len(shape)
            for k in pref:
                if k in sizes_left:
                    for i, s in enumerate(shape):
                        if not used[i] and s == sizes_left[k]:
                            dims_guess.append(k)
                            used[i] = True
                            break
                        
            # if we matched all axes, great; else leave unnamed (will still broadcast)
            if sum(used) == len(shape):
                return xr.DataArray(w_np.astype(np.float32), dims=dims_guess)
            else:
                return xr.DataArray(w_np.astype(np.float32))  # no dims; xarray will still try to align/broadcast
    
        w = _make_weight_da(weight)
        # --- normalize weight dims to match da dims ---
        # ---------- strict weight handling for (T*C,H,W) and (T,C,H,W) ----------
        T = int(self.patch_dims["time"])
        C = int(self.patch_dims["component"])
        H = int(self.patch_dims["lat"])
        W = int(self.patch_dims["lon"])

        def _weight_to_TCHW(weight):
            """
            Accept only (T*C,H,W) or (T,C,H,W), return np.float32 of shape (T,C,H,W).
            """
            if isinstance(weight, xr.DataArray):
                w_np = weight.values
            else:
                w_np = np.asarray(weight)

            if w_np.shape == (T * C, H, W):
                return w_np.reshape(T, C, H, W).astype(np.float32)

            if w_np.shape == (T, C, H, W):
                return w_np.astype(np.float32)

            raise ValueError(
                f"Unsupported weight shape {w_np.shape}; expected (T*C,H,W) or (T,C,H,W) "
                f"with T={T}, C={C}, H={H}, W={W}."
            )

        w4 = _weight_to_TCHW(weight)  # np.ndarray, (T,C,H,W)
        # ------------- accumulation loop -------------
        for da in das:
            # Build a broadcast shape for weight that matches da.dims:
            # - 1 for any leading dims (e.g., 'v0')
            # - T, C, H, W for the named patch dims.
            shape_per_dim = []
            for d in da.dims:
                if d == "time":
                    shape_per_dim.append(T)
                elif d == "component":
                    shape_per_dim.append(C)
                elif d == "lat":
                    shape_per_dim.append(H)
                elif d == "lon":
                    shape_per_dim.append(W)
                else:
                    shape_per_dim.append(1)

            w_use_np = w4.reshape(shape_per_dim)           # strictly positional broadcast
            patch_vals = da.data * w_use_np                # NumPy multiply, no xarray alignment

            rec_sel   = rec_da.sel(da.coords)
            count_sel = count_da.sel(da.coords)

            rec_da.loc[da.coords]   = xr.DataArray(rec_sel.data + patch_vals, dims=da.dims, coords=da.coords)
            count_da.loc[da.coords] = xr.DataArray(count_sel.data + w_use_np,  dims=da.dims, coords=da.coords)

        result = xr.where(count_da > 0, rec_da / count_da, rec_da)
        return result

class TransfertLazyDataModule(transfert.TransfertDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean_std_domain = kwargs.get('mean_std_domain', 'train')
        self.std_c = kwargs.get('std_c', 1.)
    
    def _select_var_da(self, obj, variable: str):
        """
        Return a DataArray for the requested 'variable', handling:
        - Dataset with data_vars -> obj[variable]
        - DataArray with 'variable' dim -> sel(variable=...)
        - Plain DataArray (no variable dim) -> return as-is
        Falls back to index [0=input, 1=tgt] if names are missing.
        """
        if isinstance(obj, xr.Dataset):
            if variable in obj.data_vars:
                return obj[variable]
            # fallback: convert to array (variable, ...)
            arr = obj.to_array()
            if 'variable' in arr.coords and (variable in arr['variable'].values):
                return arr.sel(variable=variable)
            idx = 1 if variable == 'tgt' else 0
            return arr.isel(variable=idx)

        elif isinstance(obj, xr.DataArray):
            if 'variable' in obj.dims:
                if 'variable' in obj.coords and (variable in obj['variable'].values):
                    return obj.sel(variable=variable)
                idx = 1 if variable == 'tgt' else 0
                return obj.isel(variable=idx)
            return obj

        else:
            raise TypeError(f"Unsupported xarray object type: {type(obj)}")

    def train_mean_std(self, variable='tgt'):
        da = (
            self.input_da
            .sel(self.xrds_kw.get('domain_limits', {}))
            .sel(self.domains[self.mean_std_domain])
        )
        da = self._select_var_da(da, variable)   # <- robust selection
        means = da.mean(dim=('time', 'lat', 'lon'))            # (component,)
        stds  = da.std (dim=('time', 'lat', 'lon')) * self.std_c
        return means.values, stds.values

    def min_max_norm(self, variable='tgt'):
        da = (
            self.input_da
            .sel(self.xrds_kw.get('domain_limits', {}))
            .sel(self.domains[self.mean_std_domain])
        )
        da = self._select_var_da(da, variable)
        vmin = da.min(dim=('time', 'lat', 'lon'))
        vmax = da.max(dim=('time', 'lat', 'lon'))
        return vmin.values, vmax.values

    def post_fn(self):
        means, stds = self.norm_stats()
        means = np.asarray(means, dtype=np.float32)
        stds  = np.asarray(stds,  dtype=np.float32)

        # full "global" component coordinate (vector of all components)
        if isinstance(self.input_da, xr.Dataset):
            if 'component' in self.input_da.coords:
                full_comp = self.input_da['component']
            else:
                first_var = next(iter(self.input_da.data_vars))
                full_comp = self.input_da[first_var]['component']
        else:
            full_comp = self.input_da['component']
        
        def _norm_xr(da: xr.DataArray) -> xr.DataArray:
            comp = da['component']  # the components present in this patch (often length 1)
            m_full = xr.DataArray(means, dims=['component'], coords={'component': full_comp})
            s_full = xr.DataArray(stds,  dims=['component'], coords={'component': full_comp})
            m = m_full.sel(component=comp)
            s = s_full.sel(component=comp)
            return (da - m) / (s + 1e-8)    

        def _make_item(x: xr.DataArray):
            # x dims are typically ('time','component','lat','lon') or include 'variable'
            # collect the *labels* (not positions) of the selected component slice
            comp_labels = x['component'].values  # shape (D,)
            comp_idx_torch = torch.as_tensor(comp_labels, dtype=torch.long)

            if 'variable' in x.dims:
                # assume order [input, tgt]; tweak if yours is different
                inp = _norm_xr(x.isel(variable=0))
                tgt = _norm_xr(x.isel(variable=1))
            else:
                fld = _norm_xr(x)
                inp = fld; tgt = fld

            # Build TrainingItem with .input, .tgt and .comp_idx
            # If your TrainingItem already has comp_idx, use _replace; otherwise create with 3 fields.
            try:
                # namedtuple already includes comp_idx
                return TrainingItem._make([
                    inp.data.astype(np.float32),
                    tgt.data.astype(np.float32),
                    comp_idx_torch,
                ])
            except TypeError:
                # define once at import time in your module:
                # TrainingItem = namedtuple("TrainingItem", ["input","tgt","comp_idx"])
                return TrainingItem(
                    inp.data.astype(np.float32),
                    tgt.data.astype(np.float32),
                    comp_idx_torch,
                )

        return _make_item
        
    def setup(self, stage='test'):
        post_fn = self.post_fn()
        if stage == 'fit':
            train_data = self.input_da.sel(self.domains['train'])
            train_xrds_kw = deepcopy(self.xrds_kw)
            self.train_ds = LazyXrDataset(
                train_data, **train_xrds_kw, depth_dim='component', postpro_fn=post_fn,
            )
            if self.aug_kw:
                self.train_ds = AugmentedDataset(self.train_ds, **self.aug_kw)

            self.val_ds = LazyXrDataset(
                self.input_da.sel(self.domains['val']),
                **self.xrds_kw,
                depth_dim='component',
                postpro_fn=post_fn,
            )
        else:
            self.test_ds = LazyXrDataset(
                self.input_da.sel(self.domains['test']),
                **self.xrds_kw,
                depth_dim='component',
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
        #.to_array()
    )

def load_natl_data_pca(tgt_path, tgt_var, inp_path, inp_var, **kwargs):
    tgt = (
        xr.open_dataset(tgt_path)[tgt_var]
        .sel(kwargs.get('domain', None))
        .sel(kwargs.get('period', None))
    )
    inp = (
        xr.open_dataset(inp_path)[inp_var]
        .sel(kwargs.get('domain', None))
        .sel(kwargs.get('period', None))
    )

    ds = xr.Dataset(
        dict(input=inp, tgt=(tgt.dims, tgt.values)),
        inp.coords,
    ).rename({'z': 'component'}).transpose('time', 'component', 'lat', 'lon')#.to_array()
    
    return ds

@dataclass
class VarSpec:
    path:       str
    name:       str                   
    extra_dim:  Optional[str] = None
    fill_nan:   Optional[float] = None
    threshold:  Optional[float] = None
    broadcast_time: bool = False
    input_arch:  str = field(init=False, repr=False)
    output_arch: str = field(init=False, repr=False)

@dataclass
class LoaderCfg:
    tgt_specs: list[VarSpec]
    inp_specs: list[VarSpec]
    domain: dict                
    full_time_domain: dict

def _get_optional(spec, key, default=None):
    """
    Robustly fetch an optional field from either a dataclass or a DictConfig.
    """
    if isinstance(spec, DictConfig):
        return spec.get(key, default)
    return getattr(spec, key, default)       

def _spec_to_namespace(spec: VarSpec, *, role: str) -> SimpleNamespace:
    """
    Convert VarSpec into the exact attribute set expected downstream.
    YAML stays untouched; we translate here.
    """
    key = f"{spec.name}_{role}"       
    ns =  SimpleNamespace(
        var_path       = _get_optional(spec, "path"),       # required
        var_name       = _get_optional(spec, "name"),       # required
        extra_dim      = _get_optional(spec, "extra_dim"),
        fill_nan       = _get_optional(spec, "fill_nan"),
        threshold      = _get_optional(spec, "threshold"),
        broadcast_time = bool(_get_optional(spec, "broadcast_time", False)),
        input_arch     = _get_optional(spec, "input_arch"),
        output_arch    = _get_optional(spec, "output_arch"),
    )
    return key, ns, role

def load_natl_multivar(*,
                       vars_info: dict,
                       domain: dict,
                       full_time_domain: dict,
                       drop_depth: bool = True):
    """
    Returns
    -------
    xr.DataArray  (variables, time, lat, lon, …)
    dict          (multivar_information)
    """
    full_ds, multivar_info = open_multivar_datasets(
        vars_info        = vars_info,
        domain           = domain.copy(),
        full_time_domain = full_time_domain,
        drop_depth       = drop_depth,
    )
    da_list = []
    for v in full_ds.variable.values:
        da_v = full_ds.sel(variable=v)
        if getattr(vars_info[v], "apply_threshold", False):
            da_v = da_v.pipe(threshold_xarray)
        da_list.append(da_v)
    full_ds = xr.concat(da_list, dim="variable")

    return full_ds, multivar_info


def open_multivar_datasets(
    vars_info:        Dict[str, Any],
    domain:           Dict[str, Any],
    full_time_domain: Dict[str, Any],
    drop_depth:       bool = True,
):

    # combine train+val+test so we read the file once
    domain['time'] = slice(
        full_time_domain['train']['time'].start,
        full_time_domain['test']['time'].stop,
    )

    multivar_information: Dict[str, Dict[str, Any]] = {}
    full_dataset = None

    for var, info in vars_info.items():
        print(f"\nhandling [{var}]")

        clean_ds = open_var_dataset(
            info.var_path, var, info.var_name, domain, drop_depth,
            extra_dim   = getattr(info, 'extra_dim', None),
            fill_nan    = getattr(info, 'fill_nan',  None),
            threshold   = getattr(info, 'threshold', None),
        )

        full_dataset = merge_datasets(
            full_dataset, clean_ds,
            broadcast_time = getattr(info, 'broadcast_time', False),
        )

        for new_var in clean_ds.data_vars:
            multivar_information[new_var] = {
                'input_arch':  info.input_arch,
                'output_arch': info.output_arch,
            }
    var_order = list(multivar_information.keys())       
    roles = [
        "tgt" if info["output_arch"] != "no_output" else "inp"
        for info in multivar_information.values()
    ]
    full_dataset = (
        full_dataset
        .sel(domain)
        [var_order]   # keep the chosen order
        .transpose('time', 'lat', 'lon', ...)
        .to_array()
        .assign_coords(variable=("variable", roles))
    )
    return full_dataset, multivar_information

def open_var_dataset(
    var_path:   str,
    var:        str,
    var_name:   str,
    domain:     Dict[str, Any],
    drop_depth: bool,
    *,
    extra_dim: Optional[str] = None,          
    fill_nan:  Optional[Union[int, float]] = None,
    threshold: Optional[Union[int, float]] = None,
):
    """
    Load one physical variable from disk.

    If `extra_dim` is provided *and is a dimension of the file*, the function
    splits that dimension so that each slice becomes its own variable named
    f"{var}_{label}" (where *label* is the coordinate value).

    All the resulting variables are returned together in a single Dataset.
    """
    data = xr.open_dataset(var_path)[var_name]
    # if 'depth' in data.dims and drop_depth:
    #     data = data.drop('depth')

    if 'latitude' in data.dims:
        data = data.rename({'latitude': 'lat', 'longitude': 'lon'})

    trimmed_domain = {k: v for k, v in domain.items() if k in data.dims}
    data = data.sel(trimmed_domain)

    if fill_nan is not None:
        data = data.fillna(fill_nan)

    if threshold is not None:
        data = xr.where(data > threshold, 0, data)
        data = xr.where(data <= 0, 0, data)

    feat_dim = extra_dim or ("component" if "component" in data.dims else None)

    if feat_dim and feat_dim in data.dims:                   
        data = data.rename({feat_dim: "channel"})            
    else:                                                    
        data = data.expand_dims(channel=[var])                            

    ds = xr.Dataset({var: data})

    return ds


def merge_datasets(
    original_dataset: Optional[xr.Dataset],
    new_dataset:      xr.Dataset,
    *, 
    broadcast_time: bool = False
):
    """
    Merge `new_dataset` into `original_dataset`.

    If broadcast_time=True and new_dataset is 2-D (lat,lon), it is broadcast
    along the time axis of original_dataset.
    """
    if original_dataset is None:
        return new_dataset

    if broadcast_time and 'time' not in new_dataset.dims:
        new_dataset = (
            new_dataset
            .reindex({'lat': original_dataset.lat, 'lon': original_dataset.lon},
                     method='nearest')
            .expand_dims({'time': original_dataset.time})
            .broadcast_like(original_dataset)
        )

    return original_dataset.assign(new_dataset)

def load_multivar_dataset(cfg: LoaderCfg):
    """
    Hydra entry-point.  Expects `cfg` to be instantiated from LoaderCfg.
    Returns (xarray.DataArray, multivar_information_dict)
    """
    tgt_list, inp_list = [], []                               
    for spec in cfg.tgt_specs:
        key, ns, _ = _spec_to_namespace(spec, role="tgt")
        da = open_var_dataset(
            ns.var_path, ns.var_name, ns.var_name,
            domain=cfg.domain, drop_depth=True,
            extra_dim=ns.extra_dim, fill_nan=ns.fill_nan, threshold=ns.threshold
        )[ns.var_name]                                               
        tgt_list.append(da)                                   

    for spec in cfg.inp_specs:
        key, ns, _ = _spec_to_namespace(spec, role="input")
        da = open_var_dataset(
            ns.var_path, ns.var_name, ns.var_name,
            domain=cfg.domain, drop_depth=True,
            extra_dim=ns.extra_dim, fill_nan=ns.fill_nan, threshold=ns.threshold
        )[ns.var_name]
        inp_list.append(da)                                   

    tgt_da = xr.concat(tgt_list, dim="channel")                  
    inp_da = xr.concat(inp_list, dim="channel")                  

    stacked = (
        xr.Dataset(dict(tgt=tgt_da, input=inp_da))   # <- one dataset
          .to_array()                              # -> DataArray (variable, ...)
          .transpose("variable", "time", "channel", "lat", "lon")
    )
    return stacked   

def threshold_xarray(da):
    threshold = 999
    da = xr.where(da > threshold, 0, da)
    da = xr.where(da <= 0, 0, da)
    return da

def get_triang_time_wei_coeff(patch_dims, offset=0, **crop_kw):
    pw = get_constant_crop(patch_dims, **crop_kw)
    time_weight = np.fromfunction(
        lambda t: (1 - np.abs(offset + 2*t - patch_dims["time"]) / patch_dims["time"]),
        (patch_dims["time"],),
    )
    time_weight_4d = time_weight[:, None, None, None]
    broadcast_shape = tuple(patch_dims[d] for d in patch_dims.keys())  # e.g. (15,5,240,240)
    time_weight_4d = np.broadcast_to(time_weight_4d, broadcast_shape)
    pw_4d = np.expand_dims(pw, axis=1)
    pw_4d = np.repeat(pw_4d, repeats=patch_dims["component"], axis=1)
    result_4d = time_weight_4d * pw_4d
    result_3d = result_4d.reshape(patch_dims["time"] * patch_dims["component"], 
                                  patch_dims["lat"], 
                                  patch_dims["lon"])

    return result_3d

# class Lit4dVarNet_MultiDepth(transfert.Lit4dVarNet_Fasc):
#     """
#     Enhanced 4DVarNet that processes each depth level separately 
#     then combines them using a UNet
#     """

#     def __init__(self, depth_level: int = 8, depth_gradient_weight: float = 1.0, coherence_weight: float = 0.5, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         self.depth_level = depth_level
#         self.depth_gradient_weight = depth_gradient_weight
#         self.coherence_weight = coherence_weight
    
#     def _mask_input(self, x5d):
#         """Apply spatial masking with sampling_rate; x5d: (B, T, C, H, W)."""
#         B, T, C, H, W = x5d.shape
#         device = x5d.device

#         # draw per-sample sampling rates if a range is provided
#         sr = self.sampling_rate
#         if isinstance(sr, (list, tuple)) and len(sr) == 2:
#             sr_vals = torch.empty(B, device=device).uniform_(float(sr[0]), float(sr[1]))
#         else:
#             sr_vals = torch.full((B,), float(sr), device=device)

#         # vectorized spatial mask (True == masked to NaN)
#         rnd = torch.rand(B, 1, 1, H, W, device=device)
#         mask2d = rnd > sr_vals.view(B, 1, 1, 1, 1)  # shape (B,1,1,H,W)
#         mask = mask2d.expand(B, T, C, H, W)
#         x = x5d.clone()
#         x[mask] = float("nan")
#         return x
    
#     def _denormalize_minmax(self, tensor_5d, minmax_list, comp_idx=None):
#         """
#         Denormalize a 5D tensor (B, T, C, H, W) given a list of (min_c, max_c)
#         for each component c.

#         Args:
#             tensor_5d (torch.Tensor): shape (batch, T, C, height, width)
#             minmax_list (Tuple[array, array]): 
#                 tuple of (min_values, max_values) arrays for each component c
#             comp_idx (torch.Tensor): shape (B, C) - indices mapping to global components

#         Returns:
#             denorm_5d (torch.Tensor): shape (batch, T, C, height, width)
#                 where each component c is mapped back to [min_c, max_c].
#         """
#         B, T, C, H, W = tensor_5d.shape
#         mins, maxs = minmax_list
        
#         if comp_idx is not None:
#             # Use comp_idx to map each depth level to correct statistics
#             for b in range(B):
#                 for c in range(C):
#                     # Get the global component index for this batch and depth
#                     global_c_idx = int(comp_idx[b, c].item())
#                     min_c = mins[global_c_idx]
#                     max_c = maxs[global_c_idx]
#                     tensor_5d[b, :, c, :, :] = tensor_5d[b, :, c, :, :] * (max_c - min_c) + min_c
#         else:
#             # Fallback: assume sequential component ordering
#             for c_idx, (min_c, max_c) in enumerate(zip(mins, maxs)):
#                 if c_idx < C:
#                     tensor_5d[:, :, c_idx, :, :] = tensor_5d[:, :, c_idx, :, :] * (max_c - min_c) + min_c

#         return tensor_5d
    
#     def _denormalize_zscore(self, tensor_5d, norm_list, comp_idx=None):
#         """
#         tensor_5d:  (B, T, C, H, W)
#         norm_list: (means, stds)  length = num_components
#         comp_idx:  (B, C) - indices mapping each C to global component position
#         returns    (B, T, C, H, W)
#         """
#         means, stds = norm_list
#         B, T, C, H, W = tensor_5d.shape
        
#         if comp_idx is not None:
#             # Use comp_idx to map each depth level to correct statistics
#             for b in range(B):
#                 for c in range(C):
#                     # Get the global component index for this batch and depth
#                     global_c_idx = int(comp_idx[b, c].item())
#                     mean_c = means[global_c_idx]
#                     std_c = stds[global_c_idx]
#                     tensor_5d[b, :, c, :, :] = tensor_5d[b, :, c, :, :] * std_c + mean_c
#         else:
#             # Fallback: assume sequential component ordering
#             for c_idx, (mean_c, std_c) in enumerate(zip(means, stds)):
#                 if c_idx < C:
#                     tensor_5d[:, :, c_idx, :, :] = tensor_5d[:, :, c_idx, :, :] * std_c + mean_c
#         return tensor_5d
    
#     def forward(self, batch):
#         """
#         Process each depth level through 4DVarNet, then combine with UNet
        
#         batch.input shape: (B, T, C, H, W) where C is depth/component levels
#         batch.tgt shape: (B, T, C, H, W)
#         """
#         B, T, C, H, W = batch.input.shape
        
#         # Process each depth level separately
#         depth_reconstructions = []
        
#         for c in range(C):
#             # Extract data for this depth level
#             depth_batch = batch._replace(
#                 input=batch.input[:, :, c, :, :],  # (B, T, H, W)
#                 tgt=batch.tgt[:, :, c, :, :] # (B, T, H, W)
#             )
#             # Apply 4DVarNet to this depth level
#             depth_recon = super().forward(depth_batch)  # (B, T, H, W)
#             depth_reconstructions.append(depth_recon)
        

#         return torch.stack(depth_reconstructions, dim=2) # (B, T, C, H, W)

#     def step(self, batch, phase=""):
#         """Modified step function to handle depth dimension"""
#         if self.training and batch.tgt.isfinite().float().mean() < 0.9:
#             return None, None
            
#         # Ensure proper shape handling for depth dimension
#         if batch.input.dim() == 5:  # (B, T, C, H, W)
#             B, T, C, H, W = batch.input.shape
#             # Keep depth dimension for processing
#             pass
#         else:  # (B, T*C, H, W) - reshape to add component dimension
#             B, TC, H, W = batch.input.shape
#             C = len(self.norm_stats[0])
#             T = TC // C
#             batch = batch._replace(
#                 input=batch.input.contiguous().view(B, T, C, H, W),  # (B, T, C, H, W)
#                 tgt=batch.tgt.contiguous().view(B, T, C, H, W)  # (B, T, C, H, W)
#             )
        
#         # Apply masking to each depth level
#         masked_input = batch.input.clone()
        
#         for i in range(B):
#             sr = self.sampling_rate
#             if isinstance(sr, (list, tuple)) and len(sr) == 2:
#                 sr = random.uniform(sr[0], sr[1])
            
#             # Create spatial mask and apply to all depth levels
#             spatial_mask = (torch.rand((H, W), device=batch.input.device) > sr)
#             for c in range(C):
#                 for t in range(T):
#                     masked_input[i, t, c][spatial_mask] = float("nan")
        
#         batch = batch._replace(input=masked_input)
        
#         # Forward pass
#         out = self(batch)  # (B, T, C, H, W)
#         target = batch.tgt  # (B, T, C, H, W)
        
#         out_4d = out.view(B, T * C, H, W)
#         target_4d = target.view(B, T * C, H, W)

#         rmse_loss = 0.0
#         rmsed = []
#         for c in range(C):
#             depth_out = out[:, :, c, :, :].contiguous().view(B, T, H, W)
#             depth_tgt = target[:, :, c, :, :].contiguous().view(B, T, H, W)
#             loss = self.weighted_mse(err=depth_out - depth_tgt, weight=self.rec_weight)
#             rmsed.append(loss)
#         rmse_loss = torch.stack(rmsed).mean()
        
#         self.log(f"{phase}_mse", 10000 * rmse_loss , prog_bar=True, on_step=False, on_epoch=True)
#         self.log(f"{phase}_loss", rmse_loss, prog_bar=True, on_step=False, on_epoch=True)
        
#         if self.solver.n_step > 0:
#             grad_loss = 0.0
#             prior_cost = 0.0
#             gls = []
#             pcs = []
#             for c in range(C):
#                 depth_out = out[:, :, c, :, :].contiguous().view(B, T, H, W)
#                 depth_tgt = target[:, :, c, :, :].contiguous().view(B, T, H, W)
#                 depth_batch = batch._replace(
#                     input=batch.input[:, :, c, :, :],
#                     tgt=batch.tgt[:, :, c, :, :]
#                 )
#                 grad_loss_d = self.weighted_mse(
#                     err=kfilts.sobel(depth_out) - kfilts.sobel(depth_tgt), 
#                     weight=self.rec_weight
#                 )
#                 gls.append(grad_loss_d)
#                 pcs.append(self.solver.prior_cost(self.solver.init_state(depth_batch, depth_out)))
#             prior_cost = torch.stack(pcs).mean()
#             grad_loss = torch.stack(gls).mean()
            
#             self.log(f"{phase}_gloss", grad_loss, prog_bar=True, on_step=False, on_epoch=True)
#             self.log(f"{phase}_prior_cost", prior_cost, prog_bar=True, on_step=False, on_epoch=True)

#             training_loss = 20 * rmse_loss +  prior_cost + 20 * grad_loss
#             return training_loss, out
#         return rmse_loss, out

#     def test_step(self, batch, batch_idx):
#         # reset buffer on first batch
#         if batch_idx == 0:
#             self.test_data = [] 

#         raw_input = batch.input.clone() 
        
#         # Optimized inference path - skip expensive operations
#         with torch.inference_mode():
#             masked_input = self._mask_input(raw_input)
#             masked_batch = batch._replace(input=masked_input)
#             out = self._fast_forward(masked_batch)
        
#         # denorm to (B, T, C, H, W)
#         # Extract comp_idx from batch if available
#         comp_idx = getattr(batch, 'comp_idx', None)
        
#         if self.norm_type == "z_score":
#             raw_input_den = self._denormalize_zscore(raw_input,      self.norm_stats, comp_idx)
#             masked_den    = self._denormalize_zscore(masked_batch.input, self.norm_stats, comp_idx)
#             tgt_den       = self._denormalize_zscore(batch.tgt,      self.norm_stats, comp_idx)
#             out_den       = self._denormalize_zscore(out,          self.norm_stats, comp_idx)
#         else:
#             raw_input_den = self._denormalize_minmax(raw_input,      self.norm_stats, comp_idx)
#             masked_den    = self._denormalize_minmax(masked_batch.input, self.norm_stats, comp_idx)
#             tgt_den       = self._denormalize_minmax(batch.tgt,      self.norm_stats, comp_idx)
#             out_den       = self._denormalize_minmax(out,          self.norm_stats, comp_idx)

#         self.test_data.append(torch.stack(
#             [raw_input_den.cpu(), masked_den.cpu(), tgt_den.cpu(), out_den.detach().cpu()],
#             dim=1,
#         ))
    
#     def _fast_forward(self, batch):
#         """Optimized forward pass for inference"""
#         # Skip gradient computation and use reduced steps
#         if hasattr(self.solver, 'n_step') and self.solver.n_step > 0:
#             # Temporarily reduce steps for inference
#             original_steps = self.solver.n_step
#             self.solver.n_step = max(1, original_steps // 3)  # Use 1/3 of training steps
#             try:
#                 result = self(batch=batch)
#             finally:
#                 self.solver.n_step = original_steps  # Restore original
#             return result
#         else:
#             return self(batch=batch)
    
#     @property
#     def test_quantities(self):
#         # keep your ordering & names
#         return ['input', 'inp', 'tgt', 'out']
    


class DepthPositionEnc(nn.Module):
    """
    Turns a depth *value* (index or meters) into C channels, broadcast over HxW.
    mode = 'sin' (Fourier-style) or 'mlp' (learned).
    """
    def __init__(self, out_channels: int = 8, mode: str = 'sin', min_freq=1.0, max_freq=32.0, mlp_hidden=32):
        super().__init__()
        self.out_channels = out_channels
        self.mode = mode
        if mode == 'mlp':
            self.mlp = nn.Sequential(
                nn.Linear(1, mlp_hidden), nn.ReLU(),
                nn.Linear(mlp_hidden, out_channels)
            )
        else:
            # build fixed frequencies for sin/cos embeddings
            n = out_channels // 2
            self.register_buffer(
                'freqs',
                torch.exp(torch.linspace(np.log(min_freq), np.log(max_freq), n))
            )

    @torch.no_grad()
    def _sinusoidal(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B*D, 1) depths normalized in [0,1] or meters rescaled
        returns (B*D, C)
        """
        # (B*D, n)
        phases = z * self.freqs[None, :]
        sin = torch.sin(phases)
        cos = torch.cos(phases)
        feat = torch.cat([sin, cos], dim=-1)
        # pad if odd C
        if feat.shape[-1] < self.out_channels:
            feat = torch.cat([feat, torch.zeros_like(feat[..., :1])], dim=-1)
        return feat

    def forward(self, depths_1d: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        depths_1d: (B, D) tensor of *normalized* depths in [0,1] (or any scalar scale).
        Returns: (B, D, C, H, W)
        """
        B, D = depths_1d.shape
        z = depths_1d.reshape(B*D, 1)

        if self.mode == 'mlp':
            feat = self.mlp(z)
        else:
            feat = self._sinusoidal(z)

        feat = feat.view(B, D, -1, 1, 1).expand(B, D, feat.shape[-1], H, W)
        return feat
    

    def forward(self, depth_idx: torch.Tensor) -> torch.Tensor:
        """
        depth_idx: (D,) long or int32
        returns: (D, E) float32
        """
        return self.pe[depth_idx]

class GradSolverDepth(transfert.GradSolver_Fasc):
    """
    Gradient-based solver that processes each depth level separately.
    """

    def __init__(self, depth_encoder: Optional[DepthPositionEnc] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth_encoder = depth_encoder
        self.depth_proj = None
    
    def set_depth_projection(self, in_ch: int, out_ch: int):
        self.depth_proj = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor, depth_idx: Optional[torch.Tensor] = None):
        """
        x: (B, T, D, H, W)
        We’ll reshape to (B, T*D, H, W) for the base solver.
        """
        if x.dim() == 5:
            B, T, D, H, W = x.shape
        else:
            raise ValueError("Expected x of shape (B, T, D, H, W)")

        # optional depth conditioning
        if self.depth_encoder is not None and depth_idx is not None:
            # depth_idx shape should be (D,)
            emb = self.depth_encoder(depth_idx.to(x.device))   # (D, E)
            # broadcast to (B, E, D, H, W)
            emb_hw = emb.view(1, -1, D, 1, 1).expand(B, -1, D, H, W)
            # concat on channel dim before flattening D
            x = torch.cat([x, emb_hw], dim=1)  # (B, TC+E, D, H, W)
            TC = x.size(1)

        # merge depth into batch
        x2d = x.permute(0, 2, 1, 3, 4).contiguous().view(B*D, TC, H, W)  # (B*D, TC, H, W)

        # if we need to align embedding channels to solver input, do it here
        if self.depth_encoder is not None and self.depth_proj is not None:
            x2d = self.depth_proj(x2d)

        # run shared 2-D solver
        out2d = super().forward(x2d)  # Call inherited forward function

        # un-merge depth
        TC_out = out2d.size(1)
        out = out2d.view(B, D, TC_out, H, W).permute(0, 2, 1, 3, 4).contiguous()  # (B, TC_out, D, H, W)
        return out
    
# class Lit4dVarNet_LazyTransfert(transfert.Lit4dVarNet_Fasc):
#     """
#     Handles both full-depth (T,D,H,W) and per-depth (T,H,W) patches.
#     """

#     def _get_full_component_vector(self, device, dtype=torch.long):
#         if not hasattr(self, "_full_comp_tensor") or self._full_comp_tensor is None:
#             dm = self.trainer.datamodule
#             if hasattr(dm, "input_da"):
#                 xr_obj = dm.input_da
#                 if "component" in xr_obj.coords:
#                     comp_vals = xr_obj["component"].values
#                 else:
#                     first_var = next(iter(xr_obj.data_vars))
#                     comp_vals = xr_obj[first_var]["component"].values
#             else:
#                 raise RuntimeError("Datamodule has no input_da with 'component' coord")
#             self._full_comp_tensor = torch.as_tensor(comp_vals, dtype=torch.long, device="cpu")
#         return self._full_comp_tensor.to(device=device, dtype=dtype)

#     def _labels_to_positions(self, comp_labels: torch.Tensor, full_comp: torch.Tensor):
#         orig_shape = tuple(comp_labels.shape)
#         flat_labels = comp_labels.reshape(-1).long()
#         sort_idx = torch.argsort(full_comp)
#         full_sorted = full_comp[sort_idx]
#         pos_sorted = torch.bucketize(flat_labels.to(full_sorted.device), full_sorted).clamp_(0, full_sorted.numel() - 1)
#         mismatch_mask = (full_sorted[pos_sorted] != flat_labels)
#         if mismatch_mask.any():
#             bad = flat_labels[mismatch_mask][:10].tolist()
#             raise ValueError(f"Component labels not found in full_comp: {bad} (showing up to 10)")
#         inv_sort = torch.empty_like(sort_idx)
#         inv_sort[sort_idx] = torch.arange(sort_idx.numel(), device=sort_idx.device)
#         pos_flat = inv_sort[pos_sorted]
#         return pos_flat.view(*orig_shape)

#     def _expand_param(self, param_1d, ref_tensor: torch.Tensor, comp_pos: torch.Tensor | None):
#         if ref_tensor.ndim == 4:
#             # (B,T,H,W) → scalar broadcast
#             return torch.as_tensor(param_1d, device=ref_tensor.device, dtype=ref_tensor.dtype).mean()
#         t = torch.as_tensor(param_1d, device=ref_tensor.device, dtype=ref_tensor.dtype)
#         if comp_pos is None:
#             return t
#         B, T, D, *_ = ref_tensor.shape
#         if comp_pos.dim() == 1:
#             comp_pos = comp_pos.view(B, 1)
#         elif comp_pos.numel() == B * D:
#             comp_pos = comp_pos.view(B, D)
#         flat = comp_pos.view(-1)
#         gathered = t[flat]
#         tail_ones = [1] * (ref_tensor.ndim - 3)
#         return gathered.view(B, 1, D, *tail_ones)

#     def build_denorm(self, norm_stats, norm_type: str, ref_tensor: torch.Tensor, comp_idx: torch.Tensor | None):
#         """
#         Build denormalization lambda that supports both 4D (B,T,H,W) and 5D (B,T,D,H,W) tensors.
#         """
#         if norm_stats is None:
#             return lambda z: z

#         comp_pos = None
#         if comp_idx is not None:
#             full_comp = self._get_full_component_vector(ref_tensor.device, torch.long)
#             comp_pos = self._labels_to_positions(comp_idx, full_comp)

#         if norm_type == "z_score":
#             mean, std = norm_stats
#             mean_t = torch.as_tensor(mean, device=ref_tensor.device, dtype=ref_tensor.dtype)
#             std_t = torch.as_tensor(std, device=ref_tensor.device, dtype=ref_tensor.dtype)

#             def denorm_fn(z):
#                 # Per-depth or global fallback
#                 if z.ndim == 4:  # (B,T,H,W)
#                     m = mean_t.mean().view(1, 1, 1, 1)
#                     s = std_t.mean().view(1, 1, 1, 1)
#                     return z * s + m
#                 elif z.ndim == 5:  # (B,T,D,H,W)
#                     if comp_pos is None:
#                         if mean_t.ndim == 1 and mean_t.numel() == z.shape[2]:
#                             mean_b = mean_t.view(1, 1, z.shape[2], 1, 1)
#                             std_b = std_t.view(1, 1, z.shape[2], 1, 1)
#                         else:
#                             mean_b = mean_t.mean().view(1, 1, 1, 1, 1)
#                             std_b = std_t.mean().view(1, 1, 1, 1, 1)
#                         return z * std_b + mean_b
#                     mean_sel = self._expand_param(mean_t, z, comp_pos)
#                     std_sel = self._expand_param(std_t, z, comp_pos)
#                     return z * std_sel + mean_sel
#                 else:
#                     raise ValueError(f"Unsupported tensor shape: {z.shape}")

#             return denorm_fn

#         else:  # min-max
#             vmin, vmax = norm_stats
#             vmin_t = torch.as_tensor(vmin, device=ref_tensor.device, dtype=ref_tensor.dtype)
#             vmax_t = torch.as_tensor(vmax, device=ref_tensor.device, dtype=ref_tensor.dtype)

#             def denorm_fn(z):
#                 if z.ndim == 4:
#                     vmin_s = vmin_t.mean().view(1, 1, 1, 1)
#                     vmax_s = vmax_t.mean().view(1, 1, 1, 1)
#                     return z * (vmax_s - vmin_s) + vmin_s
#                 elif z.ndim == 5:
#                     if comp_pos is None:
#                         if vmin_t.ndim == 1 and vmin_t.numel() == z.shape[2]:
#                             vmin_b = vmin_t.view(1, 1, z.shape[2], 1, 1)
#                             vmax_b = vmax_t.view(1, 1, z.shape[2], 1, 1)
#                         else:
#                             vmin_b = vmin_t.mean().view(1, 1, 1, 1, 1)
#                             vmax_b = vmax_t.mean().view(1, 1, 1, 1, 1)
#                         return z * (vmax_b - vmin_b) + vmin_b
#                     vmin_sel = self._expand_param(vmin_t, z, comp_pos)
#                     vmax_sel = self._expand_param(vmax_t, z, comp_pos)
#                     return z * (vmax_sel - vmin_sel) + vmin_sel
#                 else:
#                     raise ValueError(f"Unsupported tensor shape: {z.shape}")

#             return denorm_fn
        
#     def test_step(self, batch, batch_idx):
#         """Override to ensure correct denormalization for per-depth slices."""
#         if batch_idx == 0:
#             self.test_data = []

#         batch = batch._replace(input=self._apply_mask(batch.input))
#         out = self(batch=batch)

#         # --- Shape alignment ---
#         if out.ndim == 6 and out.shape[-1] == 1:
#             out = out.squeeze(-1)

#         # force [B,T,D,H,W] or [B,T,H,W]
#         if out.ndim == 4 and batch.input.ndim == 5:
#             # (B,T,H,W) ← (B,T,1,H,W)
#             out = out.unsqueeze(2)

#         # --- Denormalization ---
#         stats = self.norm_stats() if callable(self.norm_stats) else self.norm_stats
#         comp_idx = getattr(batch, "comp_idx", None)
#         denorm = self.build_denorm(stats, self.norm_type, batch.input, comp_idx)

#         with torch.no_grad():
#             tgt_den = denorm(batch.tgt).detach()
#             out_den = denorm(out).detach()

#             # ---- store CPU copies ----
#             tgt_cpu = tgt_den.cpu()
#             out_cpu = out_den.cpu()
#             del tgt_den, out_den, out, batch
#             torch.cuda.empty_cache() if torch.cuda.is_available() else None

#         self.test_data.append(torch.stack([tgt_cpu, out_cpu], dim=1))

#     @property
#     def test_quantities(self):
#         return ["tgt", "out"]
    
# class Lit4dVarNet_LazyTransfert_z(Lit4dVarNet_LazyTransfert):
#     """
#     Same as Lit4dVarNet_LazyTransfert, but adapted for z-models
#     that operate on per-depth slices or flattened (T×D) representations.
#     """
#     def _expand_param(self, param_1d, ref_tensor: torch.Tensor, comp_pos: torch.Tensor | None):
#         if ref_tensor.ndim == 4:
#             # (B,T,H,W) → scalar broadcast
#             return torch.as_tensor(param_1d, device=ref_tensor.device, dtype=ref_tensor.dtype).mean()

#         t = torch.as_tensor(param_1d, device=ref_tensor.device, dtype=ref_tensor.dtype)
#         if comp_pos is None:
#             return t

#         B, T, D, *_ = ref_tensor.shape

#         # --- FIXED SHAPE HANDLING ---
#         if comp_pos.dim() == 2 and comp_pos.shape[1] == 1:
#             comp_pos = comp_pos.squeeze(1)  # handle shape [B,1] → [B]

#         if comp_pos.dim() == 1:
#             # One label per batch, broadcast across depths
#             if comp_pos.numel() == B:
#                 flat = comp_pos.repeat_interleave(D)  # → length = B*D
#             else:
#                 flat = comp_pos.view(-1)
#         elif comp_pos.numel() == B * D:
#             flat = comp_pos.view(-1)
#         else:
#             raise ValueError(f"Unexpected comp_pos shape: {comp_pos.shape}, expected [B], [B,1], or [B,D]")

#         gathered = t[flat]
#         tail_ones = [1] * (ref_tensor.ndim - 3)
#         return gathered.view(B, 1, D, *tail_ones)

#     def forward(self, batch):
#         # Flatten (T,D) → TD for z-models
#         if batch.input.ndim == 5:
#             shape = batch.input.shape
#             new_shape = (shape[0], shape[1] * shape[2], shape[3], shape[4])
#             batch = batch._replace(
#                 input=batch.input.reshape(new_shape),
#                 tgt=batch.tgt.reshape(new_shape) if batch.tgt.ndim == 5 else batch.tgt
#             )

#         if self.solver.n_step > 0:
#             return super().forward(batch)

#         extra = {}
#         state = self.solver.init_state(batch, None)
#         if hasattr(batch, "depth_idx"):
#             extra["depth_idx"] = batch.depth_idx.to(state.device)
#         elif isinstance(batch, dict) and "depth_idx" in batch:
#             extra["depth_idx"] = batch["depth_idx"].to(state.device)

#         return self.solver.prior_cost.forward_ae(state, extra=extra)

    # def test_step(self, batch, batch_idx):
    #     """Override to ensure correct denormalization and reshaping for per-depth slices."""
    #     if batch_idx == 0:
    #         self.test_data = []

    #     # --- Apply observation mask ---
    #     batch = batch._replace(input=self._apply_mask(batch.input))

    #     # --- Forward pass ---
    #     out = self(batch=batch)

    #     # --- Shape alignment ---
    #     if out.ndim == 6 and out.shape[-1] == 1:
    #         out = out.squeeze(-1)

    #     if out.ndim == 4 and batch.input.ndim == 5:
    #         # (B,T,H,W) → (B,T,1,H,W)
    #         out = out.unsqueeze(2)

    #     # --- Unflatten (T×D → T,D) if needed ---
    #     if out.ndim == 5 and out.shape[1] not in [batch.input.shape[1], 1]:
    #         # Try to infer T and D
    #         if hasattr(self.trainer.datamodule, "xrds_kw"):
    #             pdims = self.trainer.datamodule.xrds_kw.patch_dims
    #             T, D = pdims.time, pdims.component
    #         else:
    #             T, D = 15, 5
    #         if T * D == out.shape[1]:
    #             out = out.view(out.shape[0], T, D, *out.shape[2:])

    #     # --- NEW: Remove any leftover singleton dimensions ---
    #     while out.ndim > 5 and 1 in out.shape:
    #         out = out.squeeze(3)  # remove the 1 between D and H if present
    #     if out.ndim == 6:
    #         out = out.squeeze()  # fallback, ensures [B,T,D,H,W]

    #     # --- Denormalization ---
    #     stats = self.norm_stats() if callable(self.norm_stats) else self.norm_stats
    #     comp_idx = getattr(batch, "comp_idx", None)
    #     denorm = self.build_denorm(stats, self.norm_type, batch.input, comp_idx)

    #     with torch.no_grad():
    #         tgt_den = denorm(batch.tgt).detach()
    #         out_den = denorm(out).detach()

    #         # Move to CPU for accumulation
    #         tgt_cpu = tgt_den.cpu()
    #         out_cpu = out_den.cpu()

    #         # Release GPU memory
    #         del tgt_den, out_den, out, batch
    #         torch.cuda.empty_cache() if torch.cuda.is_available() else None

    #     # --- Store aligned tensors ---
    #     self.test_data.append(torch.stack([tgt_cpu, out_cpu], dim=1))
    # def test_step(self, batch, batch_idx):
    #     """Same behavior as Lit4dVarNet_LazyTransfert.test_step, with depth_idx support."""
    #     if batch_idx == 0:
    #         self.test_data = []

    #     # Apply observation mask
    #     batch = batch._replace(input=self._apply_mask(batch.input))

    #     # Forward pass
    #     out = self(batch=batch)

    #     # --- Shape alignment (identical to base class) ---
    #     if out.ndim == 6 and out.shape[-1] == 1:
    #         out = out.squeeze(-1)

    #     # force [B,T,D,H,W] or [B,T,H,W]
    #     if out.ndim == 4 and batch.input.ndim == 5:
    #         # (B,T,H,W) → (B,T,1,H,W)
    #         out = out.unsqueeze(2)

    #     # --- Denormalization (same as base, but comp_idx falls back to depth_idx) ---
    #     stats = self.norm_stats() if callable(self.norm_stats) else self.norm_stats

    #     # Prefer comp_idx if provided; otherwise use depth_idx (common in z-model dataloaders)
    #     comp_idx = getattr(batch, "comp_idx", None)
    #     if comp_idx is None:
    #         comp_idx = getattr(batch, "depth_idx", None)

    #     denorm = self.build_denorm(stats, self.norm_type, batch.input, comp_idx)

    #     with torch.no_grad():
    #         tgt_den = denorm(batch.tgt).detach()
    #         out_den = denorm(out).detach()

    #         # Store CPU copies
    #         tgt_cpu = tgt_den.cpu()
    #         out_cpu = out_den.cpu()

    #         # Cleanup
    #         del tgt_den, out_den, out, batch
    #         if torch.cuda.is_available():
    #             torch.cuda.empty_cache()

    #     # Stack like base class: shape [B, 2, ...]
    #     self.test_data.append(torch.stack([tgt_cpu, out_cpu], dim=1))
    # # def test_step(self, batch, batch_idx):
    #     if batch_idx == 0:
    #         self.test_data = []
    #         self.test_save_dir = Path(self.trainer.default_root_dir) / "test_batches"
    #         self.test_save_dir.mkdir(exist_ok=True, parents=True)

    #     batch = batch._replace(input=self._apply_mask(batch.input))
    #     out = self(batch=batch)

    #     if out.ndim == 6 and out.shape[-1] == 1:
    #         out = out.squeeze(-1)
    #     if out.ndim == 4 and batch.input.ndim == 5:
    #         out = out.unsqueeze(2)

    #     if out.ndim == 5 and out.shape[1] not in [batch.input.shape[1], 1]:
    #         if hasattr(self.trainer.datamodule, "xrds_kw"):
    #             pdims = self.trainer.datamodule.xrds_kw.patch_dims
    #             T, D = pdims.time, pdims.component
    #         else:
    #             T, D = 15, 5
    #         if T * D == out.shape[1]:
    #             out = out.view(out.shape[0], T, D, *out.shape[2:])
    #     while out.ndim > 5 and 1 in out.shape:
    #         out = out.squeeze(3)

    #     stats = self.norm_stats() if callable(self.norm_stats) else self.norm_stats
    #     comp_idx = getattr(batch, "comp_idx", None)
    #     denorm = self.build_denorm(stats, self.norm_type, batch.input, comp_idx)

    #     with torch.no_grad():
    #         tgt_den = denorm(batch.tgt).detach()
    #         out_den = denorm(out).detach()
    #         tgt_cpu = tgt_den.cpu().half()
    #         out_cpu = out_den.cpu().half()

    #     # Save each batch to disk
    #     torch.save(
    #         {"tgt": tgt_cpu, "out": out_cpu},
    #         self.test_save_dir / f"batch_{batch_idx:05d}.pt"
    #     )

    #     # Keep only 1 sample in memory for visual inspection
    #     if batch_idx % 50 == 0:
    #         self.test_data.append(torch.stack([tgt_cpu[:1], out_cpu[:1]], dim=1))

    #     del tgt_den, out_den, out, batch, tgt_cpu, out_cpu
    #     torch.cuda.empty_cache()

    #     return {}

    # @property
    # def test_quantities(self):
    #     return ["tgt", "out"]
# class Lit4dVarNet_LazyTransfert(transfert.Lit4dVarNet_Fasc):
#     """
#     Handles both full-depth (T,D,H,W) and per-depth (T,H,W) patches.
#     Compatible with datamodules providing either `depth_idx` (preferred) or legacy `comp_idx`.
#     """

#     # ===============================
#     # --- utilities for component handling
#     # ===============================

#     def _get_full_component_vector(self, device, dtype=torch.long):
#         if not hasattr(self, "_full_comp_tensor") or self._full_comp_tensor is None:
#             dm = self.trainer.datamodule
#             if hasattr(dm, "input_da"):
#                 xr_obj = dm.input_da
#                 if "component" in xr_obj.coords:
#                     comp_vals = xr_obj["component"].values
#                 else:
#                     first_var = next(iter(xr_obj.data_vars))
#                     comp_vals = xr_obj[first_var]["component"].values
#             else:
#                 raise RuntimeError("Datamodule has no input_da with 'component' coord")
#             self._full_comp_tensor = torch.as_tensor(comp_vals, dtype=torch.long, device="cpu")
#         return self._full_comp_tensor.to(device=device, dtype=dtype)

#     def _labels_to_positions(self, labels: torch.Tensor, full_comp: torch.Tensor):
#         """
#         Map label values (depth_idx or comp_idx) to integer positions in the global component list.
#         Works with either discrete or continuous labels by finding the nearest match.
#         """
#         orig_shape = tuple(labels.shape)
#         flat_labels = labels.reshape(-1).float()
#         full_sorted, sort_idx = torch.sort(full_comp.float())

#         # approximate nearest index
#         pos_sorted = torch.bucketize(flat_labels, full_sorted).clamp_(0, full_sorted.numel() - 1)
#         inv_sort = torch.empty_like(sort_idx)
#         inv_sort[sort_idx] = torch.arange(sort_idx.numel(), device=sort_idx.device)
#         pos_flat = inv_sort[pos_sorted]
#         return pos_flat.view(*orig_shape)

#     def _expand_param(self, param_1d, ref_tensor: torch.Tensor, depth_pos: torch.Tensor | None):
#         """
#         Expand 1D parameter vector (mean/std/etc) to match spatial tensor based on depth positions.
#         """
#         if ref_tensor.ndim == 4:  # (B,T,H,W)
#             return torch.as_tensor(param_1d, device=ref_tensor.device, dtype=ref_tensor.dtype).mean()

#         t = torch.as_tensor(param_1d, device=ref_tensor.device, dtype=ref_tensor.dtype)
#         if depth_pos is None:
#             return t

#         B, T, D, *_ = ref_tensor.shape
#         if depth_pos.dim() == 1:
#             depth_pos = depth_pos.view(B, 1)
#         elif depth_pos.numel() == B * D:
#             depth_pos = depth_pos.view(B, D)
#         flat = depth_pos.view(-1)
#         gathered = t[flat]
#         tail_ones = [1] * (ref_tensor.ndim - 3)
#         return gathered.view(B, 1, D, *tail_ones)

#     # ===============================
#     # --- normalization / denormalization
#     # ===============================

#     def build_denorm(self, norm_stats, norm_type: str, ref_tensor: torch.Tensor, depth_idx: torch.Tensor | None):
#         """
#         Build denormalization lambda supporting both 4D (B,T,H,W) and 5D (B,T,D,H,W) tensors.
#         Backward compatible with legacy comp_idx via automatic fallback.
#         """
#         if norm_stats is None:
#             return lambda z: z

#         # --- retrocompatibility ---
#         if depth_idx is None and hasattr(self.trainer.datamodule, "input_da"):
#             # try comp_idx from dataloader batch if available
#             depth_idx = getattr(self.trainer.datamodule, "comp_idx", None)

#         full_comp = self._get_full_component_vector(ref_tensor.device, torch.long)
#         depth_pos = None
#         if depth_idx is not None:
#             # convert continuous normalized depth_idx to discrete position if needed
#             depth_pos = self._labels_to_positions(depth_idx, full_comp)

#         if norm_type == "z_score":
#             mean, std = norm_stats
#             mean_t = torch.as_tensor(mean, device=ref_tensor.device, dtype=ref_tensor.dtype)
#             std_t = torch.as_tensor(std, device=ref_tensor.device, dtype=ref_tensor.dtype)

#             def denorm_fn(z):
#                 if z.ndim == 4:  # (B,T,H,W)
#                     m = mean_t.mean().view(1, 1, 1, 1)
#                     s = std_t.mean().view(1, 1, 1, 1)
#                     return z * s + m
#                 elif z.ndim == 5:  # (B,T,D,H,W)
#                     if depth_pos is None:
#                         if mean_t.ndim == 1 and mean_t.numel() == z.shape[2]:
#                             mean_b = mean_t.view(1, 1, z.shape[2], 1, 1)
#                             std_b = std_t.view(1, 1, z.shape[2], 1, 1)
#                         else:
#                             mean_b = mean_t.mean().view(1, 1, 1, 1, 1)
#                             std_b = std_t.mean().view(1, 1, 1, 1, 1)
#                         return z * std_b + mean_b
#                     mean_sel = self._expand_param(mean_t, z, depth_pos)
#                     std_sel = self._expand_param(std_t, z, depth_pos)
#                     return z * std_sel + mean_sel
#                 else:
#                     raise ValueError(f"Unsupported tensor shape: {z.shape}")

#             return denorm_fn

#         else:  # min-max normalization
#             vmin, vmax = norm_stats
#             vmin_t = torch.as_tensor(vmin, device=ref_tensor.device, dtype=ref_tensor.dtype)
#             vmax_t = torch.as_tensor(vmax, device=ref_tensor.device, dtype=ref_tensor.dtype)

#             def denorm_fn(z):
#                 if z.ndim == 4:
#                     vmin_s = vmin_t.mean().view(1, 1, 1, 1)
#                     vmax_s = vmax_t.mean().view(1, 1, 1, 1)
#                     return z * (vmax_s - vmin_s) + vmin_s
#                 elif z.ndim == 5:
#                     if depth_pos is None:
#                         if vmin_t.ndim == 1 and vmin_t.numel() == z.shape[2]:
#                             vmin_b = vmin_t.view(1, 1, z.shape[2], 1, 1)
#                             vmax_b = vmax_t.view(1, 1, z.shape[2], 1, 1)
#                         else:
#                             vmin_b = vmin_t.mean().view(1, 1, 1, 1, 1)
#                             vmax_b = vmax_t.mean().view(1, 1, 1, 1, 1)
#                         return z * (vmax_b - vmin_b) + vmin_b
#                     vmin_sel = self._expand_param(vmin_t, z, depth_pos)
#                     vmax_sel = self._expand_param(vmax_t, z, depth_pos)
#                     return z * (vmax_sel - vmin_sel) + vmin_sel
#                 else:
#                     raise ValueError(f"Unsupported tensor shape: {z.shape}")

#             return denorm_fn

#     # ===============================
#     # --- test step (retrocompatible)
#     # ===============================

#     def test_step(self, batch, batch_idx):
#         """Support both `depth_idx` and legacy `comp_idx` batches."""
#         if batch_idx == 0:
#             self.test_data = []

#         # Observation mask
#         batch = batch._replace(input=self._apply_mask(batch.input))
#         out = self(batch=batch)

#         # Shape normalization
#         if out.ndim == 6 and out.shape[-1] == 1:
#             out = out.squeeze(-1)
#         if out.ndim == 4 and batch.input.ndim == 5:
#             out = out.unsqueeze(2)

#         # Retrieve normalization stats
#         stats = self.norm_stats() if callable(self.norm_stats) else self.norm_stats

#         # --- unified access ---
#         depth_idx = getattr(batch, "depth_idx", None)
#         if depth_idx is None:
#             depth_idx = getattr(batch, "comp_idx", None)  # fallback

#         denorm = self.build_denorm(stats, self.norm_type, batch.input, depth_idx)
#         # After denormalization
#         tgt_raw = batch.tgt.clone().cpu()  # still normalized
#         tgt_den = denorm(batch.tgt).cpu()  # denormalized
#         out_den = denorm(out).cpu()

#         # # Compare means/stds    
#         # print("tgt_raw mean:", tgt_raw.mean().item(), "std:", tgt_raw.std().item())
#         # print("tgt_den mean:", tgt_den.mean().item(), "std:", tgt_den.std().item())

#         # # Check depth indexing
#         # print("depth_idx:", getattr(batch, 'depth_idx', None))
#         # print("component coord:", dm.input_da['component'].values[:10])

#         # # If per-depth patch:
#         # print("mean/std for selected component:", means[int(comp_idx_torch.item())], stds[int(comp_idx_torch.item())])

#         with torch.no_grad():
#             tgt_den = denorm(batch.tgt).detach()
#             out_den = denorm(out).detach()

#             tgt_cpu = tgt_den.cpu()
#             out_cpu = out_den.cpu()

#             del tgt_den, out_den, out, batch
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()

#         self.test_data.append(torch.stack([tgt_cpu, out_cpu], dim=1))

#     @property
#     def test_quantities(self):
#         return ["tgt", "out"]
class Lit4dVarNet_LazyTransfert_debug(transfert.Lit4dVarNet_Fasc):
    """
    Handles both full-depth (B,T,D,H,W) and per-depth (B,T,H,W) patches.
    Compatible with datamodules providing:
      • comp_idx  – discrete integer component index (for normalization)
      • depth_idx – normalized continuous depth coordinate (for embeddings)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # toggle this to enable/disable debug prints
        self._debug_inference = True
        self._full_comp_tensor = None

    # ====================================================
    # --- Utilities for discrete/continuous component indexing
    # ====================================================

    def _get_full_component_vector(self, device, dtype=torch.long):
        """Cache the full component coordinate vector from datamodule."""
        if not hasattr(self, "_full_comp_tensor") or self._full_comp_tensor is None:
            dm = self.trainer.datamodule
            xr_obj = getattr(dm, "input_da", None)
            if xr_obj is None:
                raise RuntimeError("Datamodule has no input_da with 'component' coord")

            if "component" in xr_obj.coords:
                comp_vals = xr_obj["component"].values
            else:
                first_var = next(iter(xr_obj.data_vars))
                comp_vals = xr_obj[first_var]["component"].values

            self._full_comp_tensor = torch.as_tensor(comp_vals, dtype=torch.long, device="cpu")

            if self._debug_inference:
                print("[_get_full_component_vector] built full_comp_tensor with shape:",
                      self._full_comp_tensor.shape, "dtype:", self._full_comp_tensor.dtype)

        full_comp = self._full_comp_tensor.to(device=device, dtype=dtype)
        if self._debug_inference:
            print("[_get_full_component_vector] returning full_comp on device", device,
                  "shape:", full_comp.shape, "dtype:", full_comp.dtype)
        return full_comp

    def _labels_to_positions(self, labels: torch.Tensor, full_comp: torch.Tensor):
        """
        Map (possibly continuous) depth labels to discrete integer positions
        within the full component vector.
        """
        if self._debug_inference:
            print("[_labels_to_positions] ENTER")
            print("  labels.shape:", labels.shape, "labels.dtype:", labels.dtype)
            print("  full_comp.shape:", full_comp.shape, "full_comp.dtype:", full_comp.dtype)

        flat_labels = labels.reshape(-1).float()
        if self._debug_inference:
            print("  flat_labels.shape:", flat_labels.shape, "flat_labels.dtype:", flat_labels.dtype)

        full_sorted, sort_idx = torch.sort(full_comp.float())
        if self._debug_inference:
            print("  full_sorted.shape:", full_sorted.shape)
            print("  sort_idx.shape:", sort_idx.shape)

        pos_sorted = torch.bucketize(flat_labels, full_sorted).clamp_(0, full_sorted.numel() - 1)
        if self._debug_inference:
            print("  pos_sorted.shape:", pos_sorted.shape)

        inv_sort = torch.empty_like(sort_idx)
        inv_sort[sort_idx] = torch.arange(sort_idx.numel(), device=sort_idx.device)
        if self._debug_inference:
            print("  inv_sort.shape:", inv_sort.shape)

        pos = inv_sort[pos_sorted].view_as(labels)
        if self._debug_inference:
            print("  pos (final comp_pos) shape:", pos.shape)
            print("[_labels_to_positions] EXIT")
        return pos

    def _expand_param(self, param_1d, ref_tensor: torch.Tensor, comp_pos: torch.Tensor | None):
        """
        Expand 1-D parameter (mean/std/etc.) to match ref_tensor shape along depth dimension.
        """
        if self._debug_inference:
            print("[_expand_param] ENTER")
            print("  param_1d.shape:", np.shape(param_1d))
            print("  ref_tensor.shape:", ref_tensor.shape)
            if comp_pos is not None:
                print("  comp_pos.shape:", comp_pos.shape)
            else:
                print("  comp_pos is None")

        if ref_tensor.ndim == 4:  # (B,T,H,W)
            t = torch.as_tensor(param_1d, device=ref_tensor.device, dtype=ref_tensor.dtype)
            if self._debug_inference:
                print("  ref_tensor.ndim == 4, returning mean scalar")
            return t.mean()

        t = torch.as_tensor(param_1d, device=ref_tensor.device, dtype=ref_tensor.dtype)
        if comp_pos is None:
            if self._debug_inference:
                print("  comp_pos is None and ref_tensor.ndim != 4 -> returning t as is")
                print("[_expand_param] EXIT")
            return t

        B, T, D, *_ = ref_tensor.shape
        if comp_pos.dim() == 1:
            if self._debug_inference:
                print("  comp_pos.dim() == 1, reshaping to (B,1)")
            comp_pos = comp_pos.view(B, 1)
        elif comp_pos.numel() == B * D:
            if self._debug_inference:
                print("  comp_pos.numel == B*D, reshaping to (B,D)")
            comp_pos = comp_pos.view(B, D)

        if self._debug_inference:
            print("  comp_pos reshaped to:", comp_pos.shape)

        gathered = t[comp_pos.view(-1)]
        tail_ones = [1] * (ref_tensor.ndim - 3)

        if self._debug_inference:
            print("  gathered.shape:", gathered.shape)
            print("  tail_ones:", tail_ones)

        out = gathered.view(B, 1, D, *tail_ones)

        if self._debug_inference:
            print("  expanded param shape:", out.shape)
            print("[_expand_param] EXIT")

        return out

    # ====================================================
    # --- Normalization / Denormalization
    # ====================================================

    def build_norm_denorm(self, norm_stats, norm_type: str, ref_tensor: torch.Tensor,
                          comp_idx: torch.Tensor | None, depth_idx: torch.Tensor | None):
        """
        Returns two lambdas: (norm_fn, denorm_fn)
        using comp_idx for discrete normalization and
        depth_idx for embedding consistency.
        """

        if self._debug_inference:
            print("\n[build_norm_denorm] ENTER")
            print("  norm_type:", norm_type)
            print("  ref_tensor.shape:", ref_tensor.shape)
            print("  comp_idx:", "None" if comp_idx is None else comp_idx.shape)
            print("  depth_idx:", "None" if depth_idx is None else depth_idx.shape)

        if norm_stats is None:
            if self._debug_inference:
                print("  norm_stats is None -> identity norm/denorm")
                print("[build_norm_denorm] EXIT\n")

            return lambda z: z, lambda z: z

        full_comp = self._get_full_component_vector(ref_tensor.device, torch.long)
        comp_pos = None
        if comp_idx is not None:
            if self._debug_inference:
                print("  using comp_idx to compute comp_pos")
            comp_pos = self._labels_to_positions(comp_idx, full_comp)
        elif depth_idx is not None:
            if self._debug_inference:
                print("  comp_idx is None, using depth_idx fallback for comp_pos")
            # NOTE: this is the expensive path if depth_idx is large
            scaled_depth = depth_idx * full_comp.numel()
            if self._debug_inference:
                print("  scaled_depth.shape:", scaled_depth.shape)
            comp_pos = self._labels_to_positions(scaled_depth, full_comp)

        if self._debug_inference:
            if comp_pos is None:
                print("  comp_pos is None")
            else:
                print("  comp_pos.shape:", comp_pos.shape)

        # --- z-score case
        if norm_type == "z_score":
            mean, std = norm_stats
            mean_t = torch.as_tensor(mean, device=ref_tensor.device, dtype=ref_tensor.dtype)
            std_t = torch.as_tensor(std, device=ref_tensor.device, dtype=ref_tensor.dtype)

            if self._debug_inference:
                print("  z_score stats:")
                print("    mean_t.shape:", mean_t.shape)
                print("    std_t.shape:", std_t.shape)

            def norm_fn(z):
                if self._debug_inference:
                    print("\n  [norm_fn] called with z.shape:", z.shape)
                if z.ndim == 5 and comp_pos is not None:
                    if self._debug_inference:
                        print("    z.ndim == 5 and comp_pos is not None -> per-depth norm")
                    m = self._expand_param(mean_t, z, comp_pos)
                    s = self._expand_param(std_t, z, comp_pos)
                    if self._debug_inference:
                        print("    m.shape:", m.shape, "s.shape:", s.shape)
                    return (z - m) / (s + 1e-8)
                m, s = mean_t.mean(), std_t.mean()
                if self._debug_inference:
                    print("    using global mean/std scalars")
                return (z - m) / (s + 1e-8)

            def denorm_fn(z):
                if self._debug_inference:
                    print("\n  [denorm_fn] called with z.shape:", z.shape)
                if z.ndim == 5 and comp_pos is not None:
                    if self._debug_inference:
                        print("    z.ndim == 5 and comp_pos is not None -> per-depth denorm")
                    m = self._expand_param(mean_t, z, comp_pos)
                    s = self._expand_param(std_t, z, comp_pos)
                    if self._debug_inference:
                        print("    m.shape:", m.shape, "s.shape:", s.shape)
                    return z * s + m
                m, s = mean_t.mean(), std_t.mean()
                if self._debug_inference:
                    print("    using global mean/std scalars")
                return z * s + m

            if self._debug_inference:
                print("[build_norm_denorm] EXIT (z_score)\n")
            return norm_fn, denorm_fn

        # --- min-max case
        else:
            vmin, vmax = norm_stats
            vmin_t = torch.as_tensor(vmin, device=ref_tensor.device, dtype=ref_tensor.dtype)
            vmax_t = torch.as_tensor(vmax, device=ref_tensor.device, dtype=ref_tensor.dtype)

            if self._debug_inference:
                print("  min-max stats:")
                print("    vmin_t.shape:", vmin_t.shape)
                print("    vmax_t.shape:", vmax_t.shape)

            def norm_fn(z):
                if self._debug_inference:
                    print("\n  [norm_fn] (min-max) called with z.shape:", z.shape)
                if z.ndim == 5 and comp_pos is not None:
                    if self._debug_inference:
                        print("    z.ndim == 5 and comp_pos is not None -> per-depth min-max")
                    vmin_sel = self._expand_param(vmin_t, z, comp_pos)
                    vmax_sel = self._expand_param(vmax_t, z, comp_pos)
                    if self._debug_inference:
                        print("    vmin_sel.shape:", vmin_sel.shape, "vmax_sel.shape:", vmax_sel.shape)
                    return (z - vmin_sel) / (vmax_sel - vmin_sel + 1e-8)
                vmin_s, vmax_s = vmin_t.mean(), vmax_t.mean()
                if self._debug_inference:
                    print("    using global vmin/vmax scalars")
                return (z - vmin_s) / (vmax_s - vmin_s + 1e-8)

            def denorm_fn(z):
                if self._debug_inference:
                    print("\n  [denorm_fn] (min-max) called with z.shape:", z.shape)
                if z.ndim == 5 and comp_pos is not None:
                    if self._debug_inference:
                        print("    z.ndim == 5 and comp_pos is not None -> per-depth denorm")
                    vmin_sel = self._expand_param(vmin_t, z, comp_pos)
                    vmax_sel = self._expand_param(vmax_t, z, comp_pos)
                    if self._debug_inference:
                        print("    vmin_sel.shape:", vmin_sel.shape, "vmax_sel.shape:", vmax_sel.shape)
                    return z * (vmax_sel - vmin_sel) + vmin_sel
                vmin_s, vmax_s = vmin_t.mean(), vmax_t.mean()
                if self._debug_inference:
                    print("    using global vmin/vmax scalars")
                return z * (vmax_s - vmin_s) + vmin_s

            if self._debug_inference:
                print("[build_norm_denorm] EXIT (min-max)\n")
            return norm_fn, denorm_fn

    # ====================================================
    # --- test step using both indices
    # ====================================================
    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.test_data = []
            if self._debug_inference:
                print("\n======================")
                print("[test_step] batch_idx == 0")
                print("======================")
                print("  batch.input.shape:", batch.input.shape)
                print("  batch.tgt.shape:", batch.tgt.shape)
                if hasattr(batch, "comp_idx"):
                    print("  batch.comp_idx.shape:", None if batch.comp_idx is None else batch.comp_idx.shape)
                else:
                    print("  batch.comp_idx: attr not present")
                if hasattr(batch, "depth_idx"):
                    print("  batch.depth_idx.shape:", None if batch.depth_idx is None else batch.depth_idx.shape)
                else:
                    print("  batch.depth_idx: attr not present")

        # >>> ADD THIS BLOCK <<<
        print("  >>> VALUES <<<")
        if hasattr(batch, "comp_idx") and batch.comp_idx is not None:
            print("  comp_idx values:", batch.comp_idx.detach().cpu().numpy().reshape(-1).tolist())
        else:
            print("  comp_idx values: None")
        if hasattr(batch, "depth_idx") and batch.depth_idx is not None:
            print("  depth_idx values:", batch.depth_idx.detach().cpu().numpy().reshape(-1).tolist())
        else:
            print("  depth_idx values: None")
        print("  >>> END VALUES <<<")
        # Apply observation mask
        batch = batch._replace(input=self._apply_mask(batch.input))
        if self._debug_inference and batch_idx == 0:
            print("[test_step] after _apply_mask:")
            print("  batch.input.shape:", batch.input.shape)

        with torch.no_grad():
            out = self(batch=batch)
            if self._debug_inference and batch_idx == 0:
                print("[test_step] after forward:")
                print("  raw out.shape:", out.shape)

            # Align dimensions
            if out.ndim == 6 and out.shape[-1] == 1:
                if self._debug_inference and batch_idx == 0:
                    print("  out has last dim=1, squeezing")
                out = out.squeeze(-1)

            if out.ndim == 4 and batch.input.ndim == 5:
                if self._debug_inference and batch_idx == 0:
                    print("  out.ndim==4 and input.ndim==5, unsqueezing depth dim")
                out = out.unsqueeze(2)

            if self._debug_inference and batch_idx == 0:
                print("  aligned out.shape:", out.shape)

            stats = self.norm_stats() if callable(self.norm_stats) else self.norm_stats
            comp_idx = getattr(batch, "comp_idx", None)
            depth_idx = getattr(batch, "depth_idx", None)

            if self._debug_inference and batch_idx == 0:
                print("[test_step] norm_stats type:", type(stats))
                if stats is not None:
                    try:
                        print("  norm_stats[0] shape:", np.shape(stats[0]))
                        print("  norm_stats[1] shape:", np.shape(stats[1]))
                    except Exception as e:
                        print("  could not inspect norm_stats shapes:", e)

            _, denorm = self.build_norm_denorm(stats, self.norm_type, batch.input, comp_idx, depth_idx)

            if self._debug_inference and batch_idx == 0:
                print("[test_step] applying denorm to tgt and out")
                print("  batch.tgt.shape:", batch.tgt.shape)
                print("  out.shape:", out.shape)

            tgt_den = denorm(batch.tgt).cpu()
            out_den = denorm(out).cpu()

            if self._debug_inference and batch_idx == 0:
                print("  tgt_den.shape (cpu):", tgt_den.shape)
                print("  out_den.shape (cpu):", out_den.shape)
                print("[test_step] DONE for batch_idx == 0\n")

            self.test_data.append(torch.stack([tgt_den, out_den], dim=1))

    @property
    def test_quantities(self):
        return ["tgt", "out"]


class Lit4dVarNet_LazyTransfert(transfert.Lit4dVarNet_Fasc):
    """
    Handles both full-depth (B,T,D,H,W) and per-depth (B,T,H,W) patches.
    Compatible with datamodules providing:
      • comp_idx  – discrete integer component index (for normalization)
      • depth_idx – normalized continuous depth coordinate (for embeddings)
    """

    # ====================================================
    # --- Utilities for discrete/continuous component indexing
    # ====================================================

    def _get_full_component_vector(self, device, dtype=torch.long):
        """Cache the full component coordinate vector from datamodule."""
        if not hasattr(self, "_full_comp_tensor") or self._full_comp_tensor is None:
            dm = self.trainer.datamodule
            xr_obj = getattr(dm, "input_da", None)
            if xr_obj is None:
                raise RuntimeError("Datamodule has no input_da with 'component' coord")

            if "component" in xr_obj.coords:
                comp_vals = xr_obj["component"].values
            else:
                first_var = next(iter(xr_obj.data_vars))
                comp_vals = xr_obj[first_var]["component"].values

            self._full_comp_tensor = torch.as_tensor(comp_vals, dtype=torch.long, device="cpu")

        return self._full_comp_tensor.to(device=device, dtype=dtype)

    def _labels_to_positions(self, labels: torch.Tensor, full_comp: torch.Tensor):
        """
        Map (possibly continuous) depth labels to discrete integer positions
        within the full component vector.
        """
        flat_labels = labels.reshape(-1).float()
        full_sorted, sort_idx = torch.sort(full_comp.float())
        pos_sorted = torch.bucketize(flat_labels, full_sorted).clamp_(0, full_sorted.numel() - 1)
        inv_sort = torch.empty_like(sort_idx)
        inv_sort[sort_idx] = torch.arange(sort_idx.numel(), device=sort_idx.device)
        return inv_sort[pos_sorted].view_as(labels)
    
    def _expand_param(self, param_1d, ref_tensor: torch.Tensor, comp_pos: torch.Tensor | None):
        """
        Expand 1-D parameter (mean/std/etc.) to match ref_tensor shape along component dimension.
        
        Args:
            param_1d: 1D array/tensor of per-component parameters (length = num_components)
            ref_tensor: Reference tensor, either (B,T,H,W) or (B,T,C,H,W)
            comp_pos: Component position indices, shape (B,C) with integer indices
        
        Returns:
            Expanded parameter tensor matching ref_tensor dimensions
        """
        if ref_tensor.ndim == 4:
            # (B,T,H,W) - use global mean
            return torch.as_tensor(param_1d, device=ref_tensor.device, dtype=ref_tensor.dtype).mean()

        t = torch.as_tensor(param_1d, device=ref_tensor.device, dtype=ref_tensor.dtype)
        if comp_pos is None:
            return t

        B, T, C, *_ = ref_tensor.shape

        # Ensure comp_pos is [B, C]
        if comp_pos.ndim == 1:
            if comp_pos.numel() == B:
                # One index per batch, replicate across C
                comp_pos = comp_pos.view(B, 1).expand(B, C)
            elif comp_pos.numel() == C:
                # One index per component, replicate across batch
                comp_pos = comp_pos.view(1, C).expand(B, C)
        elif comp_pos.ndim == 2:
            if comp_pos.shape == (B, 1) and C > 1:
                # [B, 1] -> [B, C]
                comp_pos = comp_pos.expand(B, C)
            elif comp_pos.shape == (1, C) and B > 1:
                # [1, C] -> [B, C]
                comp_pos = comp_pos.expand(B, C)
            elif comp_pos.shape != (B, C):
                raise ValueError(f"comp_pos shape {comp_pos.shape} incompatible with (B={B}, C={C})")

        # Gather parameters for each (batch, component) pair
        flat = comp_pos.reshape(-1).clamp(0, len(t) - 1)
        gathered = t[flat]
        tail_ones = [1] * (ref_tensor.ndim - 3)
        return gathered.view(B, 1, C, *tail_ones)

    def build_norm_denorm(self, norm_stats, norm_type: str, ref_tensor: torch.Tensor,
                          comp_idx: torch.Tensor | None, depth_idx: torch.Tensor | None):
        """
        Build normalization and denormalization functions.

        Args:
            norm_stats: Tuple of (means, stds) for z_score or (vmin, vmax) for minmax
            norm_type: Either "z_score" or "minmax"
            ref_tensor: Reference tensor to determine shapes and device
            comp_idx: Discrete component indices for per-level normalization (torch.long) - shape (B, C)
            depth_idx: Continuous normalized depth coordinate for embeddings (unused here)

        Returns:
            Tuple of (norm_fn, denorm_fn) - both are callable
        """
        if norm_stats is None:
            return lambda z: z, lambda z: z

        # comp_idx from dataloader is already integer positions, no need to convert
        comp_pos = comp_idx
        # NOTE: depth_idx is NOT used for normalization - it serves as a model input feature

        # ====================================================================
        # Z-SCORE NORMALIZATION
        # ====================================================================
        if norm_type == "z_score":
            mean, std = norm_stats
            mean_t = torch.as_tensor(mean, device=ref_tensor.device, dtype=ref_tensor.dtype)
            std_t = torch.as_tensor(std, device=ref_tensor.device, dtype=ref_tensor.dtype)

            def norm_fn(z):
                """Normalize data: (z - mean) / std"""
                if z.ndim == 5 and comp_pos is not None:
                    # Per-component normalization for 5D tensors (B, T, D, H, W)
                    m = self._expand_param(mean_t, z, comp_pos)
                    s = self._expand_param(std_t, z, comp_pos)
                    return (z - m) / (s + 1e-8)
                elif z.ndim == 4:
                    # 4D tensors (B, T, H, W) - use global statistics
                    m = mean_t.mean().view(1, 1, 1, 1)
                    s = std_t.mean().view(1, 1, 1, 1)
                    return (z - m) / (s + 1e-8)
                else:
                    # Fallback: global statistics
                    m, s = mean_t.mean(), std_t.mean()
                    return (z - m) / (s + 1e-8)

            def denorm_fn(z):
                """Denormalize data: z * std + mean"""
                if z.ndim == 5 and comp_pos is not None:
                    # Per-component denormalization for 5D tensors
                    m = self._expand_param(mean_t, z, comp_pos)
                    s = self._expand_param(std_t, z, comp_pos)
                    return z * s + m
                elif z.ndim == 4:
                    # 4D tensors - use global statistics
                    m = mean_t.mean().view(1, 1, 1, 1)
                    s = std_t.mean().view(1, 1, 1, 1)
                    return z * s + m
                else:
                    # Fallback: global statistics
                    m, s = mean_t.mean(), std_t.mean()
                    return z * s + m

            return norm_fn, denorm_fn

        # ====================================================================
        # MIN-MAX NORMALIZATION
        # ====================================================================
        elif norm_type == "minmax" or norm_type == "min_max":
            vmin, vmax = norm_stats
            vmin_t = torch.as_tensor(vmin, device=ref_tensor.device, dtype=ref_tensor.dtype)
            vmax_t = torch.as_tensor(vmax, device=ref_tensor.device, dtype=ref_tensor.dtype)

            def norm_fn(z):
                """Normalize data: (z - min) / (max - min)"""
                if z.ndim == 5 and comp_pos is not None:
                    # Per-component normalization for 5D tensors
                    vmin_sel = self._expand_param(vmin_t, z, comp_pos)
                    vmax_sel = self._expand_param(vmax_t, z, comp_pos)
                    return (z - vmin_sel) / (vmax_sel - vmin_sel + 1e-8)
                elif z.ndim == 4:
                    # 4D tensors - use global statistics
                    vmin_s = vmin_t.mean().view(1, 1, 1, 1)
                    vmax_s = vmax_t.mean().view(1, 1, 1, 1)
                    return (z - vmin_s) / (vmax_s - vmin_s + 1e-8)
                else:
                    # Fallback: global statistics
                    vmin_s, vmax_s = vmin_t.mean(), vmax_t.mean()
                    return (z - vmin_s) / (vmax_s - vmin_s + 1e-8)

            def denorm_fn(z):
                """Denormalize data: z * (max - min) + min"""
                if z.ndim == 5 and comp_pos is not None:
                    # Per-component denormalization for 5D tensors
                    vmin_sel = self._expand_param(vmin_t, z, comp_pos)
                    vmax_sel = self._expand_param(vmax_t, z, comp_pos)
                    return z * (vmax_sel - vmin_sel) + vmin_sel
                elif z.ndim == 4:
                    # 4D tensors - use global statistics
                    vmin_s = vmin_t.mean().view(1, 1, 1, 1)
                    vmax_s = vmax_t.mean().view(1, 1, 1, 1)
                    return z * (vmax_s - vmin_s) + vmin_s
                else:
                    # Fallback: global statistics
                    vmin_s, vmax_s = vmin_t.mean(), vmax_t.mean()
                    return z * (vmax_s - vmin_s) + vmin_s

            return norm_fn, denorm_fn

        # ====================================================================
        # UNSUPPORTED NORMALIZATION TYPE
        # ====================================================================
        else:
            raise ValueError(
                f"Unsupported norm_type: '{norm_type}'. "
                f"Expected 'z_score' or 'minmax'/'min_max'"
            )

    # ====================================================
    # --- Override base_step to compute MSE on denormalized values (in cm/s)
    # ====================================================
    def base_step(self, batch, phase=""):
        """
        Compute MSE on denormalized values for accurate error metrics in cm/s.
        This ensures val_mse reflects the actual error in physical units.
        """
        out = self(batch=batch)
        
        # Compute normalized loss for backpropagation
        loss = self.weighted_mse_transfert(out - batch.tgt, self.rec_weight)

        # with torch.no_grad():
            # # Fast path: scale normalized MSE by std² to get physical units MSE
            # # For z-score: MSE_physical = std² × MSE_normalized
            # # Then convert m/s to cm/s by multiplying by 100²
            # stats = self.norm_stats() if callable(self.norm_stats) else self.norm_stats
            # _, std = stats
            # std_t = torch.as_tensor(std, device=out.device, dtype=out.dtype)
            
            # # Get appropriate std for the batch
            # comp_idx = getattr(batch, "comp_idx", None)
            # std_sel = std_t[comp_idx.flatten()].view_as(comp_idx).mean()

            # scale_factor = (std_sel * 100) ** 2
            # denorm_mse = loss * scale_factor
            
            # Log metrics
        # self.log(f"{phase}_mse", denorm_mse, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{phase}_mse", 10000 * loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{phase}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss, out

    # ====================================================
    # --- test step using both indices
    # ====================================================
    def test_step(self, batch, batch_idx):
        from pathlib import Path
        import numpy as np
        
        if batch_idx == 0:
            # Use logger directory for batch storage (more reliable than /tmp)
            if self.logger and hasattr(self.logger, 'log_dir'):
                log_dir = Path(self.logger.log_dir)
            else:
                log_dir = Path.cwd() / 'outputs' / 'test_batches'
            
            self._test_temp_dir = log_dir / 'batches_temp'
            self._test_temp_dir.mkdir(parents=True, exist_ok=True)
            self._test_batch_count = 0
            
            # Cache normalization stats (not the function itself, as comp_idx varies per batch)
            self._cached_norm_stats = self.norm_stats() if callable(self.norm_stats) else self.norm_stats
            
            # Reduce solver steps for faster inference (1/3 of training steps)
            if hasattr(self, 'solver') and hasattr(self.solver, 'n_step') and self.solver.n_step > 1:
                self._original_n_step = self.solver.n_step
                self.solver.n_step = max(1, self.solver.n_step // 3)
    
        # Apply observation mask directly: keep data where mask=1, set NaN where mask=0
        masked_input = batch.input.clone()
        if hasattr(batch, 'mask') and batch.mask is not None:
            # batch.mask is expected to be 1 where data is valid, 0 where it should be masked
            mask = batch.mask.to(batch.input.device)
            masked_input = torch.where(mask == 1, batch.input, torch.tensor(float('nan'), device=batch.input.device))
        
        # Print NaN percentage for monitoring
        if batch_idx % 50 == 0:  # Print every 50 batches
            total_elements = masked_input.numel()
            nan_count = torch.isnan(masked_input).sum().item()
            nan_percentage = (nan_count / total_elements) * 100
            print(f"Batch {batch_idx}: {nan_percentage:.2f}% NaN in masked input ({nan_count}/{total_elements})")
        
        batch = batch._replace(input=masked_input)
        
        with torch.inference_mode():  # More efficient than no_grad for inference
            out = self(batch=batch)
            
            # Align dimensions
            if out.ndim == 6 and out.shape[-1] == 1:
                out = out.squeeze(-1)
            if out.ndim == 4 and batch.input.ndim == 5:
                # Reshape from [B, T*C, H, W] to [B, T, C, H, W]
                B, TC, H, W = out.shape
                _, T, C, _, _ = batch.input.shape
                out = out.view(B, T, C, H, W)
    
            # Build denorm function per batch (comp_idx varies)
            comp_idx = getattr(batch, "comp_idx", None)
            depth_idx = getattr(batch, "depth_idx", None)
            _, denorm = self.build_norm_denorm(self._cached_norm_stats, self.norm_type, batch.input, comp_idx, depth_idx)
            
            # Denormalize and move to CPU
            tgt_den = denorm(batch.tgt).detach().cpu().numpy()
            out_den = denorm(out).detach().cpu().numpy()
            
            # Save to disk as compressed numpy (much more efficient than torch.save)
            batch_file = self._test_temp_dir / f"batch_{batch_idx:06d}.npz"
            np.savez_compressed(batch_file, tgt=tgt_den.astype(np.float16), out=out_den.astype(np.float16))
            self._test_batch_count += 1
            
            # Delete from CPU memory immediately
            del tgt_den, out_den
            
            # Clear CUDA cache every 50 batches
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
    
    def on_test_epoch_end(self):
        """Load batches from disk, reconstruct test data, and compute metrics"""
        from pathlib import Path
        import shutil
        import numpy as np
        import pandas as pd
        
        if not hasattr(self, '_test_temp_dir'):
            return
        
        print(f"\nLoading {self._test_batch_count} batches from disk...")
        self.test_data = []
        
        # Load batches in order
        for i in range(self._test_batch_count):
            batch_file = self._test_temp_dir / f"batch_{i:06d}.npz"
            if batch_file.exists():
                data = np.load(batch_file)
                tgt = torch.from_numpy(data['tgt']).float()
                out = torch.from_numpy(data['out']).float()
                self.test_data.append(torch.stack([tgt, out], dim=1))
                batch_file.unlink()  # Delete file after loading
        
        print(f"Loaded {len(self.test_data)} batches successfully.")
        
        # Reconstruct test data from batches
        rec_da = self.trainer.test_dataloaders.dataset.reconstruct(
            self.test_data, self.rec_weight.cpu().numpy()
        )

        if isinstance(rec_da, list):
            rec_da = rec_da[0]

        self.test_data = rec_da.assign_coords(
            dict(v0=self.test_quantities)
        ).to_dataset(dim='v0')

        # Compute metrics (use test_metrics if available, otherwise metrics)
        metrics_dict = getattr(self, 'test_metrics', None) or getattr(self, 'metrics', {})
        if metrics_dict:
            metric_data = self.test_data.pipe(self.pre_metric_fn)
            metrics = pd.Series({
                metric_n: metric_fn(metric_data) 
                for metric_n, metric_fn in metrics_dict.items()
            })

            print("\n" + "="*60)
            print(metrics.to_frame(name="Metrics").to_markdown())
            print("="*60)
        else:
            print("\nNo metrics defined for evaluation.")
            metrics = pd.Series()
        
        # Save results
        if self.logger:
            save_path = Path(self.logger.log_dir) / 'test_data.nc'
            self.test_data.to_netcdf(save_path)
            print(f"\nTest data saved to: {save_path}")
            if len(metrics) > 0:
                self.logger.log_metrics(metrics.to_dict())
            
            # Save metrics to CSV
            metrics_csv = Path(self.logger.log_dir) / 'test_metrics.csv'
            metrics.to_frame(name="value").to_csv(metrics_csv)
            print(f"Test metrics saved to: {metrics_csv}")
        
        # Clean up temp directory
        shutil.rmtree(self._test_temp_dir, ignore_errors=True)
    

    @property
    def test_quantities(self):
        return ["tgt", "out"]  # Updated to match reduced data storage


class Lit4dVarNet_LazyTransfert_Fast(Lit4dVarNet_LazyTransfert):
    """
    Optimized version of Lit4dVarNet_LazyTransfert with ultra-fast base_step.
    
    Key optimizations:
    - Caches normalization statistics once per epoch
    - Uses mathematical shortcut for z-score denormalization (no tensor operations)
    - Extracts per-component std using comp_idx for accurate physical units
    """
    
    def on_train_epoch_start(self):
        """Cache normalization stats at epoch start"""
        super().on_train_epoch_start() if hasattr(super(), 'on_train_epoch_start') else None
        self._cached_stats = None
    
    def on_validation_epoch_start(self):
        """Pre-compute stats for validation"""
        super().on_validation_epoch_start() if hasattr(super(), 'on_validation_epoch_start') else None
        
        # Cache stats once for the entire validation epoch
        stats = self.norm_stats() if callable(self.norm_stats) else self.norm_stats
        self._cached_stats = stats
    
    def base_step(self, batch, phase=""):
        """
        Ultra-fast base_step with minimal operations.
        
        For z-score normalization:
        - Uses mathematical property: MSE_physical = std² × MSE_normalized
        - Extracts std for batch components using comp_idx
        - No tensor denormalization needed
        """
        out = self(batch=batch)
        
        # Compute normalized loss for backpropagation
        loss = self.weighted_mse_transfert(out - batch.tgt, self.rec_weight)
        
        # Fast logging path using comp_idx
        if phase == "val" and self.norm_type == "z_score" and hasattr(batch, 'comp_idx') and batch.comp_idx is not None:
            # Extract std for the specific components in this batch
            _, std = self._cached_stats
            std_t = torch.as_tensor(std, device=out.device, dtype=out.dtype)
            
            # Get std values for batch components and compute mean
            comp_idx = batch.comp_idx.flatten()
            std_sel = std_t[comp_idx].mean()
            
            # Scale: normalized MSE -> physical MSE in (cm/s)²
            scale_factor = (std_sel * 100) ** 2
            denorm_mse = loss * scale_factor
            self.log(f"{phase}_mse", denorm_mse, prog_bar=True, on_step=False, on_epoch=True)
        else:
            # Training or fallback: use simple scaling
            self.log(f"{phase}_mse", 10000 * loss, prog_bar=True, on_step=False, on_epoch=True)
        
        self.log(f"{phase}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss, out
    
class GradSolver_Fasc_withStep(transfert.GradSolver_Fasc):
    def __init__(self, prior_cost, obs_cost, grad_mod, n_step, lr_grad=0.2, lbd=1.0, **kwargs):
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
        super().__init__(prior_cost, obs_cost, grad_mod, n_step=n_step, lr_grad=lr_grad,**kwargs)
        self.lbd = lbd
        self._grad_norm = None

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
        analysed, _ = transfert.optimal_interpolation(sample, **oi_kw)
        return analysed
    
    def solver_step(self, state, batch, step):
        prior = self.prior_cost(state)
        obs = self.obs_cost(state, batch)
        var_cost = prior + self.lbd**2 * obs

        grad, = torch.autograd.grad(var_cost, state, create_graph=True, allow_unused=False)

        t = torch.tensor([step], device=grad.device).repeat(grad.shape[0])
        gmod = self.grad_mod(grad, t)

        state_update = (1.0 / (step + 1)) * gmod + self.lr_grad * (step + 1) / self.n_step * grad
        new_state = state - state_update

        return new_state
    
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
            self.grad_mod.reset_state(batch.input)

            for step in range(self.n_step):
                state = self.solver_step(state, batch, step=step)
                if not self.training:
                    state = state.detach().requires_grad_(True)

            #if not self.training:
            #    state = self.prior_cost.forward_ae(state)
        return state

class GradSolver_Fasc_withStep_z(transfert.GradSolver_Fasc):
    def __init__(self, prior_cost, obs_cost, grad_mod, n_step, lr_grad=0.2, lbd=1.0, **kwargs):
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
        super().__init__(prior_cost, obs_cost, grad_mod, n_step=n_step, lr_grad=lr_grad,**kwargs)
        self.lbd = lbd
        self._grad_norm = None

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
        analysed, _ = transfert.optimal_interpolation(sample, **oi_kw)
        return analysed
    
    def solver_step(self, state, batch, step):

        extra = {'time': t}
        if hasattr(batch, "depth_idx"):
            extra["depth_idx"] = batch.depth_idx.to(state.device)
        elif isinstance(batch, dict) and "depth_idx" in batch:
            extra["depth_idx"] = batch["depth_idx"].to(state.device)
        prior = self.prior_cost.forward_ae(state, extra=extra)

        obs = self.obs_cost(state, batch)
        var_cost = prior + self.lbd**2 * obs

        grad, = torch.autograd.grad(var_cost, state, create_graph=True, allow_unused=False)

        t = torch.tensor([step], device=grad.device).repeat(grad.shape[0])
        gmod = self.grad_mod(grad, t)

        state_update = (1.0 / (step + 1)) * gmod + self.lr_grad * (step + 1) / self.n_step * grad
        new_state = state - state_update

        return new_state
    
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
            self.grad_mod.reset_state(batch.input)

            for step in range(self.n_step):
                state = self.solver_step(state, batch, step=step)
                if not self.training:
                    state = state.detach().requires_grad_(True)

            #if not self.training:
            #    state = self.prior_cost.forward_ae(state)
        return state
    
class BilinAEPriorCostTwoScale(torch.nn.Module):
    """
    A prior cost model using bilinear autoencoders.

    Attributes:
        bilin_quad (bool): Whether to use bilinear quadratic terms.
        conv_in (nn.Conv2d): Convolutional layer for input.
        conv_hidden (nn.Conv2d): Convolutional layer for hidden states.
        bilin_1 (nn.Conv2d): Bilinear layer 1.
        bilin_21 (nn.Conv2d): Bilinear layer 2 (part 1).
        bilin_22 (nn.Conv2d): Bilinear layer 2 (part 2).
        conv_out (nn.Conv2d): Convolutional layer for output.
        down (nn.Module): Downsampling layer.
        up (nn.Module): Upsampling layer.
    """

    def __init__(self, dim_in, dim_hidden, kernel_size=3, downsamp=None, bilin_quad=True, bias=True):
        """
        Initialize the BilinAEPriorCost module.

        Args:
            dim_in (int): Number of input dimensions.
            dim_hidden (int): Number of hidden dimensions.
            kernel_size (int, optional): Kernel size for convolutions. Defaults to 3.
            downsamp (int, optional): Downsampling factor. Defaults to None.
            bilin_quad (bool, optional): Whether to use bilinear quadratic terms. Defaults to True.
        """
        super().__init__()
        self.bilin_quad = bilin_quad
        self.conv_in = torch.nn.Conv2d(
            dim_in, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2,bias=bias
        )
        self.conv_hidden = torch.nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2,bias=bias
        )

        self.bilin_1 = torch.nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2,bias=bias
        )
        self.bilin_21 = torch.nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2,bias=bias
        )
        self.bilin_22 = torch.nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2,bias=bias
        )

        self.conv_out = torch.nn.Conv2d(
            2 * dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size // 2,bias=bias
        )


        self.conv_in_lr = torch.nn.Conv2d(
            dim_in, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2,bias=bias
        )
        self.conv_hidden_lr = torch.nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2,bias=bias
        )

        self.bilin_1_lr = torch.nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2,bias=bias
        )
        self.bilin_21_lr = torch.nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2,bias=bias
        )
        self.bilin_22_lr = torch.nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2,bias=bias
        )

        self.conv_out_lr = torch.nn.Conv2d(
            2 * dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size // 2,bias=bias
        )


        self.down = torch.nn.AvgPool2d(downsamp) if downsamp is not None else torch.nn.Identity()
        self.up = (
            torch.nn.UpsamplingBilinear2d(scale_factor=downsamp)
            if downsamp is not None
            else torch.nn.Identity()
        )

    def forward_ae(self, x):
        # keep everything finite
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        # (rest unchanged)
        x_ = self.down(x)
        x_ = self.conv_in_lr(x_)
        x_ = self.conv_hidden_lr(F.relu(x_))
        nonlin = self.bilin_21_lr(x_)**2 if self.bilin_quad else (self.bilin_21_lr(x_) * self.bilin_22_lr(x_))
        x_ = self.conv_out_lr(torch.cat([self.bilin_1_lr(x_), nonlin], dim=1))
        dx = self.up(x_)
        x  = self.conv_in(x)
        x  = self.conv_hidden(F.relu(x))
        nonlin = self.bilin_21(x)**2 if self.bilin_quad else (self.bilin_21(x) * self.bilin_22(x))
        x  = self.conv_out(torch.cat([self.bilin_1(x), nonlin], dim=1))
        out = x + dx
        return torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)

    def forward(self, state):
        state = torch.nan_to_num(state, nan=0.0, posinf=1e6, neginf=-1e6)
        ae = self.forward_ae(state)
        diff = state - ae
        diff = torch.nan_to_num(diff, nan=0.0, posinf=1e6, neginf=-1e6)
        return F.mse_loss(diff, torch.zeros_like(diff))  # equivalent to mse(state, ae) but numerically robust

        
class Dense(torch.nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = torch.nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)[..., None, None]

class ResBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, embed_dim, dropout=0.0, bias=False):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=bias)
        self.gn1 = torch.nn.GroupNorm(max(1, out_ch // 8), out_ch)
        self.conv2 = torch.nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=bias)
        self.gn2 = torch.nn.GroupNorm(max(1, out_ch // 8), out_ch)

        self.act = lambda x: x * torch.sigmoid(x)  # Swish
        self.dense = Dense(embed_dim, out_ch)      # time embedding projection
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()

        # skip 1x1 conv si dimensions changent
        self.skip = torch.nn.Conv2d(in_ch, out_ch, 1, bias=bias) if in_ch != out_ch else torch.nn.Identity()

    def forward(self, x, embed):
        h = self.conv1(x)
        h = self.gn1(h)
        h = self.act(h + self.dense(embed))
        h = self.dropout(h)

        h = self.conv2(h)
        h = self.gn2(h)
        h = self.act(h + self.dense(embed))

        return h + self.skip(x)
    
class GaussianFourierProjection(torch.nn.Module):
  """Gaussian random features for encoding time steps."""  
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    self.W = torch.nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    ret = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    return ret    

class UnetGradModelUnet(torch.nn.Module):
    """Score-based UNet avec ResNet blocks et time embedding."""

    def __init__(self, dim_in, dim_hidden, embed_dim, num_levels, unet, out_activation=None, bias=False, dropout=0.0):
        super().__init__()

        # progression des canaux
        channels = [dim_hidden]
        for i in range(num_levels - 1):
            channels.append(channels[-1] * 2)

        # time embedding
        self.embed = torch.nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            torch.nn.Linear(embed_dim, embed_dim)
        )
        self.act = lambda x: x * torch.sigmoid(x)
        self.norm = torch.nn.Parameter(torch.tensor([1.]))

        # --- Encoding ---
        self.enc_blocks = torch.nn.ModuleList()
        in_ch = dim_in
        for ch in channels:
            self.enc_blocks.append(ResBlock(in_ch, ch, embed_dim, bias=bias, dropout=dropout))
            in_ch = ch

        # --- Bottleneck ---
        self.bottleneck = ResBlock(channels[-1], channels[-1], embed_dim, dropout=dropout)

        # --- Decoding ---
        self.dec_blocks = torch.nn.ModuleList()
        for i in reversed(range(1, len(channels))):
            self.dec_blocks.append(
                torch.nn.ModuleDict({
                    "upsample": torch.nn.ConvTranspose2d(channels[i], channels[i-1], 4, stride=2, padding=1, bias=bias),
                    "resblock": ResBlock(channels[i-1]*2, channels[i-1], embed_dim, dropout=dropout, bias=bias)
                })
            )

        # --- Final ---
        if unet is not None:
            self.conv_out = unet
            self.use_unet = True
        else:
            self.use_unet = False
            self.conv_out = torch.nn.Conv2d(channels[0]*2, dim_in, 3, padding=1)
        
        # Option activation de sortie
        if out_activation == "tanh":
            self.out_act = torch.nn.Tanh()
        elif out_activation == "sigmoid":
            self.out_act = torch.nn.Sigmoid()
        else:
            self.out_act = torch.nn.Identity()
        # optional learnable tiny gain (starts at 1.0)
        self.norm = nn.Parameter(torch.tensor([1.0]), requires_grad=True)

        # --- init the last conv small if you're not using an external UNet head
        if not self.use_unet:
            nn.init.zeros_(self.conv_out.bias)
            nn.init.normal_(self.conv_out.weight, std=1e-3)
    def reset_state(self, inp):
        self._grad_norm = None

    def forward(self, x, t):
        if self._grad_norm is None:
            self._grad_norm = (x ** 2).mean().sqrt()
        x = x / self._grad_norm

        # time embedding 
        embed = self.act(self.embed(t))

        # --- Encoder ---
        hs = []
        h = x
        for block in self.enc_blocks:
            h = block(h, embed)
            hs.append(h)
            h = torch.nn.functional.avg_pool2d(h, 2) if block != self.enc_blocks[-1] else h  # downsample sauf dernier

        # --- Bottleneck ---
        h = self.bottleneck(h, embed)

        # --- Decoder ---
        skip_connections = hs[::-1]
        for skip, dec in zip(skip_connections[1:], self.dec_blocks):  # on garde aussi skip du niveau le + bas
            h = dec["upsample"](h)
            h = dec["resblock"](torch.cat([h, skip], dim=1), embed)

        # --- Final ---
        if self.use_unet == True:
            out = self.conv_out.predict(torch.cat([h, skip_connections[-1]], dim=1))
        else:
            out = self.conv_out(torch.cat([h, skip_connections[-1]], dim=1))

        return self.out_act(out)
        
class UnetSolver(torch.nn.Module):
    def __init__(self, dim_in, channel_dims, max_depth=None,bias=True):
        super().__init__()

        if max_depth is not None :
            self.max_depth = np.max( max_depth , len(channel_dims) // 3 )
        else: 
            self.max_depth = len(channel_dims) // 3
        
        self.ups = torch.nn.ModuleList()
        self.up_pools = torch.nn.ModuleList()
        self.downs = torch.nn.ModuleList()
        self.down_pools = torch.nn.ModuleList()
        self.residues = list()

        self.bottom_transform = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=channel_dims[self.max_depth * 3 - 1],
                out_channels=channel_dims[self.max_depth * 3],
                padding="same",
                kernel_size=3,
                bias=bias
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=channel_dims[self.max_depth * 3],
                out_channels=channel_dims[self.max_depth * 3],
                padding="same",
                kernel_size=3,
                bias=bias
            ),
            torch.nn.ReLU(),
        )

        self.final_up = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=channel_dims[0],
                out_channels=dim_in,
                padding="same",
                kernel_size=3,
                bias=bias
            )
        )

        self.final_linear = torch.nn.Sequential(torch.nn.Linear(dim_in, dim_in))

        for depth in range(self.max_depth):
            self.ups.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=channel_dims[depth * 3 + 2] * 2,
                        out_channels=channel_dims[depth * 3 + 1],
                        padding="same",
                        kernel_size=3,
                        bias=bias
                    ),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(
                        in_channels=channel_dims[depth * 3 + 1],
                        out_channels=channel_dims[depth * 3],
                        padding="same",
                        kernel_size=3,
                        bias=bias
                    ),
                    torch.nn.ReLU(),
                )
            )
            self.up_pools.append(
                torch.nn.ConvTranspose2d(
                    in_channels=channel_dims[depth * 3 + 3],
                    out_channels=channel_dims[depth * 3 + 2],
                    kernel_size=2,
                    stride=2,
                    bias=bias
                )
            )
            self.downs.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=dim_in
                        if depth == 0
                        else channel_dims[depth * 3 - 1],
                        out_channels=channel_dims[depth * 3],
                        padding="same",
                        kernel_size=3,
                        bias=bias
                    ),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(
                        in_channels=channel_dims[depth * 3],
                        out_channels=channel_dims[depth * 3 + 1],
                        padding="same",
                        kernel_size=3,
                        bias=bias
                    ),
                    torch.nn.ReLU(),
                )
            )
            self.down_pools.append(torch.nn.MaxPool2d(kernel_size=2))

    def unet_step(self, x, depth):
        x, residue = self.down(x, depth)
        self.residues.append(residue)

        if depth == self.max_depth - 1:
            x = self.bottom_transform(x)
        else:
            x = self.unet_step(x, depth + 1)

        return self.up(x, depth)

    def forward(self, batch):
        x = batch.input
        x = x.nan_to_num()
 #       x = self.final_up(self.unet_step(x, depth=0))
 #       x = torch.permute(x, dims=(0, 2, 3, 1))
 #       x = self.final_linear(x)
 #       x = torch.permute(x, dims=(0, 3, 1, 2))
        return self.predict(x)

    def predict(self,x):
        x = self.final_up(self.unet_step(x, depth=0))
        x = torch.permute(x, dims=(0, 2, 3, 1))
        x = self.final_linear(x)
        x = torch.permute(x, dims=(0, 3, 1, 2))
        return x        

    def down(self, x, depth):
        x = self.downs[depth](x)
        return self.down_pools[depth](x), x

    def up(self, x, depth):
        x = self.up_pools[depth](x)
        x = self.concat_residue(x)
        return self.ups[depth](x)

    def concat_residue(self, x):
        if len(self.residues) != 0:
            residue = self.residues.pop(-1)

            _, _, h_x, w_x = x.shape
            _, _, h_r, w_r = residue.shape

            pad_h = h_r - h_x
            pad_w = w_r - w_x

            if pad_h > 0 or pad_w > 0:
                x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect", value=0)

            return torch.concat((x, residue), dim=1)
        else:
            return x
            
class UnetSolver2(UnetSolver):
    def __init__(self, dim_in, channel_dims, max_depth=None,dim_out=None,bias=True):
        super().__init__(dim_in, channel_dims, max_depth)

        if dim_out is None :
            dim_out = dim_in

        self.final_up = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=channel_dims[0],
                out_channels=4*dim_out,
                padding="same",
                kernel_size=3,
                bias=bias
            ) )

        self.final_linear = torch.nn.Sequential(torch.nn.Linear(4*dim_out, dim_out))


class UpsampleWInterpolate(torch.nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, out_channels=None, interp_mode='bilinear',bias=True):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.interp_mode = interp_mode
        if use_conv:
            self.conv  = torch.nn.Conv2d(in_channels=channels,out_channels=out_channels,
                                        padding="same",kernel_size=1,bias=bias)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode=self.interp_mode)
        if self.use_conv:
            x = self.conv(x)
        return x
    
class UnetSolverBilin(UnetSolver2):
    def __init__(self, dim_in, channel_dims, max_depth=None,dim_out=None,interp_mode='bilinear',dropout=0.1,activation_layer=torch.nn.ReLU(),bias=True):
        super().__init__(dim_in, channel_dims, max_depth=max_depth,bias=bias)

        if dim_out is None :
            dim_out = dim_in

        self.up_pools   = torch.nn.ModuleList()
        self.down_pools = torch.nn.ModuleList()
        self.downs = torch.nn.ModuleList()
        self.ups = torch.nn.ModuleList()

        self.interp_mode = interp_mode
        self.dropout = dropout
        
        self.bottom_transform = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=channel_dims[self.max_depth * 3 - 1],
                out_channels=channel_dims[self.max_depth * 3],
                padding="same",
                kernel_size=3,
                bias=bias,
            ),
            activation_layer,
            torch.nn.Dropout(p=dropout),
            torch.nn.Conv2d(
                in_channels=channel_dims[self.max_depth * 3],
                out_channels=channel_dims[self.max_depth * 3],
                padding="same",
                kernel_size=3,
                bias=bias,
            ),
            activation_layer,
        )

        self.final_up = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=channel_dims[0],
                out_channels=dim_in,
                padding="same",
                kernel_size=3,
                bias=bias,
            )
        )

        for depth in range(self.max_depth):
            self.ups.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=channel_dims[depth * 3 + 2] * 2,
                        out_channels=channel_dims[depth * 3 + 1],
                        padding="same",
                        kernel_size=3,
                        bias=bias,
                    ),
                    activation_layer,
                    torch.nn.Dropout(p=dropout),
                    torch.nn.Conv2d(
                        in_channels=channel_dims[depth * 3 + 1],
                        out_channels=channel_dims[depth * 3],
                        padding="same",
                        kernel_size=3,
                        bias=bias,
                    ),
                    activation_layer,
                )
            )
            self.up_pools.append(
                    UpsampleWInterpolate(channels=channel_dims[depth * 3 + 3], use_conv=True, 
                                        out_channels=channel_dims[depth * 3 + 2], interp_mode= self.interp_mode)
            )
            self.downs.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=dim_in
                        if depth == 0
                        else channel_dims[depth * 3 - 1],
                        out_channels=channel_dims[depth * 3],
                        padding="same",
                        kernel_size=3,
                        bias=bias,
                    ),
                    activation_layer,
                    torch.nn.Dropout(p=dropout),
                    torch.nn.Conv2d(
                        in_channels=channel_dims[depth * 3],
                        out_channels=channel_dims[depth * 3 + 1],
                        padding="same",
                        kernel_size=3,
                        bias=bias,
                    ),
                    activation_layer,
                )
            )

            self.down_pools.append(torch.nn.AvgPool2d(kernel_size=2))

        self.final_up = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=channel_dims[0],
                out_channels=4*dim_out,
                padding="same",
                kernel_size=3,
                bias=bias,
            ) )
        self.final_linear = torch.nn.Sequential(torch.nn.Linear(4*dim_out, dim_out,bias=bias))

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

def run_test(trainer, train_dm, test_dm, lit_mod, ckpt=None):
    """
    Fit and test on two distinct domains.
    """
    if trainer.logger is not None:
        print()
        print('Logdir:', trainer.logger.log_dir)
        print()

    #trainer.fit(lit_mod, datamodule=train_dm, ckpt_path=ckpt)
    trainer.test(lit_mod, datamodule=test_dm, ckpt_path=ckpt)

def run_fast_inference(trainer, test_dm, lit_mod, ckpt=None):
    """
    Optimized inference-only runner with minimal overhead.
    """
    import time
    
    print('-*--- Fast Inference Mode ---*-')
    
    # Set model to eval mode and optimize for inference
    lit_mod.eval()
    
    # Disable unnecessary features
    trainer.enable_model_summary = False
    trainer.enable_checkpointing = False
    
    # Use torch.inference_mode for maximum speed
    with torch.inference_mode():
        start_time = time.time()
        trainer.test(lit_mod, datamodule=test_dm, ckpt_path=ckpt)
        inference_time = time.time() - start_time
        
    print(f"Inference completed in {inference_time:.2f} seconds")
    try:
        test_loader = test_dm.test_dataloader()
        print(f"Approximate speed: {len(test_loader) / inference_time:.2f} batches/second")
    except:
        print("Could not calculate batch speed")
    
    return inference_time
