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
from typing import Sequence, Union
from dataclasses import dataclass, field
torch.set_float32_matmul_precision('high')
import itertools 
from torch.utils.data._utils.collate import default_collate

TrainingItem = namedtuple('TrainingItem', ['input', 'tgt', 'comp_idx'])
TrainingItemwithDepth = namedtuple('TrainingItemwithDepth', ['input', 'tgt', 'depth_idx', 'comp_idx'])
TrainingItemwithDepthTime = namedtuple('TrainingItemwithDepthTime', ['input', 'tgt', 'depth_idx', 'time_idx', 'comp_idx'])

class LazyXrDataset_Depth(torch.utils.data.Dataset):
    """
    Lazily iterate over windowed patches of an xarray DataArray or Dataset.

    Returns patches and applies a postprocessing function that yields:
        TrainingItemwithDepth(input, tgt, depth_idx, comp_idx)
    """
    def __init__(
        self,
        ds,
        patch_dims,
        domain_limits=None,
        strides=None,
        postpro_fn=None,
        *,
        var: str | None = None,
        depth_dim: str | None = "component",
        **kwargs,
    ):
        super().__init__()
        self.postpro_fn = postpro_fn
        self.depth_dim = depth_dim
        self.mask = None
        self.return_coords = False

        # --- 1) crop domain if requested
        self.ds = ds.sel(**(domain_limits or {}))

        # --- 2) if Dataset, optionally pick one variable
        if isinstance(self.ds, xr.Dataset):
            if var is not None:
                if var not in self.ds.data_vars:
                    raise KeyError(
                        f"`var='{var}'` not in dataset variables: {list(self.ds.data_vars)}"
                    )
                self.ds = self.ds[var]
            else:
                # default: keep the whole Dataset (will be handled in __getitem__)
                pass

        # --- 3) prepare patch sizes and strides
        self.patch_dims = dict(patch_dims)
        self.strides = dict(strides or {})
        self._sizes = {dim: self.ds.sizes[dim] for dim in self.ds.dims}

        for dim in self.patch_dims:
            if self.patch_dims[dim] in (None, -1):
                self.patch_dims[dim] = self._sizes[dim]
            else:
                self.patch_dims[dim] = int(self.patch_dims[dim])
            self.strides.setdefault(dim, 1)

        # --- 4) compute number of windows per dim
        self.ds_size = {}
        for dim, psize in self.patch_dims.items():
            full = (psize == self._sizes[dim])
            if full:
                self.ds_size[dim] = 1
            else:
                stride = self.strides[dim]
                self.ds_size[dim] = max((self._sizes[dim] - psize) // stride + 1, 0)

        self._scan_dims = tuple(self.ds_size.keys())
        self._scan_counts = tuple(self.ds_size[d] for d in self._scan_dims)
        self._num = int(np.prod(self._scan_counts)) if len(self._scan_counts) else 1

    # ----------------------------------------------------------
    def __len__(self):
        return self._num

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    # ----------------------------------------------------------
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
                stop = start + psize
                sl[dim] = slice(start, stop)
        for dim in self.ds.dims:
            if dim not in sl:
                sl[dim] = slice(None)
        return sl

    # ----------------------------------------------------------
    def get_coords(self):
        """Return xarray Dataset of coordinates for each patch."""
        self.return_coords = True
        coords = []
        try:
            for i in range(len(self)):
                coords.append(self[i])
        finally:
            self.return_coords = False
        return coords

    # ----------------------------------------------------------
    def __getitem__(self, item):
        sl = self._build_slices(item)
        da = self.ds.isel(**sl)
        if isinstance(da, xr.Dataset):
            da = da[next(iter(da.data_vars))]

        if self.return_coords:
            return da.coords.to_dataset()[list(self.patch_dims)]

        da = da.astype(np.float32)
        return self.postpro_fn(da) if self.postpro_fn is not None else da
    # ---------------- reconstruction (same as LazyXrDataset) ----------------
    def reconstruct(self, batches, weight=None):
        items = list(itertools.chain(*batches))
        return self.reconstruct_from_items(items, weight)

    def reconstruct_from_items(self, items, weight=None):
        """Vectorized reconstruction from overlapping patches."""
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

        n_lead = item_np.ndim - len(coord_dims)
        new_dims = [f'v{i}' for i in range(n_lead)]
        dims = new_dims + coord_dims

        # --- Prepare full output arrays ---
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

        # Create numpy arrays directly (faster than xarray accumulation)
        rec_arr = np.zeros([full_shape[dim] for dim in dims], dtype=np.float32)
        count_arr = np.zeros_like(rec_arr)

        # --- Prepare weight ---
        T = int(self.patch_dims.get("time", 1))
        C = int(self.patch_dims.get("component", 1))
        H = int(self.patch_dims.get("lat", 1))
        W = int(self.patch_dims.get("lon", 1))

        if weight is None:
            w_np = np.ones((T, C, H, W), dtype=np.float32)
        elif isinstance(weight, xr.DataArray):
            w_np = weight.values.astype(np.float32)
            if w_np.shape == (T*C, H, W):
                w_np = w_np.reshape(T, C, H, W)
        else:
            w_np = np.asarray(weight, dtype=np.float32)
            if w_np.shape == (T*C, H, W):
                w_np = w_np.reshape(T, C, H, W)

        # --- Vectorized accumulation using advanced indexing ---
        for item, co in zip(items, coords):
            item_np = _to_numpy(item)

            # Build slice indices for this patch
            slices = []
            for dim in dims:
                if dim in coord_dims:
                    coord_vals = co[dim].values
                    if dim == "time":
                        slices.append(np.searchsorted(full_coords[dim].values, coord_vals))
                    elif dim in ["component", "lat", "lon"]:
                        slices.append(np.searchsorted(full_coords[dim].values, coord_vals))
                    else:
                        slices.append(slice(None))
                else:
                    slices.append(slice(None))

            # Convert to numpy indexing
            idx = tuple(slice(s.min(), s.max()+1) if isinstance(s, np.ndarray) else s for s in slices)

            # Broadcast weight
            w_broadcast = w_np.reshape([1]*n_lead + list(w_np.shape))

            # Accumulate (in-place for speed)
            rec_arr[idx] += item_np * w_broadcast
            count_arr[idx] += w_broadcast

        # Final division
        rec_arr = np.divide(rec_arr, count_arr, out=rec_arr, where=(count_arr > 0))

        # Convert back to xarray only once
        result = xr.DataArray(rec_arr, dims=dims, coords=full_coords)
        return result
    # def reconstruct_from_items(self, items, weight=None):
    #     """Rebuild full xarray field from overlapping patches."""
    #     def _to_numpy(a):
    #         if torch.is_tensor(a):
    #             return a.detach().cpu().numpy()
    #         if isinstance(a, xr.DataArray):
    #             return a.values
    #         return np.asarray(a)

    #     coords = self.get_coords()
    #     sample = items[0]
    #     item_np = _to_numpy(sample)
    #     coord_dims = list(coords[0].dims)

    #     # leading dims (v0, v1, ...)
    #     n_lead = item_np.ndim - len(coord_dims)
    #     new_dims = [f'v{i}' for i in range(n_lead)]
    #     dims = new_dims + coord_dims

    #     das = [
    #         xr.DataArray(_to_numpy(it), dims=dims, coords=co.coords)
    #         for it, co in zip(items, coords)
    #     ]

    #     # full output shape & coords
    #     full_shape = {}
    #     for dim in coord_dims:
    #         if dim in self.ds.dims:
    #             full_shape[dim] = self.ds.sizes[dim]
    #         else:
    #             full_shape[dim] = max(co[dim].size for co in coords)
    #     for i, dim in enumerate(new_dims):
    #         full_shape[dim] = item_np.shape[i]
    #     full_coords = {}
    #     for dim in coord_dims:
    #         if dim in self.ds.coords:
    #             full_coords[dim] = self.ds[dim]
    #         else:
    #             full_coords[dim] = np.arange(full_shape[dim])

    #     rec_da = xr.DataArray(
    #         np.zeros([full_shape[dim] for dim in dims], dtype=np.float32),
    #         dims=dims,
    #         coords=full_coords,
    #     )
    #     count_da = xr.zeros_like(rec_da)

    #     # ---------- weight handling ----------
    #     if getattr(self, "depth_dim", None) is not None:
    #         spatial_temporal_keys = [k for k in self.patch_dims if k != self.depth_dim]
    #     else:
    #         spatial_temporal_keys = list(self.patch_dims.keys())

    #     if weight is None:
    #         w_np = np.ones([self.patch_dims[k] for k in spatial_temporal_keys], dtype=np.float32)
    #     elif isinstance(weight, xr.DataArray):
    #         w_np = weight.values.astype(np.float32)
    #     else:
    #         w_np = np.asarray(weight, dtype=np.float32)

    #     # ensure broadcast shape (T,C,H,W)
    #     T = int(self.patch_dims.get("time", 1))
    #     C = int(self.patch_dims.get("component", 1))
    #     H = int(self.patch_dims.get("lat", 1))
    #     W = int(self.patch_dims.get("lon", 1))
    #     if w_np.shape not in [(T*C, H, W), (T, C, H, W)]:
    #         w_np = np.ones((T, C, H, W), dtype=np.float32)
    #     elif w_np.shape == (T*C, H, W):
    #         w_np = w_np.reshape(T, C, H, W)

    #     # ------------- accumulation loop -------------
    #     for da in das:
    #         w_use_np = w_np.reshape([1]*n_lead + list(w_np.shape))
    #         patch_vals = da.data * w_use_np
    #         rec_sel = rec_da.sel(da.coords)
    #         count_sel = count_da.sel(da.coords)

    #         rec_da.loc[da.coords] = xr.DataArray(
    #             rec_sel.data + patch_vals, dims=da.dims, coords=da.coords
    #         )
    #         count_da.loc[da.coords] = xr.DataArray(
    #             count_sel.data + w_use_np, dims=da.dims, coords=da.coords
    #         )

    #     result = xr.where(count_da > 0, rec_da / count_da, rec_da)
    #     return result

# ============================================================
# --- TransfertLazyDataModule_Depth
# ============================================================
class TransfertLazyDataModule_Depth(transfert.TransfertDataModule):
    """
    Depth-aware DataModule for xarray datasets.

    Provides:
        - comp_idx: discrete integer depth index
        - depth_idx: normalized depth coordinate (z-score or min-max)
    Normalizes both fields and depth_idx per depth level.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean_std_domain = kwargs.get("mean_std_domain", "train")
        self.std_c = kwargs.get("std_c", 1.0)

    # ----------------------------------------------------------
    def _select_var_da(self, obj, variable: str):
        """Extract DataArray for the requested variable robustly."""
        if isinstance(obj, xr.Dataset):
            if variable in obj.data_vars:
                return obj[variable]
            arr = obj.to_array()
            if "variable" in arr.coords and variable in arr["variable"].values:
                return arr.sel(variable=variable)
            idx = 1 if variable == "tgt" else 0
            return arr.isel(variable=idx)
        elif isinstance(obj, xr.DataArray):
            if "variable" in obj.dims:
                if "variable" in obj.coords and variable in obj["variable"].values:
                    return obj.sel(variable=variable)
                idx = 1 if variable == "tgt" else 0
                return obj.isel(variable=idx)
            return obj
        else:
            raise TypeError(f"Unsupported xarray object type: {type(obj)}")

    # ----------------------------------------------------------
    def train_mean_std(self, variable="tgt"):
        """
        Compute per-component mean/std for normalization.
        """
        da = (
            self.input_da
            .sel(self.xrds_kw.get("domain_limits", {}))
            .sel(self.domains[self.mean_std_domain])
        )
        da = self._select_var_da(da, variable)
        means = da.mean(dim=("time", "lat", "lon"))
        stds = da.std(dim=("time", "lat", "lon")) * self.std_c
        return means.values, stds.values

    def min_max_norm(self, variable="tgt"):
        """
        Compute per-component min/max normalization.
        """
        da = (
            self.input_da
            .sel(self.xrds_kw.get("domain_limits", {}))
            .sel(self.domains[self.mean_std_domain])
        )
        da = self._select_var_da(da, variable)
        vmin = da.min(dim=("time", "lat", "lon"))
        vmax = da.max(dim=("time", "lat", "lon"))
        return vmin.values, vmax.values

    # ----------------------------------------------------------
    def norm_stats(self):
        """Convenience wrapper for normalization statistics."""
        if self.norm_type == "z_score":
            return self.train_mean_std("tgt")
        elif self.norm_type == "minmax":
            return self.min_max_norm("tgt")
        else:
            raise ValueError(f"Unsupported norm_type: {self.norm_type}")
        
    def post_fn(self):
        means, stds = self.norm_stats()  # Field statistics
        means = np.asarray(means, dtype=np.float32)
        stds = np.asarray(stds, dtype=np.float32)

        # Get full component coordinate
        if isinstance(self.input_da, xr.Dataset):
            full_comp = self.input_da['component'] if 'component' in self.input_da.coords \
                        else self.input_da[next(iter(self.input_da.data_vars))]['component']
        else:
            full_comp = self.input_da['component']

        # DEPTH NORMALIZATION: Use min/max of coordinate values for depth_idx embedding
        depth_min = float(full_comp.min())
        depth_max = float(full_comp.max())

        def _norm_xr(da: xr.DataArray) -> xr.DataArray:
            comp = da['component']
            m_full = xr.DataArray(means, dims=['component'], coords={'component': full_comp})
            s_full = xr.DataArray(stds, dims=['component'], coords={'component': full_comp})
            m = m_full.sel(component=comp)
            s = s_full.sel(component=comp)
            return (da - m) / (s + 1e-8)

        def _make_item(x: xr.DataArray):
            comp = x["component"].values.astype(np.float32)

            # DISCRETE INDEX (comp_idx): Map to integer positions for normalization/denormalization lookup
            full_comp_vals = full_comp.values.astype(np.float32)
            comp_idx = np.searchsorted(full_comp_vals, comp)
            comp_idx = np.clip(comp_idx, 0, len(full_comp_vals) - 1)
            comp_idx_torch = torch.as_tensor(comp_idx, dtype=torch.long)

            # CONTINUOUS DEPTH EMBEDDING (depth_idx): Normalize coordinate to [0, 1] for embedding
            depth_idx_norm = (comp - depth_min) / (depth_max - depth_min + 1e-8)
            depth_idx_norm = torch.as_tensor(depth_idx_norm.mean(), dtype=torch.float32).unsqueeze(0)

            # Normalize fields using per-component statistics
            if "variable" in x.dims:
                inp = _norm_xr(x.isel(variable=0))
                tgt = _norm_xr(x.isel(variable=1))
            else:
                fld = _norm_xr(x)
                inp, tgt = fld, fld

            inp_np = inp.transpose("time", "component", "lat", "lon").data.astype(np.float32)
            tgt_np = tgt.transpose("time", "component", "lat", "lon").data.astype(np.float32)

            inp_t = torch.as_tensor(inp_np).contiguous()
            tgt_t = torch.as_tensor(tgt_np).contiguous()

            return TrainingItemwithDepth(inp_t, tgt_t, depth_idx_norm, comp_idx_torch)

        return _make_item
    # # ----------------------------------------------------------
    # def post_fn(self):
    #     means, stds = self.norm_stats()
    #     means = np.asarray(means, dtype=np.float32)
    #     stds = np.asarray(stds, dtype=np.float32)

    #     # ---- full component coordinate ----
    #     if isinstance(self.input_da, xr.Dataset):
    #         if "component" in self.input_da.coords:
    #             full_comp = self.input_da["component"]
    #         else:
    #             first_var = next(iter(self.input_da.data_vars))
    #             full_comp = self.input_da[first_var]["component"]
    #     else:
    #         full_comp = self.input_da["component"]

        # # ---- normalization helper for the actual data ----
        # def _norm_xr(da: xr.DataArray) -> xr.DataArray:
        #     comp = da["component"]
        #     m_full = xr.DataArray(means, dims=["component"], coords={"component": full_comp})
        #     s_full = xr.DataArray(stds, dims=["component"], coords={"component": full_comp})
        #     m = m_full.sel(component=comp)
        #     s = s_full.sel(component=comp)
        #     return (da - m) / (s + 1e-8)

        # def _make_item(x: xr.DataArray):
        #     comp = x["component"].values.astype(np.float32)

        #     # --- map component coordinate to safe index ---
        #     full_comp_vals = np.asarray(full_comp.values, dtype=np.float32)
        #     comp_idx = np.searchsorted(full_comp_vals, comp)
        #     comp_idx = np.clip(comp_idx, 0, len(full_comp_vals) - 1)
        #     comp_idx_torch = torch.as_tensor(comp_idx, dtype=torch.long)

        #     # --- per-component mean/std normalization for depth ---
        #     comp_mean = torch.as_tensor(means[comp_idx], dtype=torch.float32)
        #     comp_std = torch.as_tensor(stds[comp_idx], dtype=torch.float32)

        #     if self.norm_type == "z_score":
        #         depth_idx_norm = ((comp - comp_mean.numpy()) / (comp_std.numpy() + 1e-8)).mean()
        #     else:
        #         depth_idx_norm = ((comp - full_comp_vals.min()) / (full_comp_vals.max() - full_comp_vals.min() + 1e-8)).mean()

        #     depth_idx_norm = torch.as_tensor(depth_idx_norm, dtype=torch.float32).unsqueeze(0)

        #     # --- normalize input and target fields ---
        #     if "variable" in x.dims:
        #         inp = _norm_xr(x.isel(variable=0))
        #         tgt = _norm_xr(x.isel(variable=1))
        #     else:
        #         fld = _norm_xr(x)
        #         inp, tgt = fld, fld

        #     inp_np = np.asarray(inp.transpose("time", "component", "lat", "lon").values, dtype=np.float32)
        #     tgt_np = np.asarray(tgt.transpose("time", "component", "lat", "lon").values, dtype=np.float32)

        #     inp_t = torch.as_tensor(inp_np).contiguous()
        #     tgt_t = torch.as_tensor(tgt_np).contiguous()

        #     return TrainingItemwithDepth(inp_t, tgt_t, depth_idx_norm, comp_idx_torch)

        # return _make_item
    
    def setup(self, stage="test"):
        post_fn = self.post_fn()
        if stage == "fit":
            train_data = self.input_da.sel(self.domains["train"])
            train_xrds_kw = deepcopy(self.xrds_kw)
            self.train_ds = LazyXrDataset_Depth(
                train_data, **train_xrds_kw, depth_dim="component", postpro_fn=post_fn
            )
            if self.aug_kw:
                self.train_ds = AugmentedDataset(self.train_ds, **self.aug_kw)

            self.val_ds = LazyXrDataset_Depth(
                self.input_da.sel(self.domains["val"]),
                **self.xrds_kw,
                depth_dim="component",
                postpro_fn=post_fn,
            )
        else:
            self.test_ds = LazyXrDataset_Depth(
                self.input_da.sel(self.domains["test"]),
                **self.xrds_kw,
                depth_dim="component",
                postpro_fn=post_fn,
            )

    # ----------------------------------------------------------
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds, shuffle=True, collate_fn=self.collate_fn_namedtuple, **self.dl_kw
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds, shuffle=False, collate_fn=self.collate_fn_namedtuple, **self.dl_kw
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_ds, shuffle=False, collate_fn=self.collate_fn_namedtuple, **self.dl_kw
        )

    # ----------------------------------------------------------
    # @staticmethod
    # def collate_fn_namedtuple(batch):
    #     """Strict collate for TrainingItemwithDepth: forces all fields to tensors."""
    #     elem = batch[0]
    #     if not hasattr(elem, "_fields"):
    #         from torch.utils.data._utils.collate import default_collate
    #         return default_collate(batch)

    #     collated = []
    #     for f in elem._fields:
    #         vals = [getattr(b, f) for b in batch]
    #         tensors = [v if torch.is_tensor(v) else torch.as_tensor(v) for v in vals]

    #         try:
    #             out = torch.stack(tensors)
    #         except Exception as e:
    #             print(f"[WARN collate] Could not stack field {f}: {e}. Shapes: {[t.shape for t in tensors]}")
    #             out = torch.cat([t.unsqueeze(0) for t in tensors], dim=0)

    #         collated.append(out)

    #     return type(elem)(*collated)
    @staticmethod
    def collate_fn_namedtuple(batch):
        """Robust collate that handles dicts/namedtuples with flexible patch shapes, depth_idx, and comp_idx."""
        elem = batch[0]

        def safe_stack(vals):
            if all(isinstance(v, torch.Tensor) for v in vals):
                try:
                    return torch.stack(vals)
                except RuntimeError:
                    shapes = [v.shape for v in vals]
                    if all(len(s) == len(shapes[0]) for s in shapes):
                        return torch.stack(vals)
                    else:
                        vals = [v.unsqueeze(0) if v.ndim == (len(shapes[0]) - 1) else v for v in vals]
                        return torch.stack(vals)
            elif all(isinstance(v, np.ndarray) for v in vals):
                return torch.as_tensor(np.array(vals))
            else:
                return torch.as_tensor(np.array(vals))

        # Dict case
        if isinstance(elem, dict):
            return {k: safe_stack([b[k] for b in batch]) for k in elem}

        # Namedtuple case
        elif hasattr(elem, "_fields"):
            collated = []
            for f in elem._fields:
                vals = [getattr(b, f) for b in batch]
                collated.append(safe_stack(vals))
            return type(elem)(*collated)

        # Default PyTorch collate
        return torch.utils.data._utils.collate.default_collate(batch)

# class LazyXrDataset_Depth(torch.utils.data.Dataset):
#     """
#     Lazily iterate over windowed patches of an xarray DataArray or Dataset.

#     - pass `var="pca_test"` if `ds` is a Dataset (we'll select that DataArray)
#     - 'full-extent' per-dim via None / -1 in patch_dims
#     - keeps xarray.DataArray all the way to postpro_fn (for safe coord-aware ops)
#     """
#     def __init__(
#         self,
#         ds,
#         patch_dims,                 # dict: {dim: size or None/-1 for full}
#         domain_limits=None,
#         strides=None,               # dict: {dim: stride}; default 1
#         postpro_fn=None,
#         *,
#         var: str | None = None,     # <- NEW: select variable if ds is a Dataset
#         depth_dim: str | None = None,  # <- NEW: used by reconstruct_*
#         **kwargs,
#     ):
#         super().__init__()
#         self.return_coords = False
#         self.postpro_fn = postpro_fn
#         self.depth_dim = depth_dim  # may be None; reconstruct_* checks it
#         self.mask = None

#         # 1) crop
#         self.ds = ds.sel(**(domain_limits or {}))

#         # 2) if Dataset, optionally pick one var (strongly recommended)
#         if isinstance(self.ds, xr.Dataset):
#             if var is not None:
#                 if var not in self.ds.data_vars:
#                     raise KeyError(f"`var='{var}'` not in dataset variables: {list(self.ds.data_vars)}")
#                 self.ds = self.ds[var]  # now DataArray
#             else:
#                 # if no var provided, keep Dataset but warn via assertive guard in __getitem__
#                 pass

#         # 3) store dims/strides
#         self.patch_dims = dict(patch_dims)
#         self.strides = dict(strides or {})

#         # 4) sizes and early guards
#         #    If self.ds is a Dataset here, it means user intentionally wants to pass a Dataset
#         #    (e.g., will use mask branch building a stacked 'variable' dim). That also works.
#         self._sizes = {dim: self.ds.sizes[dim] for dim in self.ds.dims}
#         missing = [k for k in self.patch_dims if k not in self._sizes]
#         if missing:
#             raise KeyError(
#                 f"patch_dims keys not found in dataset dims: {missing}. "
#                 f"Available dims: {list(self._sizes.keys())}"
#             )

#         # 5) normalize 'full extent'
#         for dim in list(self.patch_dims.keys()):
#             val = self.patch_dims[dim]
#             if val is None or val == -1:
#                 self.patch_dims[dim] = self._sizes[dim]
#             else:
#                 self.patch_dims[dim] = int(val)

#         # 6) default stride 1
#         for dim in self.patch_dims:
#             self.strides.setdefault(dim, 1)

#         # 7) number of windows per dim
#         self.ds_size = {}
#         for dim, psize in self.patch_dims.items():
#             full = (psize == self._sizes[dim])
#             if full:
#                 self.ds_size[dim] = 1
#             else:
#                 stride = self.strides.get(dim, 1)
#                 if psize > self._sizes[dim]:
#                     raise ValueError(f"patch_dims[{dim}]={psize} exceeds dataset size {self._sizes[dim]}")
#                 self.ds_size[dim] = max((self._sizes[dim] - psize) // stride + 1, 0)

#         # 8) unravel helpers
#         self._scan_dims = tuple(self.ds_size.keys())
#         self._scan_counts = tuple(self.ds_size[d] for d in self._scan_dims)
#         self._num = int(np.prod(self._scan_counts)) if len(self._scan_counts) else 1

#     def __len__(self):
#         return self._num

#     def __iter__(self):
#         for i in range(len(self)):
#             yield self[i]

#     def get_coords(self):
#         """Return the coords (as xarray Datasets) corresponding to each lazy patch."""
#         self.return_coords = True
#         coords = []
#         try:
#             for i in range(len(self)):
#                 coords.append(self[i])
#         finally:
#             self.return_coords = False
#         return coords

#     def _build_slices(self, flat_idx):
#         if len(self._scan_counts) == 0:
#             return {dim: slice(None) for dim in self.ds.dims}
#         multi_idx = np.unravel_index(flat_idx, self._scan_counts)
#         sl = {}
#         for dim, count, idx in zip(self._scan_dims, self._scan_counts, multi_idx):
#             psize = self.patch_dims[dim]
#             stride = self.strides.get(dim, 1)
#             if psize == self._sizes[dim]:
#                 sl[dim] = slice(None)
#             else:
#                 start = stride * int(idx)
#                 stop  = start + psize
#                 sl[dim] = slice(start, stop)
#         for dim in self.ds.dims:
#             if dim not in sl:
#                 sl[dim] = slice(None)
#         return sl

#     def __getitem__(self, item):
#         """
#         Return a single patch (as xarray or tensor) with enforced dimension order (time, component, lat, lon).
#         Compatible with both masked and unmasked datasets.
#         """
#         sl = self._build_slices(item)
    
#         # ---------- CASE 1: masked ----------
#         if self.mask is not None:
#             # Time-wrap mask if needed
#             if "time" in sl and self.mask is not None:
#                 start = sl["time"].start or 0
#                 stop  = sl["time"].stop or self._sizes["time"]
#                 start_mod, stop_mod = start % 365, stop % 365
#                 if start_mod > stop_mod:
#                     start_mod -= stop_mod
#                     stop_mod = None
#                 sl_mask = dict(sl)
#                 sl_mask["time"] = slice(start_mod, stop_mod)
#             else:
#                 sl_mask = sl
    
#             da = self.ds.isel(**sl)
    
#             # Pick main variable
#             if isinstance(da, xr.Dataset):
#                 main_name = "tgt" if "tgt" in da.data_vars else next(iter(da.data_vars))
#                 da_main = da[main_name]
#             else:
#                 da_main = da
    
#             ds_stack = da_main.to_dataset(name="tgt")
#             inp = da_main.where(self.mask.isel(**sl_mask).values)
#             ds_stack = ds_stack.assign(input=inp)
#             item_xr = ds_stack.to_array()  # dims: ('variable', ..., 'time','lat','lon','component'?)
    
#         # ---------- CASE 2: unmasked ----------
#         else:
#             item_xr = self.ds.isel(**sl)
#             if isinstance(item_xr, xr.Dataset):
#                 pick = next(iter(item_xr.data_vars))
#                 item_xr = item_xr[pick]
    
#         # ---------- ENFORCE DIM ORDER ----------
#         # Many ocean xarray datasets store component last → fix that
#         dims = list(item_xr.dims)
#         # Standard expected order
#         expected_order = ["time", "component", "lat", "lon"]
    
#         # Detect if all dims present
#         if set(["time", "lat", "lon"]).issubset(dims):
#             if "component" in dims:
#                 # if component last (e.g. time,lat,lon,component) → transpose
#                 if dims[-1] == "component":
#                     item_xr = item_xr.transpose("time", "component", "lat", "lon")
#                 # if component first or already correct → keep
#                 elif dims != expected_order:
#                     try:
#                         item_xr = item_xr.transpose(*expected_order)
#                     except Exception:
#                         # fallback: ignore missing dims
#                         pass
#             else:
#                 # No component dimension → (time,lat,lon) field
#                 item_xr = item_xr.transpose("time", "lat", "lon")
    
#         # ---------- RETURN COORDS ----------
#         if self.return_coords:
#             return item_xr.coords.to_dataset()[list(self.patch_dims)]
    
#         # ---------- FINAL CONVERSION ----------
#         x = item_xr.astype(np.float32)
#         if set(["time","component","lat","lon"]).issubset(x.dims):
#             x = x.transpose("time","component","lat","lon")
#         assert list(x.dims)[:2] == ["time","component"], f"Bad dim order {x.dims}"

#         # ---------- POSTPROCESSING ----------
#         if self.postpro_fn is not None:
#             return self.postpro_fn(x)
#         else:
#             return x
    

#     # ---------------- reconstruction (unchanged except: self.depth_dim may be None) ----------------
#     def reconstruct(self, batches, weight=None):
#         items = list(itertools.chain(*batches))
#         return self.reconstruct_from_items(items, weight)

#     def reconstruct_from_items(self, items, weight=None):
#         # normalize items to numpy and collect coords as before
#         def _to_numpy(a):
#             if torch.is_tensor(a):
#                 return a.detach().cpu().numpy()
#             if isinstance(a, xr.DataArray):
#                 return a.values
#             return np.asarray(a)
    
#         coords = self.get_coords()
#         sample = items[0]
#         item_np = _to_numpy(sample)
#         coord_dims = list(coords[0].dims)
    
#         # leading dims (v0, v1, ...) if any
#         n_lead = item_np.ndim - len(coord_dims)
#         new_dims = [f'v{i}' for i in range(n_lead)]
#         dims = new_dims + coord_dims
    
#         das = [
#             xr.DataArray(_to_numpy(it), dims=dims, coords=co.coords)
#             for it, co in zip(items, coords)
#         ]
    
#         # full output shape & coords
#         full_shape = {}
#         for dim in coord_dims:
#             if dim in self.ds.dims:
#                 full_shape[dim] = self.ds.sizes[dim]
#             else:
#                 full_shape[dim] = max(co[dim].size for co in coords)
    
#         for i, dim in enumerate(new_dims):
#             full_shape[dim] = item_np.shape[i]
    
#         full_coords = {}
#         for dim in coord_dims:
#             if dim in self.ds.coords:
#                 full_coords[dim] = self.ds[dim]
#             else:
#                 full_coords[dim] = np.arange(full_shape[dim])
    
#         rec_da = xr.DataArray(
#             np.zeros([full_shape[dim] for dim in dims], dtype=np.float32),
#             dims=dims,
#             coords=full_coords,
#         )
#         count_da = xr.zeros_like(rec_da)
    
#         # ---------- robust weight handling ----------
#         # keys we "slide" on (exclude depth_dim if provided)
#         if getattr(self, "depth_dim", None) is not None:
#             spatial_temporal_keys = [k for k in self.patch_dims if k != self.depth_dim]
#         else:
#             spatial_temporal_keys = list(self.patch_dims.keys())
    
#         # Make an xr.DataArray 'w' with whatever dims user supplied; broadcast later.
#         def _make_weight_da(weight):
#             if weight is None:
#                 # default to ones over all non-depth patch dims
#                 shp = [self.patch_dims[k] for k in spatial_temporal_keys]
#                 return xr.DataArray(np.ones(shp, dtype=np.float32), dims=spatial_temporal_keys)
    
#             if isinstance(weight, xr.DataArray):
#                 return weight.astype(np.float32)
    
#             w_np = np.asarray(weight)
#             # Try to assign dims heuristically by matching known sizes
#             known_sizes = {k: self.patch_dims[k] for k in self.patch_dims}
#             # candidates in preferred order
#             pref = ['time', 'lat', 'lon', 'component']
#             # build a dims list that matches w_np.shape
#             dims_guess = []
#             sizes_left = dict(known_sizes)
#             shape = list(w_np.shape)
    
#             # Simple cases first
#             if w_np.ndim == 0:
#                 return xr.DataArray(float(w_np))
#             if w_np.ndim == 1:
#                 # try to match a single known dim by size
#                 for k in pref:
#                     if k in sizes_left and sizes_left[k] == shape[0]:
#                         return xr.DataArray(w_np.astype(np.float32), dims=[k])
#                 # fallback: unnamed 1D (broadcast later)
#                 return xr.DataArray(w_np.astype(np.float32))
    
#             # Multi-dim: greedily match from pref list
#             used = [False] * len(shape)
#             for k in pref:
#                 if k in sizes_left:
#                     for i, s in enumerate(shape):
#                         if not used[i] and s == sizes_left[k]:
#                             dims_guess.append(k)
#                             used[i] = True
#                             break
                        
#             # if we matched all axes, great; else leave unnamed (will still broadcast)
#             if sum(used) == len(shape):
#                 return xr.DataArray(w_np.astype(np.float32), dims=dims_guess)
#             else:
#                 return xr.DataArray(w_np.astype(np.float32))  # no dims; xarray will still try to align/broadcast
    
#         w = _make_weight_da(weight)
#         # --- normalize weight dims to match da dims ---
#         # ---------- strict weight handling for (T*C,H,W) and (T,C,H,W) ----------
#         T = int(self.patch_dims["time"])
#         C = int(self.patch_dims["component"])
#         H = int(self.patch_dims["lat"])
#         W = int(self.patch_dims["lon"])

#         def _weight_to_TCHW(weight):
#             """
#             Accept only (T*C,H,W) or (T,C,H,W), return np.float32 of shape (T,C,H,W).
#             """
#             if isinstance(weight, xr.DataArray):
#                 w_np = weight.values
#             else:
#                 w_np = np.asarray(weight)

#             if w_np.shape == (T * C, H, W):
#                 return w_np.reshape(T, C, H, W).astype(np.float32)

#             if w_np.shape == (T, C, H, W):
#                 return w_np.astype(np.float32)

#             raise ValueError(
#                 f"Unsupported weight shape {w_np.shape}; expected (T*C,H,W) or (T,C,H,W) "
#                 f"with T={T}, C={C}, H={H}, W={W}."
#             )

#         w4 = _weight_to_TCHW(weight)  # np.ndarray, (T,C,H,W)
#         # ------------- accumulation loop -------------
#         for da in das:
#             # Build a broadcast shape for weight that matches da.dims:
#             # - 1 for any leading dims (e.g., 'v0')
#             # - T, C, H, W for the named patch dims.
#             shape_per_dim = []
#             for d in da.dims:
#                 if d == "time":
#                     shape_per_dim.append(T)
#                 elif d == "component":
#                     shape_per_dim.append(C)
#                 elif d == "lat":
#                     shape_per_dim.append(H)
#                 elif d == "lon":
#                     shape_per_dim.append(W)
#                 else:
#                     shape_per_dim.append(1)

#             w_use_np = w4.reshape(shape_per_dim)           # strictly positional broadcast
#             patch_vals = da.data * w_use_np                # NumPy multiply, no xarray alignment

#             rec_sel   = rec_da.sel(da.coords)
#             count_sel = count_da.sel(da.coords)

#             rec_da.loc[da.coords]   = xr.DataArray(rec_sel.data + patch_vals, dims=da.dims, coords=da.coords)
#             count_da.loc[da.coords] = xr.DataArray(count_sel.data + w_use_np,  dims=da.dims, coords=da.coords)

#         result = xr.where(count_da > 0, rec_da / count_da, rec_da)
#         return result

# class TransfertLazyDataModule_Depth(transfert.TransfertDataModule):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.mean_std_domain = kwargs.get('mean_std_domain', 'train')
#         self.std_c = kwargs.get('std_c', 1.)
    
#     def _select_var_da(self, obj, variable: str):
#         """
#         Return a DataArray for the requested 'variable', handling:
#         - Dataset with data_vars -> obj[variable]
#         - DataArray with 'variable' dim -> sel(variable=...)
#         - Plain DataArray (no variable dim) -> return as-is
#         Falls back to index [0=input, 1=tgt] if names are missing.
#         """
#         if isinstance(obj, xr.Dataset):
#             if variable in obj.data_vars:
#                 return obj[variable]
#             # fallback: convert to array (variable, ...)
#             arr = obj.to_array()
#             if 'variable' in arr.coords and (variable in arr['variable'].values):
#                 return arr.sel(variable=variable)
#             idx = 1 if variable == 'tgt' else 0
#             return arr.isel(variable=idx)

#         elif isinstance(obj, xr.DataArray):
#             if 'variable' in obj.dims:
#                 if 'variable' in obj.coords and (variable in obj['variable'].values):
#                     return obj.sel(variable=variable)
#                 idx = 1 if variable == 'tgt' else 0
#                 return obj.isel(variable=idx)
#             return obj

#         else:
#             raise TypeError(f"Unsupported xarray object type: {type(obj)}")

#     def train_mean_std(self, variable='tgt'):
#         da = (
#             self.input_da
#             .sel(self.xrds_kw.get('domain_limits', {}))
#             .sel(self.domains[self.mean_std_domain])
#         )
#         da = self._select_var_da(da, variable)   # <- robust selection
#         means = da.mean(dim=('time', 'lat', 'lon'))            # (component,)
#         stds  = da.std (dim=('time', 'lat', 'lon')) * self.std_c
#         return means.values, stds.values

#     def min_max_norm(self, variable='tgt'):
#         da = (
#             self.input_da
#             .sel(self.xrds_kw.get('domain_limits', {}))
#             .sel(self.domains[self.mean_std_domain])
#         )
#         da = self._select_var_da(da, variable)
#         vmin = da.min(dim=('time', 'lat', 'lon'))
#         vmax = da.max(dim=('time', 'lat', 'lon'))
#         return vmin.values, vmax.values

#     def post_fn(self):
#         means, stds = self.norm_stats()
#         means = np.asarray(means, dtype=np.float32)
#         stds  = np.asarray(stds,  dtype=np.float32)

#         # full "global" component coordinate (vector of all components)
#         if isinstance(self.input_da, xr.Dataset):
#             if 'component' in self.input_da.coords:
#                 full_comp = self.input_da['component']
#             else:
#                 first_var = next(iter(self.input_da.data_vars))
#                 full_comp = self.input_da[first_var]['component']
#         else:
#             full_comp = self.input_da['component']
        
#         def _norm_xr(da: xr.DataArray) -> xr.DataArray:
#             comp = da['component']  # the components present in this patch (often length 1)
#             m_full = xr.DataArray(means, dims=['component'], coords={'component': full_comp})
#             s_full = xr.DataArray(stds,  dims=['component'], coords={'component': full_comp})
#             m = m_full.sel(component=comp)
#             s = s_full.sel(component=comp)
#             return (da - m) / (s + 1e-8)    

#         def _make_item(x: xr.DataArray):
#             comp = x["component"].values  # e.g. array([0]) or array([12])
#             comp_idx_torch = torch.as_tensor(comp, dtype=torch.float32)
#             # --- NEW: normalize using full depth coordinate range ---
#             zmin = float(full_comp.min())
#             zmax = float(full_comp.max())
#             depth_idx_norm = (comp_idx_torch - zmin) / (zmax - zmin + 1e-8)
#             depth_idx_norm = depth_idx_norm.mean().unsqueeze(0)  # shape [1] for batch consistency
#             if "variable" in x.dims:
#                 inp = _norm_xr(x.isel(variable=0))
#                 tgt = _norm_xr(x.isel(variable=1))
#             else:
#                 fld = _norm_xr(x)
#                 inp = fld
#                 tgt = fld

#             inp_np = inp.transpose("time", "component", "lat", "lon").data.astype(np.float32)
#             tgt_np = tgt.transpose("time", "component", "lat", "lon").data.astype(np.float32)

#             # Convert to torch tensors in consistent order
#             inp_t = torch.as_tensor(inp_np)      # (T,C,H,W)
#             tgt_t = torch.as_tensor(tgt_np)      # (T,C,H,W)

#             return TrainingItemwithDepth(inp_t, tgt_t, depth_idx_norm)

#         return _make_item

#     def setup(self, stage='test'):
#         post_fn = self.post_fn()
#         if stage == 'fit':
#             train_data = self.input_da.sel(self.domains['train'])
#             train_xrds_kw = deepcopy(self.xrds_kw)
#             self.train_ds = LazyXrDataset_Depth(
#                 train_data, **train_xrds_kw, depth_dim='component', postpro_fn=post_fn,
#             )
#             if self.aug_kw:
#                 self.train_ds = AugmentedDataset(self.train_ds, **self.aug_kw)

#             self.val_ds = LazyXrDataset_Depth(
#                 self.input_da.sel(self.domains['val']),
#                 **self.xrds_kw,
#                 depth_dim='component',
#                 postpro_fn=post_fn,
#             )
#         else:
#             self.test_ds = LazyXrDataset_Depth(
#                 self.input_da.sel(self.domains['test']),
#                 **self.xrds_kw,
#                 depth_dim='component',
#                 postpro_fn=post_fn,
#             )

#     def train_dataloader(self):
#         return torch.utils.data.DataLoader(
#             self.train_ds, 
#             shuffle=True, 
#             collate_fn=collate_fn_namedtuple,
#             **self.dl_kw
#         )

#     def val_dataloader(self):
#         return torch.utils.data.DataLoader(
#             self.val_ds, 
#             shuffle=False, 
#             collate_fn=collate_fn_namedtuple,
#             **self.dl_kw
#         )

#     def test_dataloader(self):
#         return torch.utils.data.DataLoader(
#             self.test_ds, 
#             shuffle=False, 
#             collate_fn=collate_fn_namedtuple,
#             **self.dl_kw
#         )
    
class LazyXrDataset_DepthTime(LazyXrDataset_Depth):
    """
    Same as LazyXrDataset_Depth, but compatible with TransfertLazyDataModule_DepthTime.
    Adds support for providing a normalized time index if needed.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Could store extra flags if needed later (e.g., seasonal normalization)
        self.has_time_idx = True

class TransfertLazyDataModule_DepthTime(TransfertLazyDataModule_Depth):
    """
    Extends TransfertLazyDataModule_Depth to include normalized time indices.
    Produces TrainingItemwithDepthTime(input, tgt, depth_idx, time_idx).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def post_fn(self):
        means, stds = self.norm_stats()
        means = np.asarray(means, dtype=np.float32)
        stds  = np.asarray(stds,  dtype=np.float32)

        # full "global" component coordinate
        if isinstance(self.input_da, xr.Dataset):
            if 'component' in self.input_da.coords:
                full_comp = self.input_da['component']
            else:
                first_var = next(iter(self.input_da.data_vars))
                full_comp = self.input_da[first_var]['component']
        else:
            full_comp = self.input_da['component']

        # full "global" time coordinate
        if 'time' in self.input_da.coords:
            full_time = self.input_da['time']
            if np.issubdtype(full_time.dtype, np.datetime64):
                tvals = (full_time - full_time.min()) / np.timedelta64(1, 'D')
            else:
                tvals = np.asarray(full_time.values, dtype=np.float32)
            tmin, tmax = float(tvals.min()), float(tvals.max())
        else:
            tmin, tmax = 0.0, 1.0

        # -----------------------------
        def _norm_xr(da: xr.DataArray) -> xr.DataArray:
            comp = da['component']
            m_full = xr.DataArray(means, dims=['component'], coords={'component': full_comp})
            s_full = xr.DataArray(stds,  dims=['component'], coords={'component': full_comp})
            m = m_full.sel(component=comp)
            s = s_full.sel(component=comp)
            return (da - m) / (s + 1e-8)

        def _make_item(x: xr.DataArray):
            # ---- depth index normalization ----
            comp = x["component"].values
            comp_idx_torch = torch.as_tensor(np.round(comp).astype(int), dtype=torch.long)
            zmin = float(full_comp.min())
            zmax = float(full_comp.max())
            depth_idx_norm = (comp - zmin) / (zmax - zmin + 1e-8)
            depth_idx_norm = torch.as_tensor(depth_idx_norm.mean(), dtype=torch.float32).unsqueeze(0)

            # ---- time index normalization ----
            if "time" in x.coords:
                tvals = x["time"].values
                if np.issubdtype(tvals.dtype, np.datetime64):
                    tvals = (tvals - x["time"].values.min()) / np.timedelta64(1, 'D')
                else:
                    tvals = np.asarray(tvals, dtype=np.float32)
                tvals = (tvals - tmin) / (tmax - tmin + 1e-8)
                time_idx_norm = torch.as_tensor(tvals.mean(), dtype=torch.float32).unsqueeze(0)
            else:
                time_idx_norm = torch.zeros(1)

            # ---- normalize data ----
            if "variable" in x.dims:
                inp = _norm_xr(x.isel(variable=0))
                tgt = _norm_xr(x.isel(variable=1))
            else:
                fld = _norm_xr(x)
                inp = fld
                tgt = fld

            inp_np = inp.transpose("time", "component", "lat", "lon").data.astype(np.float32)
            tgt_np = tgt.transpose("time", "component", "lat", "lon").data.astype(np.float32)

            inp_t = torch.as_tensor(inp_np)
            tgt_t = torch.as_tensor(tgt_np)

            return TrainingItemwithDepthTime(inp_t, tgt_t, depth_idx_norm, time_idx_norm)

        return _make_item

    def setup(self, stage='test'):
        post_fn = self.post_fn()
        if stage == 'fit':
            train_data = self.input_da.sel(self.domains['train'])
            train_xrds_kw = deepcopy(self.xrds_kw)
            self.train_ds = LazyXrDataset_DepthTime(
                train_data,
                **train_xrds_kw,
                depth_dim='component',
                postpro_fn=post_fn,
            )
            if self.aug_kw:
                self.train_ds = AugmentedDataset(self.train_ds, **self.aug_kw)

            self.val_ds = LazyXrDataset_DepthTime(
                self.input_da.sel(self.domains['val']),
                **self.xrds_kw,
                depth_dim='component',
                postpro_fn=post_fn,
            )
        else:
            self.test_ds = LazyXrDataset_DepthTime(
                self.input_da.sel(self.domains['test']),
                **self.xrds_kw,
                depth_dim='component',
                postpro_fn=post_fn,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            shuffle=True,
            collate_fn=self.collate_fn_namedtuple,
            **self.dl_kw
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            shuffle=False,
            collate_fn=self.collate_fn_namedtuple,
            **self.dl_kw
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_ds,
            shuffle=False,
            collate_fn=self.collate_fn_namedtuple,
            **self.dl_kw
        )
