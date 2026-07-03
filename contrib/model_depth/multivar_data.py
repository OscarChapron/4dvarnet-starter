import xarray as xr
import numpy as np
import functools as ft
import einops
import torch
import torch.nn as nn
import collections
import src.data
import random
from copy import deepcopy
from pathlib import Path
import pandas as pd
from src.data import AugmentedDataset, BaseDataModule, XrDataset, TrainingItem
from src.utils import get_constant_crop
from contrib import transfert
from typing import Dict, Any, Optional, Union
from types import SimpleNamespace as NS
from typing import Sequence, Tuple
import pickle
from dataclasses import dataclass, field


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
        for channel in train_data.channel.values:
            print(f'Channel: {channel}')
            print(train_data.sel(channel=channel).values)
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
        means, stds = self.norm_stats()  # shape: (component,)

        def normalize(values):
            return (values - means[None, :, None, None]) / stds[None, :, None, None]

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

        # sample_ds = (
        #     self.train_ds or self.val_ds or self.test_ds
        # )
        # c = sample_ds[0].input.shape[0]
        # t = self.xrds_kw["patch_dims"]["time"]
        # self.dim_in = c * t 

def threshold_xarray(da):
    threshold = 999
    da = xr.where(da > threshold, 0, da)
    da = xr.where(da <= 0, 0, da)
    return da

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

    # ── NEW: loop once over the stacked variable axis ───────────────────────
    # full_ds coords: `variable` holds the short names (= keys in vars_info)
    da_list = []
    for v in full_ds.variable.values:
        da_v = full_ds.sel(variable=v)
        if getattr(vars_info[v], "apply_threshold", False):
            da_v = da_v.pipe(threshold_xarray)
        da_list.append(da_v)

    # Re‑assemble the DataArray with the original ordering
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

        # ----------- first pass: masked copy, if requested -----------------
        if getattr(info, 'mask_path', None):
            masked_ds = open_var_dataset(
                info.var_path, var, info.var_name, domain, drop_depth,
                extra_dim   = getattr(info, 'extra_dim', None),
                fill_nan    = getattr(info, 'fill_nan',  None),
                threshold   = getattr(info, 'threshold', None),
                mask_path   = info.mask_path,
            )
            full_dataset = merge_datasets(full_dataset, masked_ds)
            for new_var in masked_ds.data_vars:
                multivar_information[new_var] = {
                    'input_arch':  'no_input',
                    'output_arch': 'no_output',
                    'masked_obs':  True,
                }

        # ----------- second pass: normal (possibly multi-slice) copy -------
        clean_ds = open_var_dataset(
            info.var_path, var, info.var_name, domain, drop_depth,
            extra_dim   = getattr(info, 'extra_dim', None),
            fill_nan    = getattr(info, 'fill_nan',  None),
            threshold   = getattr(info, 'threshold', None),
            mask_path   = None,
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

    # ---------------------------------------------------- final stacking ---
    full_dataset = (
        full_dataset
        .sel(domain)
        [list(multivar_information.keys())]   # keep the chosen order
        .transpose('time', 'lat', 'lon', ...)
        .to_array()
    )

    print(full_dataset.var())   # quick sanity check
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
    # ------------------------------------------------------------------ open
    data = xr.open_dataset(var_path)[var_name]
    if 'depth' in data.dims and drop_depth:
        data = data.drop('depth')

    if 'latitude' in data.dims:
        data = data.rename({'latitude': 'lat', 'longitude': 'lon'})

    # ---------------------------------------------------------------- domain
    trimmed_domain = {k: v for k, v in domain.items() if k in data.dims}
    data = data.sel(trimmed_domain)

    # -------------------------------------------------------- fill / threshold
    if fill_nan is not None:
        data = data.fillna(fill_nan)

    if threshold is not None:
        data = xr.where(data > threshold, 0, data)
        data = xr.where(data <= 0, 0, data)

    # --------------------------------------------------- split extra dimension
    if extra_dim and extra_dim in data.dims:
        ds = xr.Dataset({
            f"{var}_{label}": data.sel({extra_dim: label}).drop_vars(extra_dim)
            for label in data[extra_dim].values
        })
    else:
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
