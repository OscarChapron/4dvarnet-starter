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
import pytorch_lightning as pl
from omegaconf import ListConfig, DictConfig
from copy import deepcopy
from pathlib import Path
import pandas as pd
import torch.nn.functional as F
from src.data import AugmentedDataset, BaseDataModule, XrDataset
from src.utils import get_constant_crop
from collections import namedtuple
from contrib import transfert
from types import SimpleNamespace
from typing import Sequence, Tuple, Union, Dict, Any
from dataclasses import dataclass, field
from typing import Optional
import itertools  
TrainingItem = namedtuple('TrainingItem', ['input', 'tgt'])


class TransfertXrDatasetv1(src.data.XrDataset):
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

class TransfertDataModulev1(src.data.BaseDataModule):
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

        # Return them as NumPy arrays for convenience
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
        if self.norm_type == 'z_score':
            means, stds = self.norm_stats()  # shape: (component,)

            def normalize(values):
                return (values - means[:, None, None]) / stds[:, None, None]

        elif self.norm_type == 'min_max':
            min_vals, max_vals = self.norm_stats()  # shape: (component,)

            def normalize(values):
                return (values - min_vals[:, None, None]) / (max_vals[:, None, None] - min_vals[:, None, None])

        else:
            def normalize(values):
                return values

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
            
            self.train_ds = TransfertXrDatasetv1(
                train_data, **train_xrds_kw, postpro_fn=post_fn,
            )
            if self.aug_kw:
                self.train_ds = AugmentedDataset(self.train_ds, **self.aug_kw)

            self.val_ds = TransfertXrDatasetv1(
                self.input_da.sel(self.domains['val']),
                **self.xrds_kw,
                postpro_fn=post_fn,
            )
        else:
            self.test_ds = TransfertXrDatasetv1(
                self.input_da.sel(self.domains['test']),
                **self.xrds_kw,
                postpro_fn=post_fn,
            )

class TransfertXrDataset(XrDataset):
    def reconstruct(self, batches, weight=None):
        """
        Parameters
        ----------
        batches : list[list[torch.Tensor]]
            Each outer-list element is a mini-batch; each inner element is
            a patch (no shuffling assumed).
            Shape of one patch  =
                (<non-patch dims> , *patch_dims.values()).

        weight : np.ndarray | xr.DataArray | None
            Spatial weighting over one *entire* patch.  Must have one entry
            per entry in `self.patch_dims` and will be broadcast over the
            non-patch (lead-time, channel, …) dimensions.

        Returns
        -------
        xr.DataArray
            The stitched prediction with proper coordinates.
        """
        items = list(itertools.chain(*batches))
        return self.reconstruct_from_items(items, weight)

    # # ──────────────────────────────────────────────────────────────────────────
    # def reconstruct_from_items(self, items, weight=None):
    #     # ------------------------------------------------------------------ #
    #     # 1.  Handle the weight tensor                                       #
    #     # ------------------------------------------------------------------ #
    #     patch_dims      = list(self.patch_dims.keys())      # e.g. ['z','lat','lon']
    #     patch_shape     = list(self.patch_dims.values())    # e.g. [ 5 ,  32 ,  32 ]

    #     if weight is None:
    #         weight = np.ones(patch_shape, dtype=np.float32)
    #     else:
    #         weight = np.asarray(weight, dtype=np.float32).reshape(patch_shape)

    #     w = xr.DataArray(weight, dims=patch_dims)   # (z,lat,lon) or (lat,lon) …

    #     # ------------------------------------------------------------------ #
    #     # 2.  Split the item shape into “non-patch” and “patch” axes          #
    #     # ------------------------------------------------------------------ #
    #     n_patch_axes    = len(patch_dims)
    #     n_non_patch     = items[0].ndim - n_patch_axes
    #     non_patch_dims  = [f"v{i}" for i in range(n_non_patch)]   # v0, v1, …

    #     all_dims        = non_patch_dims + patch_dims            # final dim list

    #     # ------------------------------------------------------------------ #
    #     # 3.  Wrap every torch-tensor patch into a DataArray with coords      #
    #     # ------------------------------------------------------------------ #
    #     coords_per_patch = self.get_coords()   # already aligned with items

    #     das = []
    #     for tensor, patch_coords in zip(items, coords_per_patch):
    #         da = xr.DataArray(
    #             tensor.detach().cpu().numpy(),
    #             dims     = all_dims,
    #             coords   = {**{d: np.arange(s) for d, s
    #                            in zip(non_patch_dims, tensor.shape[:n_non_patch])},
    #                         **patch_coords.coords}
    #         )
    #         das.append(da)

    #     # ------------------------------------------------------------------ #
    #     # 4.  Prepare the big, empty canvas                                   #
    #     # ------------------------------------------------------------------ #
    #     # sizes for the non-patch part (lead-time, channel, …)
    #     non_patch_sizes = dict(zip(non_patch_dims,
    #                                items[0].shape[:n_non_patch]))

    #     # sizes for the spatial/patch part (t, z, lat, lon …)
    #     dataset_sizes   = {d: self.da[d].size for d in patch_dims}

    #     rec_da  = xr.DataArray(
    #         np.zeros([*non_patch_sizes.values(), *dataset_sizes.values()],
    #                  dtype=np.float32),
    #         dims   = all_dims,
    #         coords = {**{d: np.arange(s) for d, s in non_patch_sizes.items()},
    #                   **{d: self.da[d]   for d in patch_dims}}
    #     )
    #     cnt_da  = xr.zeros_like(rec_da)

    #     # ------------------------------------------------------------------ #
    #     # 5.  Stitch the patches                                              #
    #     # ------------------------------------------------------------------ #
    #     for da in das:
    #         rec_da.loc[da.coords]  = rec_da.sel(da.coords)  + da * w
    #         cnt_da.loc[da.coords]  = cnt_da.sel(da.coords)  + w

    #     # ------------------------------------------------------------------ #
    #     # 6.  Normalise and return                                            #
    #     # ------------------------------------------------------------------ #
    #     return rec_da / cnt_da

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

class TransfertDataModule(transfert.BaseDataModule_Fasc):
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
    ).transpose('time', 'component', 'lat', 'lon').to_array()
    print(ds.shape)
    
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
    return getattr(spec, key, default)        # dataclass path

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

        # ----------- second pass: normal (possibly multi-slice) copy -------
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
    var_order = list(multivar_information.keys())           # keep the original order
    roles = [
        "tgt" if info["output_arch"] != "no_output" else "inp"
        for info in multivar_information.values()
    ]
    # ---------------------------------------------------- final stacking ---
    full_dataset = (
        full_dataset
        .sel(domain)
        [var_order]   # keep the chosen order
        .transpose('time', 'lat', 'lon', ...)
        .to_array()
        .assign_coords(variable=("variable", roles))
    )

    # optional: keep the original names in an attribute for debugging
    #full_dataset.attrs["var_order"] = var_order
    # -------------------------------------------------------------------------------

    # don't print with the non-existent .var_name() anymore
    print("shape:", full_dataset.shape)
    print("coords:", full_dataset.coords)
    print("dims:", full_dataset.dims)
    print("values:", full_dataset.values)
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

    # --------------------------------------------------- normalise feature dim
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
    # n_chan = stacked.sizes["channel"]               # make sure we really have 4
    # stacked = stacked.assign_coords(
    # channel=("channel", np.arange(n_chan, dtype="int32"))
    # )
    print("Non-NaN values in input:", np.count_nonzero(~np.isnan(stacked.sel(variable="input").values)))
    print("Non-NaN values in tgt:", np.count_nonzero(~np.isnan(stacked.sel(variable="tgt").values)))
    print("stacked shape:", stacked.shape)
    print("stacked coords:", stacked.coords)
    print("stacked dims:", stacked.dims)
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
    
class Lit4dVarNet_depth(transfert.Lit4dVarNet_Fasc):
    @staticmethod
    def weighted_mse_norm(err, weight, norm=None):
        err_w = err * weight[None, ...]
        B, TC, H, W = err.shape
        C = norm.numel()
        T = TC // C
        if norm is not None:
            n = norm.repeat(T).view(1, TC, 1, 1)
            err_w = err_w * n
        non_zeros = (torch.ones_like(err) * weight[None, ...]) == 0.0
        err_num = err.isfinite() & ~non_zeros
        if err_num.sum() == 0:
            return torch.scalar_tensor(1000.0, device=err_num.device).requires_grad_()
        loss = F.mse_loss(err_w[err_num], torch.zeros_like(err_w[err_num]))
        return loss
    
    def weighted_mse(self, err, weight):
        # err: (B,TC,H,W), weight: (H,W)
        w = weight[None, None, ...]                 # (1,1,H,W)
        sel = err.isfinite() & (w != 0)
        if not sel.any():
            return torch.scalar_tensor(1000.0, device=err.device, dtype=err.dtype)
        return F.mse_loss((err * w)[sel], torch.zeros((), device=err.device, dtype=err.dtype))

    def step(self, batch, phase=""):
        if self.training and batch.tgt.isfinite().float().mean() < 0.9:
            return None, None
        batch = batch._replace(
            input=batch.input.view(batch.input.size(0), -1, batch.input.size(-2), batch.input.size(-1)),
            tgt=batch.tgt.view(batch.tgt.size(0), -1, batch.tgt.size(-2), batch.tgt.size(-1))
        )
        masked_input = batch.input.clone()
    
        C = len(self.norm_stats[0])               # how many time steps per patch
        B, TC, H, W = batch.input.shape
        T = TC // C 
        # for i in range(batch.input.size(0)):  # Assuming the first dimension is the batch size
        #     sr = self.sampling_rate
        #     if isinstance(self.sampling_rate, (list, tuple, ListConfig)) and len(self.sampling_rate) == 2:
        #         sr = random.uniform(self.sampling_rate[0], self.sampling_rate[1])
        
        #     mask = (torch.rand(batch.input[i].size()) > sr).to(batch.input.device)
        #     masked_input[i][mask] = float('nan')
        for i in range(B):  # Assuming the first dimension is the batch size
            sr = self.sampling_rate
            if isinstance(self.sampling_rate, (list, tuple, ListConfig)) and len(sr) == 2:
                sr = random.uniform(sr[0], sr[1])

            # Create the mask for the first component
            spatial_mask = (torch.rand((H, W), device=batch.input.device) > sr)
            mask_3d = spatial_mask.unsqueeze(0).expand(C, -1, -1)
            mask_3d = mask_3d.repeat_interleave(T, dim=0)
            # Apply the same mask to all components
            masked_input[i][mask_3d] = float("nan")

        batch = batch._replace(input=masked_input)
        
        if self.solver.n_step > 0:

            loss, out = self.base_step(batch, phase)
            norm = torch.tensor(self.norm_stats[0], device=out.device, dtype=out.dtype)
            grad_loss = self.weighted_mse( err = kfilts.sobel(out) - kfilts.sobel(batch.tgt), weight=self.rec_weight)
            prior_cost = self.solver.prior_cost(self.solver.init_state(batch, out))
        
            self.log( f"{phase}_gloss", grad_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log( f"{phase}_prior_cost", prior_cost, prog_bar=True, on_step=False, on_epoch=True)
            #weight_obs = self.solver.obs_cost.weight1_torch
            #weight_prior = self.solver.prior_cost.weight3_torch
            self.log('sampling_rate', sr, on_step=False, on_epoch=True)
            #self.log('weight obs', weight_obs , on_step=False, on_epoch=True)
            #self.log('weight prior', weight_prior,on_step=False, on_epoch=True)

            training_loss = 10 * loss + 20 * prior_cost + 5 * grad_loss
            #training_loss = 50 * loss + 1000 * grad_loss + 1.0 * prior_cost

            return training_loss, out
        
        else:
            loss, out = self.base_step(batch, phase)
            return loss, out
    
    def base_step(self, batch, phase=""):
        out = self(batch=batch)
        norm = torch.tensor(self.norm_stats[0], device=out.device, dtype=out.dtype)
        #loss = self.weighted_rel_mse(out - batch.tgt, batch.tgt, self.rec_weight)
        loss = self.weighted_mse(err = out - batch.tgt, weight=self.rec_weight)
        with torch.no_grad():
            self.log(f"{phase}_mse",  loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{phase}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss, out
    
    def test_step(self, batch, batch_idx):
        print(batch.input.shape)
        if batch_idx == 0:
            self.test_data = []
        batch = batch._replace(
            input=batch.input.view(batch.input.size(0), -1, batch.input.size(-2), batch.input.size(-1)),
            tgt=batch.tgt.view(batch.tgt.size(0), -1, batch.tgt.size(-2), batch.tgt.size(-1))
        )
        batch_input_clone = batch.input.clone()
        masked_input = batch.input.clone()

        C = len(self.norm_stats[0])               # how many time steps per patch
        B, TC, H, W = batch.input.shape
        T = TC // C
        # for i in range(batch.input.size(0)):  # Assuming the first dimension is the batch size
        #     sr = self.sampling_rate
        #     if isinstance(self.sampling_rate, (list, tuple, ListConfig)) and len(self.sampling_rate) == 2:
        #         sr = random.uniform(self.sampling_rate[0], self.sampling_rate[1])
        
        #     mask = (torch.rand(batch.input[i].size()) > sr).to(batch.input.device)
        #     masked_input[i][mask] = float('nan')
        for i in range(B):  # Assuming the first dimension is the batch size
            sr = self.sampling_rate
            if isinstance(self.sampling_rate, (list, tuple, ListConfig)) and len(sr) == 2:
                sr = random.uniform(sr[0], sr[1])

            # Create the mask for the first component
            spatial_mask = (torch.rand((H, W), device=batch.input.device) > sr)
            mask_3d = spatial_mask.unsqueeze(0).expand(C, -1, -1)
            mask_3d = mask_3d.repeat_interleave(T, dim=0)
            # Apply the same mask to all components
            masked_input[i][mask_3d] = float("nan")
        batch = batch._replace(input=masked_input)
    
        out = self(batch=batch)

        if self.norm_type == 'z_score':
            batch_input_clone_unnorm = self._denormalize_zscore(batch_input_clone, self.norm_stats)
            print(batch_input_clone_unnorm.shape)
            masked_input_unnorm      = self._denormalize_zscore(masked_input,       self.norm_stats)
            batch_tgt_unnorm        = self._denormalize_zscore(batch.tgt,          self.norm_stats)
            out_unnorm              = self._denormalize_zscore(out,                self.norm_stats)
            self.test_data.append(torch.stack(
                [   batch_input_clone_unnorm.cpu(),
                    masked_input_unnorm.cpu(),
                    batch_tgt_unnorm.cpu(),
                    out_unnorm.squeeze(dim=-1).detach().cpu(),
                ],
                dim=1,
            ))
        
        if self.norm_type == 'min_max':
            batch_input_clone_unnorm = self._denormalize_minmax(batch_input_clone, self.norm_stats)
            masked_input_unnorm      = self._denormalize_minmax(masked_input,       self.norm_stats)
            batch_tgt_unnorm        = self._denormalize_minmax(batch.tgt,          self.norm_stats)
            out_unnorm              = self._denormalize_minmax(out,                self.norm_stats)
            self.test_data.append(torch.stack(
                [   batch_input_clone_unnorm.cpu(),
                    masked_input_unnorm.cpu(),
                    batch_tgt_unnorm.cpu(),
                    out_unnorm.squeeze(dim=-1).detach().cpu(),
                ],
                dim=1,
            ))
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
        C = len(means)                 # <- infer it
        assert TC % C == 0, "TC is not divisible by C"
        T = TC // C
        # Reshape to (B, T, C, H, W)
        reshaped = tensor_4d.view(B, T, C, H, W)

        # For each component c, apply x * (max_c - min_c) + min_c
        for c_idx, (min_c, max_c) in enumerate(zip(minmax_list[0], minmax_list[1])):
            reshaped[:, :, c_idx, :, :] = reshaped[:, :, c_idx, :, :] * (max_c - min_c) + min_c

        return reshaped
    
    def _denormalize_zscore(self, tensor_4d, norm_list):
        """
        tensor_4d:  (B, T*C, H, W)
        norm_list: (means, stds)  length = C
        returns    (B, T, C, H, W)
        """
        B, TC, H, W = tensor_4d.shape
        means, stds = norm_list
        C = len(means)                 # <- infer it
        assert TC % C == 0, "TC is not divisible by C"
        T = TC // C

        reshaped = tensor_4d.view(B, T, C, H, W)
        for c_idx, (mean_c, std_c) in enumerate(zip(means, stds)):
            reshaped[:, :, c_idx, :, :] = reshaped[:, :, c_idx, :, :] * std_c + mean_c
        return reshaped
    
    @property
    def test_quantities(self):
        return ['input', 'inp', 'tgt', 'out']

    def on_test_epoch_end(self):
        rec_da = self.trainer.test_dataloaders.dataset.reconstruct(
            self.test_data, self.rec_weight.cpu().numpy()
        )

        if isinstance(rec_da, list):
            rec_da = rec_da[0]

        self.test_data = rec_da.assign_coords(
            dict(v0=self.test_quantities)
        ).to_dataset(dim='v0')

        metric_data = self.test_data.pipe(self.pre_metric_fn)
        metrics = pd.Series({
            metric_n: metric_fn(metric_data) 
            for metric_n, metric_fn in self.metrics.items()
        })

        print(metrics.to_frame(name="Metrics").to_markdown())
        if self.logger:
            self.test_data.to_netcdf(Path(self.logger.log_dir) / 'test_data.nc')
            print(Path(self.trainer.log_dir) / 'test_data.nc')
            self.logger.log_metrics(metrics.to_dict())
 

# class Lit4dVarNet_depthv1(src.models.Lit4dVarNet):
#     def step(self, batch, phase=""):
#         if self.training and batch.tgt.isfinite().float().mean() < 0.9:
#             return None, None
#         batch = batch._replace(
#             input=batch.input.view(batch.input.size(0), -1, batch.input.size(-2), batch.input.size(-1)),
#             tgt=batch.tgt.view(batch.tgt.size(0), -1, batch.tgt.size(-2), batch.tgt.size(-1))
#         )
#         masked_input = batch.input.clone()
#         for i in range(batch.input.size(0)):  # Assuming the first dimension is the batch size
#             sr = self.sampling_rate
#             if isinstance(self.sampling_rate, (list, tuple, ListConfig)) and len(self.sampling_rate) == 2:
#                 sr = random.uniform(self.sampling_rate[0], self.sampling_rate[1])
        
#             mask = (torch.rand(batch.input[i].size()) > sr).to(batch.input.device)
#             masked_input[i][mask] = float('nan')
    
#         batch = batch._replace(input=masked_input)
#         if self.solver.n_step > 0:

#             loss, out = self.base_step(batch, phase)
#             grad_loss = self.weighted_mse( kfilts.sobel(out) - kfilts.sobel(batch.tgt), self.rec_weight)
#             prior_cost = self.solver.prior_cost(self.solver.init_state(batch, out))
        
#             self.log( f"{phase}_gloss", grad_loss, prog_bar=True, on_step=False, on_epoch=True)
#             self.log( f"{phase}_prior_cost", prior_cost, prog_bar=True, on_step=False, on_epoch=True)
#             #weight_obs = self.solver.obs_cost.weight1_torch
#             #weight_prior = self.solver.prior_cost.weight3_torch
#             self.log('sampling_rate', sr, on_step=False, on_epoch=True)
#             #self.log('weight obs', weight_obs , on_step=False, on_epoch=True)
#             #self.log('weight prior', weight_prior,on_step=False, on_epoch=True)

#             training_loss = 10 * loss + 20 * prior_cost + 5 * grad_loss
#             #training_loss = 50 * loss + 1000 * grad_loss + 1.0 * prior_cost

#             return training_loss, out
        
#         else:
#             loss, out = self.base_step(batch, phase)
#             return loss, out
    
#     def base_step(self, batch, phase=""):
#         # batch = batch._replace(input = batch.input / torch.bernoulli(torch.full(batch.input.size(), self.sampling_rate)).to('cuda:0'))
#         out = self(batch=batch)
#         #loss = self.weighted_rel_mse(out - batch.tgt, batch.tgt, self.rec_weight)
#         loss = self.weighted_mse(out - batch.tgt, self.rec_weight)
#         with torch.no_grad():
#             self.log(f"{phase}_mse",  loss, prog_bar=True, on_step=False, on_epoch=True)
#             self.log(f"{phase}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
#         return loss, out
    
#     def test_step(self, batch, batch_idx):
#         if batch_idx == 0:
#             self.test_data = []
#         batch = batch._replace(
#             input=batch.input.view(batch.input.size(0), -1, batch.input.size(-2), batch.input.size(-1)),
#             tgt=batch.tgt.view(batch.tgt.size(0), -1, batch.tgt.size(-2), batch.tgt.size(-1))
#         )
#         batch_input_clone = batch.input.clone()
#         masked_input = batch.input.clone()

#         for i in range(batch.input.size(0)):  # Assuming the first dimension is the batch size
#             sr = self.sampling_rate
#             if isinstance(self.sampling_rate, (list, tuple, ListConfig)) and len(self.sampling_rate) == 2:
#                 sr = random.uniform(self.sampling_rate[0], self.sampling_rate[1])
        
#             mask = (torch.rand(batch.input[i].size()) > sr).to(batch.input.device)
#             masked_input[i][mask] = float('nan')
    
#         batch = batch._replace(input=masked_input)
    
#         out = self(batch=batch)

#         if self.norm_type == 'z_score':
#             batch_input_clone_unnorm = self._denormalize_zscore(batch_input_clone, self.norm_stats)
#             print(batch_input_clone_unnorm.shape)
#             masked_input_unnorm      = self._denormalize_zscore(masked_input,       self.norm_stats)
#             batch_tgt_unnorm        = self._denormalize_zscore(batch.tgt,          self.norm_stats)
#             out_unnorm              = self._denormalize_zscore(out,                self.norm_stats)
#             self.test_data.append(torch.stack(
#                 [   batch_input_clone_unnorm.cpu(),
#                     masked_input_unnorm.cpu(),
#                     batch_tgt_unnorm.cpu(),
#                     out_unnorm.squeeze(dim=-1).detach().cpu(),
#                 ],
#                 dim=1,
#             ))
        
#         if self.norm_type == 'min_max':
#             batch_input_clone_unnorm = self._denormalize_minmax(batch_input_clone, self.norm_stats)
#             masked_input_unnorm      = self._denormalize_minmax(masked_input,       self.norm_stats)
#             batch_tgt_unnorm        = self._denormalize_minmax(batch.tgt,          self.norm_stats)
#             out_unnorm              = self._denormalize_minmax(out,                self.norm_stats)
#             self.test_data.append(torch.stack(
#                 [   batch_input_clone_unnorm.cpu(),
#                     masked_input_unnorm.cpu(),
#                     batch_tgt_unnorm.cpu(),
#                     out_unnorm.squeeze(dim=-1).detach().cpu(),
#                 ],
#                 dim=1,
#             ))
#     def _denormalize_minmax(self, tensor_4d, minmax_list):
#         """
#         Denormalize a 4D tensor (B, T*C, H, W) given a list of (min_c, max_c)
#         for each component c.

#         Args:
#             tensor_4d (torch.Tensor): shape (batch, T*C, height, width)
#             minmax_list (List[Tuple[float, float]]): 
#                 list of (min_value, max_value) for each component c

#         Returns:
#             denorm_5d (torch.Tensor): shape (batch, T, C, height, width)
#                 where each component c is mapped back to [min_c, max_c].
#         """
#         B, TC, H, W = tensor_4d.shape
#         C = 5  # number of components
#         T = TC // C           # number of timesteps
#         # Reshape to (B, T, C, H, W)
#         reshaped = tensor_4d.view(B, T, C, H, W)

#         # For each component c, apply x * (max_c - min_c) + min_c
#         for c_idx, (min_c, max_c) in enumerate(zip(minmax_list[0], minmax_list[1])):
#             reshaped[:, :, c_idx, :, :] = reshaped[:, :, c_idx, :, :] * (max_c - min_c) + min_c

#         return reshaped
    
#     def _denormalize_zscore(self, tensor_4d, norm_list):
#         """
#         tensor_4d: shape (B, T*C, H, W)
#         norm_list: list of (mean_c, std_c) for each component c
#         Returns shape (B, T, C, H, W) with each component unnormalized.
#         """
#         B, TC, H, W = tensor_4d.shape
#         C = 5       # e.g. 4 components
#         T = TC // C              # e.g. 60 // 4 = 15
#         # Reshape
#         reshaped = tensor_4d.view(B, T, C, H, W)
#         print(norm_list[0])
#         # Loop over each component c
#         for c_idx, (mean_c, std_c) in enumerate(zip(norm_list[0], norm_list[1])):
#             reshaped[:, :, c_idx, :, :] = reshaped[:, :, c_idx, :, :] * std_c + mean_c
#         # Reshape back to the original shape (B, T*C, H, W)
#         output = reshaped#.view(B, TC, H, W)
#         print('-----')
#         print(output.shape)
#         return output
    
#     @property
#     def test_quantities(self):
#         return ['input', 'inp', 'tgt', 'out']

#     def on_test_epoch_end(self):
#         rec_da = self.trainer.test_dataloaders.dataset.reconstruct(
#             self.test_data, self.rec_weight.cpu().numpy()
#         )

#         if isinstance(rec_da, list):
#             rec_da = rec_da[0]

#         self.test_data = rec_da.assign_coords(
#             dict(v0=self.test_quantities)
#         ).to_dataset(dim='v0')

#         metric_data = self.test_data.pipe(self.pre_metric_fn)
#         metrics = pd.Series({
#             metric_n: metric_fn(metric_data) 
#             for metric_n, metric_fn in self.metrics.items()
#         })

#         print(metrics.to_frame(name="Metrics").to_markdown())
#         if self.logger:
#             self.test_data.to_netcdf(Path(self.logger.log_dir) / 'test_data.nc')
#             print(Path(self.trainer.log_dir) / 'test_data.nc')
#             self.logger.log_metrics(metrics.to_dict())

