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
import src.unet
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
torch.set_float32_matmul_precision('high')
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

# class Lit4dVarNet_depthUnet(transfert.Lit4dVarNet_Fasc):
#     @staticmethod
#     def weighted_mse_norm(err, weight, norm=None):
#         err_w = err * weight[None, ...]
#         B, TC, H, W = err.shape
#         C = norm.numel()
#         T = TC // C
#         if norm is not None:
#             n = norm.repeat(T).view(1, TC, 1, 1)
#             err_w = err_w * n
#         non_zeros = (torch.ones_like(err) * weight[None, ...]) == 0.0
#         err_num = err.isfinite() & ~non_zeros
#         if err_num.sum() == 0:
#             return torch.scalar_tensor(1000.0, device=err_num.device).requires_grad_()
#         loss = F.mse_loss(err_w[err_num], torch.zeros_like(err_w[err_num]))
#         return loss
    
#     def step(self, batch, phase=""):
#         if self.training and batch.tgt.isfinite().float().mean() < 0.9:
#             return None, None
#         batch = batch._replace(
#             input=batch.input.view(batch.input.size(0), -1, batch.input.size(-2), batch.input.size(-1)),
#             tgt=batch.tgt.view(batch.tgt.size(0), -1, batch.tgt.size(-2), batch.tgt.size(-1))
#         )
#         masked_input = batch.input.clone()
    
#         C = len(self.norm_stats[0])               # how many time steps per patch
#         B, TC, H, W = batch.input.shape
#         T = TC // C 
#         # for i in range(batch.input.size(0)):  # Assuming the first dimension is the batch size
#         #     sr = self.sampling_rate
#         #     if isinstance(self.sampling_rate, (list, tuple, ListConfig)) and len(self.sampling_rate) == 2:
#         #         sr = random.uniform(self.sampling_rate[0], self.sampling_rate[1])
        
#         #     mask = (torch.rand(batch.input[i].size()) > sr).to(batch.input.device)
#         #     masked_input[i][mask] = float('nan')
#         for i in range(B):  # Assuming the first dimension is the batch size
#             sr = self.sampling_rate
#             if isinstance(self.sampling_rate, (list, tuple, ListConfig)) and len(sr) == 2:
#                 sr = random.uniform(sr[0], sr[1])

#             # Create the mask for the first component
#             spatial_mask = (torch.rand((H, W), device=batch.input.device) > sr)
#             mask_3d = spatial_mask.unsqueeze(0).expand(C, -1, -1)
#             mask_3d = mask_3d.repeat_interleave(T, dim=0)
#             # Apply the same mask to all components
#             masked_input[i][mask_3d] = float("nan")

#         batch = batch._replace(input=masked_input)
        
#         if self.solver.n_step > 0:

#             loss, out = self.base_step(batch, phase)
#             norm = torch.tensor(self.norm_stats[0], device=out.device, dtype=out.dtype)
#             grad_loss = self.weighted_mse( err = kfilts.sobel(out) - kfilts.sobel(batch.tgt), weight=self.rec_weight)
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
#         out = self(batch=batch)
#         norm = torch.tensor(self.norm_stats[0], device=out.device, dtype=out.dtype)
#         #loss = self.weighted_rel_mse(out - batch.tgt, batch.tgt, self.rec_weight)
#         loss = self.weighted_mse(err = out - batch.tgt, weight=self.rec_weight)
#         with torch.no_grad():
#             self.log(f"{phase}_mse",  loss, prog_bar=True, on_step=False, on_epoch=True)
#             self.log(f"{phase}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
#         return loss, out
    
#     def test_step(self, batch, batch_idx):
#         print(batch.input.shape)
#         if batch_idx == 0:
#             self.test_data = []
#         batch = batch._replace(
#             input=batch.input.view(batch.input.size(0), -1, batch.input.size(-2), batch.input.size(-1)),
#             tgt=batch.tgt.view(batch.tgt.size(0), -1, batch.tgt.size(-2), batch.tgt.size(-1))
#         )
#         batch_input_clone = batch.input.clone()
#         masked_input = batch.input.clone()

#         C = len(self.norm_stats[0])               # how many time steps per patch
#         B, TC, H, W = batch.input.shape
#         T = TC // C
#         # for i in range(batch.input.size(0)):  # Assuming the first dimension is the batch size
#         #     sr = self.sampling_rate
#         #     if isinstance(self.sampling_rate, (list, tuple, ListConfig)) and len(self.sampling_rate) == 2:
#         #         sr = random.uniform(self.sampling_rate[0], self.sampling_rate[1])
        
#         #     mask = (torch.rand(batch.input[i].size()) > sr).to(batch.input.device)
#         #     masked_input[i][mask] = float('nan')
#         for i in range(B):  # Assuming the first dimension is the batch size
#             sr = self.sampling_rate
#             if isinstance(self.sampling_rate, (list, tuple, ListConfig)) and len(sr) == 2:
#                 sr = random.uniform(sr[0], sr[1])

#             # Create the mask for the first component
#             spatial_mask = (torch.rand((H, W), device=batch.input.device) > sr)
#             mask_3d = spatial_mask.unsqueeze(0).expand(C, -1, -1)
#             mask_3d = mask_3d.repeat_interleave(T, dim=0)
#             # Apply the same mask to all components
#             masked_input[i][mask_3d] = float("nan")
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
#         means, stds = minmax_list
#         C = len(means)                 # <- infer it
#         assert TC % C == 0, "TC is not divisible by C"
#         T = TC // C
#         # Reshape to (B, T, C, H, W)
#         reshaped = tensor_4d.view(B, T, C, H, W)

#         # For each component c, apply x * (max_c - min_c) + min_c
#         for c_idx, (min_c, max_c) in enumerate(zip(minmax_list[0], minmax_list[1])):
#             reshaped[:, :, c_idx, :, :] = reshaped[:, :, c_idx, :, :] * (max_c - min_c) + min_c

#         return reshaped
    
#     def _denormalize_zscore(self, tensor_4d, norm_list):
#         """
#         tensor_4d:  (B, T*C, H, W)
#         norm_list: (means, stds)  length = C
#         returns    (B, T, C, H, W)
#         """
#         B, TC, H, W = tensor_4d.shape
#         means, stds = norm_list
#         C = len(means)                 # <- infer it
#         assert TC % C == 0, "TC is not divisible by C"
#         T = TC // C

#         reshaped = tensor_4d.view(B, T, C, H, W)
#         for c_idx, (mean_c, std_c) in enumerate(zip(means, stds)):
#             reshaped[:, :, c_idx, :, :] = reshaped[:, :, c_idx, :, :] * std_c + mean_c
#         return reshaped
    
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

class Lit4dVarNet_depth(transfert.Lit4dVarNet_Fasc):
    def __init__(self, *args, rec_weight_coeff, unet_fuser: nn.Module, **kwargs):
        """
        unet_2D: nn.Module mapping (B, T*C, H, W) -> (B, T*C, H, W).
        """
        super().__init__(*args, **kwargs)
        self.register_buffer('rec_weight_coeff', torch.from_numpy(rec_weight_coeff), persistent=True)
        self.fuser = unet_fuser

    def _shape_info(self, x4):
        """Infer (B,T,C,H,W) from a 4D (B, TC, H, W) using C from norm_stats."""
        if not torch.is_tensor(x4) or x4.dim() != 4:
            raise ValueError(f"_shape_info expects 4D tensor, got {type(x4)} with shape "
                             f"{getattr(x4, 'shape', None)}")
        B, TC, H, W = x4.shape
        C = len(self.norm_stats[0])
        assert TC % C == 0, f"TC={TC} not divisible by C={C}"
        T = TC // C
        return B, T, C, H, W
    
    def _apply_spatially_tied_mask(self, x: torch.Tensor):
        """
        One (H,W) mask per sample, replicated over all (T,C).
        Returns: masked_x, mean_sr
        """
        x = x.clone()
        B, T, C, H, W = self._shape_info(x)
        TC = T * C

        srs = []
        for b in range(B):
            sr = self.sampling_rate
            if isinstance(sr, (list, tuple, ListConfig)) and len(sr) == 2:
                sr = random.uniform(sr[0], sr[1])
            smask = (torch.rand((H, W), device=x.device) > sr)  # True -> NaN
            mask_3d = smask.unsqueeze(0).expand(TC, -1, -1)     # (TC,H,W)
            x[b][mask_3d] = float("nan")
            srs.append(sr)

        mean_sr = float(sum(srs) / len(srs)) if srs else float(self.sampling_rate)
        return x, mean_sr
    
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
    
    def _denormalize_zscore(self, tensor_4d, norm_list):
        """
        tensor_4d:  (B, T*C, H, W)
        norm_list: (means, stds)  length = C
        returns    (B, T, C, H, W)
        """
        B, TC, H, W = tensor_4d.shape
        means, stds = norm_list
        C = len(means)
        assert TC % C == 0, "TC is not divisible by C"
        T = TC // C

        reshaped = tensor_4d.view(B, T, C, H, W)
        for c_idx, (mean_c, std_c) in enumerate(zip(means, stds)):
            reshaped[:, :, c_idx, :, :] = reshaped[:, :, c_idx, :, :] * std_c + mean_c
        return reshaped
    
    def _std_scale2(self, ref_tensor: torch.Tensor) -> torch.Tensor:
        """
        Return a scalar torch tensor equal to mean(std^2) on ref_tensor.device/dtype.
        Works when norm_stats is (means, stds) with stds as list/np.ndarray/torch.Tensor.
        """
        _, stds = self.norm_stats
        stds_t = torch.as_tensor(stds, device=ref_tensor.device, dtype=ref_tensor.dtype)
        if stds_t.ndim > 0:
            stds_t = (stds_t ** 2).mean()           # mean over components
        else:
            stds_t = stds_t ** 2                     # already scalar
        return stds_t
    
    def _run_fdv_per_component(self, batch_tc, masked_tc, phase = ''):
        inp4 = masked_tc.input
        tgt4 = batch_tc.tgt

        B, T, C, H, W = self._shape_info(inp4)
        inp_TCHW = inp4.view(B, T, C, H, W)
        tgt_TCHW = tgt4.view(B, T, C, H, W)
        print(f"Running FDV per component on {C} components, each with {T} time steps")
        outs = []
        prior_total = 0.0
        grad_total = 0.0
        mse_total = 0.0
        loss_total = 0.0
        training_total = 0.0
        
        for c in range(C):
            sub = batch_tc._replace(
                input=inp_TCHW[:, :, c, :, :].contiguous(),
                tgt=  tgt_TCHW[:, :, c, :, :].contiguous(),
            )
            print(f" Component {c}")
            print(sub.input.shape)      
            out_c = super().forward(sub)
            if out_c.dim() == 3:                   # (B,H,W) -> (B,T=1,H,W)
                out_c = out_c.unsqueeze(1)
            assert out_c.dim() == 4, f"Per-comp FDV must return 4D (B,T,H,W), got {out_c.shape}"
            
            recon = self.weighted_mse(out_c - sub.tgt, self.rec_weight)
            gloss = self.weighted_mse(kfilts.sobel(out_c) - kfilts.sobel(sub.tgt), self.rec_weight)
            prior = self.solver.prior_cost(self.solver.init_state(sub, out_c))
            
            training_total = training_total + 50 * recon + 1000 * gloss + 1.0 * prior
            loss_total = loss_total + recon
            mse_total = mse_total + 1000*recon
            grad_total = grad_total + gloss

            if self.solver.n_step > 0:
                prior_total = prior_total + self.solver.prior_cost(self.solver.init_state(sub, out_c))
            
            outs.append(out_c)

        outs_TCHW = torch.stack(outs, dim=2)                 # (B,T,C,H,W)
        outs_TCHW = outs_TCHW.contiguous().view(B, T * C, H, W)  # (B,TC,H,W)
        return outs_TCHW, training_total

    # -------------------- Lightning hooks --------------------
    def step(self, batch, phase=""):
        if self.training and batch.tgt.isfinite().float().mean() < 0.9:
            return None, None

        # flatten to (B, TC, H, W)
        batch = batch._replace(
            input=batch.input.view(batch.input.size(0), -1, batch.input.size(-2), batch.input.size(-1)),
            tgt=  batch.tgt.view(  batch.tgt.size(0),   -1,   batch.tgt.size(-2),   batch.tgt.size(-1))
        )

        # spatially-tied mask across all (T,C)
        masked_input, mean_sr = self._apply_spatially_tied_mask(batch.input)
        masked_batch = batch._replace(input=masked_input)

        # run shared FDV solver per component
        fdv_out, prior_cost_sum = self._run_fdv_per_component(batch, masked_batch, phase = phase)
        # multivariate fusion with UNet
        fused = self.fuser(fdv_out)

        # losses on fused output
        recon_loss = self.weighted_mse(fused - batch.tgt, self.rec_weight_coeff)
        grad_loss  = self.weighted_mse(kfilts.sobel(fused) - kfilts.sobel(batch.tgt), self.rec_weight_coeff)
        # print(prior_cost_sum)
        # print(recon_loss)
        # print(grad_loss)
        if self.solver.n_step > 0:
            self.log(f"{phase}_prior_cost", prior_cost_sum, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{phase}_gloss", grad_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("sampling_rate", mean_sr, on_step=False, on_epoch=True)
        
        with torch.no_grad():
            self.log(f"{phase}_mse", 10000 * recon_loss  * self._std_scale2(recon_loss), prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{phase}_loss", recon_loss, prog_bar=True, on_step=False, on_epoch=True)

        # compose training objective (same weights you used)
        training_loss = 500 * recon_loss + 10 * prior_cost_sum + 1000 * grad_loss
        return training_loss, fused

    def base_step(self, batch, phase=""):
        # Not used directly anymore; kept for compatibility
        out = self(batch=batch)
        loss = self.weighted_mse(err=out - batch.tgt, weight=self.rec_weight)
        with torch.no_grad():
            self.log(f"{phase}_mse",  loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{phase}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss, out

    def forward(self, batch):
        """
        Inference path: FDV per component -> stack -> U-Net fuse.
        Expects batch.input shaped (B, TC, H, W) when called directly.
        """
        # forward() may be called from parent code; ensure correct shape
        B, T, C, H, W = self._shape_info(batch.input)
        fdv_out, _ = self._run_fdv_per_component(batch, batch)  # no extra masking here
        fused = self.fuser(fdv_out)
        return fused

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.test_data = []

        # (B, TC, H, W)
        batch = batch._replace(
            input=batch.input.view(batch.input.size(0), -1, batch.input.size(-2), batch.input.size(-1)),
            tgt=  batch.tgt.view(  batch.tgt.size(0),   -1,   batch.tgt.size(-2),   batch.tgt.size(-1))
        )
        raw_input = batch.input.clone()

        # mask
        masked_input, _ = self._apply_spatially_tied_mask(batch.input)
        masked_batch = batch._replace(input=masked_input)

        # per-component FDV + fusion
        fdv_out, _ = self._run_fdv_per_component(batch, masked_batch)
        fused = self.fuser(fdv_out)

        # denorm to (B, T, C, H, W)
        if self.norm_type == "z_score":
            raw_input_den = self._denormalize_zscore(raw_input,      self.norm_stats)
            masked_den    = self._denormalize_zscore(masked_batch.input, self.norm_stats)
            tgt_den       = self._denormalize_zscore(batch.tgt,      self.norm_stats)
            out_den       = self._denormalize_zscore(fused,          self.norm_stats)
        else:
            raw_input_den = self._denormalize_minmax(raw_input,      self.norm_stats)
            masked_den    = self._denormalize_minmax(masked_batch.input, self.norm_stats)
            tgt_den       = self._denormalize_minmax(batch.tgt,      self.norm_stats)
            out_den       = self._denormalize_minmax(fused,          self.norm_stats)

        self.test_data.append(torch.stack(
            [raw_input_den.cpu(), masked_den.cpu(), tgt_den.cpu(), out_den.detach().cpu()],
            dim=1,
        ))

    @property
    def test_quantities(self):
        # keep your ordering & names
        return ['input', 'inp', 'tgt', 'out']

    def on_test_epoch_end(self):
        print(self.test_data[0].shape)
        rec_da = self.trainer.test_dataloaders.dataset.reconstruct(
            self.test_data, self.rec_weight_coeff.cpu().numpy()
        )
        if isinstance(rec_da, list):
            rec_da = rec_da[0]

        self.test_data = rec_da.assign_coords(dict(v0=self.test_quantities)).to_dataset(dim='v0')

        metric_data = self.test_data.pipe(self.pre_metric_fn)
        metrics = pd.Series({metric_n: metric_fn(metric_data) for metric_n, metric_fn in self.metrics.items()})

        print(metrics.to_frame(name="Metrics").to_markdown())
        if self.logger:
            self.test_data.to_netcdf(Path(self.logger.log_dir) / 'test_data.nc')
            print(Path(self.trainer.log_dir) / 'test_data.nc')
            if hasattr(self.logger, "log_metrics"):
                self.logger.log_metrics(metrics.to_dict())

       
# class Lit4dVarNet_Unet(pl.LightningModule):
#     def __init__(self, fdv_cfg, unet_cfg, *args, **kwargs):
#         super().__init__()
#         self.save_hyperparameters(ignore=['fdv_cfg', 'unet_cfg'])

#         self.fdv_model = fdv_cfg
#         self.unet = unet_cfg
#         self.rec_weight = self.fdv_model.rec_weight
#         self.norm_stats = self.fdv_model.norm_stats
#         self.sampling_rate = self.fdv_model.sampling_rate
#         self.norm_type = self.fdv_model.norm_type
#         self.metrics = self.fdv_model.metrics
#         self.pre_metric_fn = self.fdv_model.pre_metric_fn
   
#     def setup(self, stage=None):
#         if self.trainer is None:
#             return 

#     def forward(self, batch):
#         B, TC, H, W = batch.input.shape
#         C = len(self.norm_stats[0])
#         T = TC // C

#         out_comps = []
#         for c in range(C):
#             # Extract component c: shape (B, T, H, W)
#             comp_input = batch.input[:, c::C, :, :]

#             # Build a batch with just 1 component and pass to fdv_model
#             sub_batch = batch._replace(
#                 input=comp_input,
#                 tgt=batch.tgt[:, c::C, :, :] if batch.tgt is not None else None,
#             )

#             comp_output = self.fdv_model(batch=sub_batch)  # (B, T, 1, H, W)
#             comp_output = comp_output.mean(dim=1)          # (B, 1, H, W)
#             out_comps.append(comp_output)

#         # Concatenate components: (B, C, H, W)
#         fused = torch.cat(out_comps, dim=1)

#         # Apply U-Net: (B, C_out, H, W)
#         output = self.unet(fused)

#         return output

#     def weighted_mse(self, err, weight):
#         err_w = err * weight[None, ...]
#         non_zeros = (torch.ones_like(err) * weight[None, ...]) == 0.0
#         err_num = err.isfinite() & ~non_zeros
#         if err_num.sum() == 0:
#             return torch.scalar_tensor(1000.0, device=err.device).requires_grad_()
#         loss = F.mse_loss(err_w[err_num], torch.zeros_like(err_w[err_num]))
#         return loss

#     def base_step(self, batch, phase=""):
#         out = self(batch=batch)
#         loss = self.weighted_mse(out - batch.tgt, self.rec_weight)
#         with torch.no_grad():
#             self.log(f"{phase}_mse",  loss, prog_bar=True, on_step=False, on_epoch=True)
#             self.log(f"{phase}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
#         return loss, out

#     def step(self, batch, phase=""):
#         if self.training and batch.tgt.isfinite().float().mean() < 0.9:
#             return None, None

#         B, TC, H, W = batch.input.shape
#         C = len(self.norm_stats[0])
#         T = TC // C

#         batch = batch._replace(
#             input=batch.input.view(B, -1, H, W),
#             tgt=batch.tgt.view(B, -1, H, W)
#         )

#         masked_input = batch.input.clone()
#         for i in range(B):
#             sr = self.sampling_rate
#             if isinstance(sr, (list, tuple)) and len(sr) == 2:
#                 sr = random.uniform(sr[0], sr[1])
#             spatial_mask = (torch.rand((H, W), device=batch.input.device) > sr)
#             mask_3d = spatial_mask.unsqueeze(0).expand(C, -1, -1).repeat_interleave(T, dim=0)
#             masked_input[i][mask_3d] = float("nan")

#         batch = batch._replace(input=masked_input)

#         loss, out = self.base_step(batch, phase)
#         return loss, out

#     def training_step(self, batch, batch_idx):
#         loss, _ = self.step(batch, "train")
#         return loss

#     def validation_step(self, batch, batch_idx):
#         loss, _ = self.step(batch, "val")
#         return loss

#     def test_step(self, batch, batch_idx):
#         _, out = self.step(batch, "test")
#         return out

#     @property
#     def test_quantities(self):
#         return ['input', 'inp', 'tgt', 'out']

#     def on_test_epoch_end(self):
#         rec_da = self.trainer.test_dataloaders.dataset.reconstruct(
#             self.test_data, self.rec_weight.cpu().numpy()
#         )
#         if isinstance(rec_da, list):
#             rec_da = rec_da[0]
#         self.test_data = rec_da.assign_coords(dict(v0=self.test_quantities)).to_dataset(dim='v0')
#         metric_data = self.test_data.pipe(self.pre_metric_fn)
#         metrics = pd.Series({
#             metric_n: metric_fn(metric_data)
#             for metric_n, metric_fn in self.metrics.items()
#         })

#         print(metrics.to_frame(name="Metrics").to_markdown())
#         if self.logger:
#             self.test_data.to_netcdf(Path(self.logger.log_dir) / 'test_data.nc')
#             self.logger.log_metrics(metrics.to_dict())

# class FdvModelWrapper(nn.Module):
#     def __init__(
#         self,
#         solver,
#         rec_weight,
#         norm_stats,
#         sampling_rate=1.0,
#         norm_type="z_score",
#     ):
#         super().__init__()
#         self.solver = solver
#         self.rec_weight = rec_weight
#         self.norm_stats = norm_stats
#         self.sampling_rate = sampling_rate
#         self.norm_type = norm_type

#     def forward(self, batch):
#         """
#         Forward pass for a single PCA component: input shape (B, T, 1, H, W).
#         Output: same shape (B, T, 1, H, W)
#         """
#         B, T, C, H, W = batch.input.shape
#         assert C == 1, "FdvModelWrapper expects a single PCA component per call."

#         # Flatten to (B, T*C, H, W)
#         batch = batch._replace(
#             input=batch.input.view(B, -1, H, W),
#             tgt=batch.tgt.view(B, -1, H, W) if batch.tgt is not None else None
#         )

#         # Apply NaN mask
#         masked_input = batch.input.clone()
#         for i in range(B):
#             sr = self.sampling_rate
#             if isinstance(sr, (list, tuple, ListConfig)) and len(sr) == 2:
#                 sr = random.uniform(sr[0], sr[1])
#             spatial_mask = (torch.rand((H, W), device=masked_input.device) > sr)
#             mask_3d = spatial_mask.unsqueeze(0).expand(C, -1, -1).repeat_interleave(T, dim=0)
#             masked_input[i][mask_3d] = float("nan")
#         batch = batch._replace(input=masked_input)

#         # Run 4DVar solver
#         with torch.no_grad():  # remove if you want gradients
#             out = self.solver(batch)

#         # Reshape to (B, T, C, H, W)
#         return out.view(B, T, C, H, W)