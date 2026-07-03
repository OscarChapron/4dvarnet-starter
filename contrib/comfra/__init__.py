from copy import deepcopy
import functools as ft
import torch
import xarray as xr
import random
from typing import Sequence, Union
from omegaconf import ListConfig   # only for type hints – optional
import pytorch_lightning as pl
import torch.nn.functional as F
import kornia.filters as kfilts
from src.data import AugmentedDataset, BaseDataModule, TrainingItem, XrDataset

class BaseDataModule_comfra(pl.LightningDataModule):
    def __init__(self, input_da, domains, xrds_kw, dl_kw, aug_kw=None, norm_type = 'z_score', norm_stats=None, **kwargs):
        super().__init__()
        self.input_da = input_da
        self.domains = domains
        self.xrds_kw = xrds_kw
        self.dl_kw = dl_kw
        self.aug_kw = aug_kw if aug_kw is not None else {}
        self._norm_stats = norm_stats
        self.norm_type = norm_type

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self._post_fn = None

    def norm_stats(self):
        if self._norm_stats is None:
            if self.norm_type == 'z_score':
                self._norm_stats = self.train_mean_std()
                print("Norm stats", self._norm_stats)
            if self.norm_type == 'min_max':
                self._norm_stats = self.min_max_norm()
                print("Norm stats", self._norm_stats)
        return self._norm_stats

    def train_mean_std(self, variable='tgt'):
        train_data = self.input_da.sel(self.xrds_kw.get('domain_limits', {})).sel(self.domains['train'])
        return train_data.sel(variable=variable).pipe(lambda da: (da.mean().values.item(), da.std().values.item()))
    
    def min_max_norm(self, variable = 'tgt'):
        train_data = self.input_da.sel(self.xrds_kw.get('domain_limits', {})).sel(self.domains['train'])
        min_value = train_data.sel(variable=variable).min().values.item()
        max_value = train_data.sel(variable=variable).max().values.item()
        return min_value, max_value
    
    def post_fn(self):
        if self.norm_type == 'z_score':
            m, s = self.norm_stats()
            normalize = lambda item: (item - m) / s
        if self.norm_type == 'min_max':
            min_value, max_value = self.norm_stats()
            normalize = lambda item: (item - min_value) / (max_value - min_value)
        return ft.partial(ft.reduce,lambda i, f: f(i), [
            TrainingItem._make,
            lambda item: item._replace(tgt=normalize(item.tgt)),
            lambda item: item._replace(input=normalize(item.input)),
        ])

    def setup(self, stage='test'):
        post_fn = self.post_fn()

        if stage == 'fit':
            train_data = self.input_da.sel(self.domains['train'])

            # Drop the validation period from the training period
            try:
                train_data = train_data.drop_sel(time=xr.date_range(
                    self.domains['val']['time'].start,
                    self.domains['val']['time'].stop,
                ))
            except KeyError:
                raise KeyError(
                    'Validation period cannot be extracted from given training period!'
                )

            self.train_ds = XrDataset(
                train_data, **self.xrds_kw, postpro_fn=post_fn,
            )
            if self.aug_kw:
                self.train_ds = AugmentedDataset(self.train_ds, **self.aug_kw)

            self.val_ds = XrDataset(
                self.input_da.sel(self.domains['val']), **self.xrds_kw, postpro_fn=post_fn,
            )
        elif stage == 'test':
            self.test_ds = XrDataset(
                self.input_da.sel(self.domains['test']), **self.xrds_kw, postpro_fn=post_fn,
            )
        else:
            raise Exception(f'Unhandled stage {stage} in BaseDataModule.setup')


    def train_dataloader(self):
        return  torch.utils.data.DataLoader(self.train_ds, shuffle=True, **self.dl_kw)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, shuffle=False, **self.dl_kw)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, shuffle=False, **self.dl_kw)
