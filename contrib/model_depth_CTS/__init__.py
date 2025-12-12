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
torch.set_float32_matmul_precision('high')
import pytorch_lightning as pl
import torch
from torch import Tensor
from math import inf

class PCA(torch.nn.Module):
    """
    Principal Component Analysis in PyTorch
    ---------------------------------------
    Parameters
    ----------
    n_components : int | float | None
        * int   – keep exactly this many principal directions
        * float – keep enough directions to explain this fraction
                  of the total variance (e.g. 0.95 for 95 %)
        * None  – keep the full basis (acts as an orthogonal whitening)
    whiten : bool (default False)
        If True, scales each principal component to unit variance.
    device : torch.device or str (default='cpu')
    """

    def __init__(self, n_components=None, whiten=False, device="cpu"):
        super().__init__()
        self.n_components = n_components
        self.whiten = whiten
        self.device = torch.device(device)
        # learned attributes
        self.mean_ = None
        self.components_ = None
        self.explained_var_ = None
        self.explained_var_ratio_ = None
        self.noise_var_ = None
        self.whitening_scales_ = None

    @torch.no_grad()
    def fit(self, X: torch.Tensor):
        """
        Learns the PCA basis from a (n_samples × n_features) tensor.
        """
        X = X.to(self.device, dtype=torch.float32)
        n, p = X.shape

        self.mean_ = X.mean(dim=0, keepdim=True)
        Xc = X - self.mean_

        U, S, Vt = torch.linalg.svd(Xc, full_matrices=False)

        eigvals = (S ** 2) / (n - 1)

        if self.n_components is None:
            k = p
        elif isinstance(self.n_components, int):
            k = min(self.n_components, p)
        elif isinstance(self.n_components, float):
            cumvar = torch.cumsum(eigvals, 0) / eigvals.sum()
            k = int(torch.searchsorted(cumvar, self.n_components).item()) + 1
        else:
            raise ValueError("n_components must be int, float, or None")

        self.components_ = Vt[:k]                     # (k × p)
        self.explained_var_ = eigvals[:k]             # (k,)
        self.explained_var_ratio_ = (
            self.explained_var_ / eigvals.sum()
        )                                             # (k,)

        if self.whiten:
            self.whitening_scales_ = torch.sqrt(self.explained_var_)
        else:
            self.whitening_scales_ = torch.ones_like(self.explained_var_)
        if k < p:
            self.noise_var_ = eigvals[k:].mean()
        else:
            self.noise_var_ = torch.tensor(0.0, device=self.device)

        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Projects data into the principal-component space.
        """
        Xc = (X.to(self.device) - self.mean_)
        X_pca = (Xc @ self.components_.T) / self.whitening_scales_
        return X_pca

    def inverse_transform(self, X_pca: torch.Tensor) -> torch.Tensor:
        """
        Maps samples back to the original feature space.
        """
        Xc = X_pca * self.whitening_scales_
        return Xc @ self.components_ + self.mean_

    # handy property
    @property
    def n_features_(self):
        return None if self.components_ is None else self.components_.shape[1]
    

class EMPCA(torch.nn.Module):
    r"""
    Probabilistic PCA trained with the EM algorithm (Tipping & Bishop, 1999).

    Model:
        x = mu + W z + ε,     z ~ 𝓝(0, I_k),     ε ~ 𝓝(0, σ² I_d)

    Attributes learned after `fit`
    ------------------------------
    mean_                – (1 × d) data mean μ
    components_          – (k × d) orthonormal principal directions
    W_                   – (d × k) loading matrix W (not orthonormal)
    sigma2_              – isotropic noise variance σ²
    explained_var_       – k eigen-values λ_j of the sample covariance
    explained_var_ratio_ – λ_j / Σ_i λ_i
    """

    def __init__(
        self,
        n_components: int,
        max_iters: int = 1000,
        tol: float = 1e-6,
        device: str | torch.device = "cpu",
        verbose: bool = False,
    ):
        super().__init__()
        self.k = n_components
        self.max_iters = max_iters
        self.tol = tol
        self.device = torch.device(device)
        self.verbose = verbose

        # learned parameters
        self.mean_ = None
        self.components_ = None
        self.W_ = None
        self.sigma2_ = None
        self.explained_var_ = None
        self.explained_var_ratio_ = None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _init_params(self, d: int):
        """
        Gaussian-like random initialisation of W and σ².
        """
        W = torch.randn(d, self.k, device=self.device)
        sigma2 = torch.tensor(1.0, device=self.device)
        return W, sigma2

    # ------------------------------------------------------------------ #
    # EM algorithm
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def fit(self, X: Tensor):
        print('I am in the fit function')
        X = X.to(self.device, dtype=torch.float32)
        n, d = X.shape

        # centre data
        self.mean_ = X.mean(0, keepdim=True)
        Xc = X - self.mean_

        # -- initial parameters ------------------------------------------------
        W, sigma2 = self._init_params(d)

        I_k = torch.eye(self.k, device=self.device)

        # pre-compute sample covariance norm for convergence check
        sq_norm_X = (Xc**2).sum().item()

        last_ll = -inf
        for it in range(self.max_iters):
            # ---------- E-step ----------------------------------------------
            M = W.T @ W + sigma2 * I_k              # (k × k)
            M_inv = torch.linalg.inv(M)             # (k × k)

            Ez = (M_inv @ W.T @ Xc.T)               # (k × n)
            Ezz = n * sigma2 * M_inv + Ez @ Ez.T    # (k × k)

            # ---------- M-step ----------------------------------------------
            W_new = (Xc.T @ Ez.T) @ torch.linalg.inv(Ezz)   # (d × k)

            term1 = sq_norm_X
            term2 = -2.0 * torch.sum(Ez.T * (Xc @ W_new))   # trace trick
            term3 = torch.sum(Ezz * (W_new.T @ W_new))
            sigma2_new = (term1 + term2 + term3) / (d * n)

            # ---------- likelihood & convergence ---------------------------
            # log-likelihood for monitoring (optional, cheap)
            C = W_new @ W_new.T + sigma2_new * torch.eye(d, device=self.device)
            sign, logdet = torch.slogdet(C)
            ll = -0.5 * n * (d * torch.log(torch.tensor(2 * torch.pi, device=self.device)) + logdet
                             + (torch.linalg.solve(C, Xc.T) * Xc.T).sum() / n)

            if self.verbose:
                print(f"iter {it:3d}:  log-lik = {ll.item():.4f}   σ² = {sigma2_new.item():.4f}")

            if abs(ll - last_ll) < self.tol:
                if self.verbose:
                    print("Converged.")
                break

            W, sigma2, last_ll = W_new, sigma2_new, ll

        # --------------------------------------------------------------------
        # store learned parameters
        self.W_ = W
        self.sigma2_ = sigma2

        # orthonormalise to get standard PCA directions & eigenvalues
        # (columns of W span the same sub-space but aren't unit/orthogonal)
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)  # W = U Σ Vᵀ
        self.components_ = Vt                                # (k × d)
        self.explained_var_ = S**2                           # λ_j
        self.explained_var_ratio_ = self.explained_var_ / self.explained_var_.sum()

        return self

    # ------------------------------------------------------------------ #
    # Transform / inverse
    # ------------------------------------------------------------------ #
    def transform(self, X: Tensor) -> Tensor:
        Xc = X.to(self.device) - self.mean_
        # posterior mean of z given x
        M_inv = torch.linalg.inv(self.W_.T @ self.W_ + self.sigma2_ * torch.eye(self.k, device=self.device))
        Ez = (M_inv @ self.W_.T @ Xc.T).T
        return Ez  # (n × k)

    def inverse_transform(self, Z: Tensor) -> Tensor:
        return Z.to(self.device) @ self.W_.T + self.mean_

class PCALightning(pl.LightningModule):
    """
    Thin Lightning wrapper around (E)M-PCA.
    Nothing is optimised; `Trainer` is used only to run the data loop once.
    """
    def __init__(self, n_components: int = 8, variant: str = "em", **kwargs):
        super().__init__()
        if variant.lower() == "em":
            self.pca = EMPCA(n_components=n_components, **kwargs)
        elif variant.lower() == "svd":
            self.pca = PCA(n_components=n_components, **kwargs)
        else:
            raise ValueError("variant must be 'em' or 'svd'")

        self._has_fitted = False          # guard so we fit only *once*

    def training_step(self, batch, _):
        """Collect the whole dataset once, then fit in `on_train_epoch_end`."""
        x, *_ = batch                      # assume (x, y) or (x,)
        x = x.flatten(1)                  # (B, d)
        self.log("dummy_loss", 0.0)       # Lightning wants a tensor
        return {"x": x}

    def on_train_epoch_end(self, outputs):
        if self._has_fitted:
            return
        X = torch.cat([o["x"] for o in outputs], dim=0)  # (N, d)
        self.pca.fit(X)                                  # ← real work
        self._has_fitted = True
        self.print("✓ PCA fitted. Explained variance ratio:",
                   self.pca.explained_var_ratio_.detach().cpu().numpy())

    # ---- Lightning boiler-plate, not actually used ------------
    def configure_optimizers(self):
        return []        # no optimiser, because we don’t train by SGD