import numpy as np
import hydra
from omegaconf import OmegaConf
from pathlib import Path
import functools as ft
import metpy.calc as mpcalc
import kornia
import pandas as pd
import xrft
import torch
import pyinterp
import pyinterp.fill
import pyinterp.backends.xarray
import src.data
import xarray as xr
import matplotlib.pyplot as plt


def pipe(inp, fns):
    for f in fns:
        inp = f(inp)
    return inp

def kwgetattr(obj, name):
    return getattr(obj, name)

def callmap(inp, fns):
    return [fn(inp) for fn in fns]

def half_lr_adam(lit_mod, lr):
    return torch.optim.Adam(
        [
            {"params": lit_mod.solver.grad_mod.parameters(), "lr": lr},
            {"params": lit_mod.solver.obs_cost.parameters(), "lr": lr},
            {"params": lit_mod.solver.prior_cost.parameters(), "lr": lr / 2},
        ],
    )

def cosanneal_lr_adam_AE(lit_mod, lr, T_max=100, weight_decay=0.):
    opt = torch.optim.Adam(
        [
            {"params": lit_mod.prior_cost.parameters(), "lr": lr },
        ], weight_decay=weight_decay
    )
    return {
        "optimizer": opt,
        "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max),
    }


def cosanneal_lr_adam(lit_mod, lr, T_max=100, weight_decay=0.):
    opt = torch.optim.Adam(
        [
            {"params": lit_mod.solver.grad_mod.parameters(), "lr": lr},
            {"params": lit_mod.solver.obs_cost.parameters(), "lr": lr},
            {"params": lit_mod.solver.prior_cost.parameters(), "lr": lr / 2},
        ], weight_decay=weight_decay
    )
    return {
        "optimizer": opt,
        "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max),
    }

def cosanneal_lr_lion(lit_mod, lr, T_max=100):
    import lion_pytorch
    opt = lion_pytorch.Lion(
        [
            {"params": lit_mod.solver.grad_mod.parameters(), "lr": lr},
            {"params": lit_mod.solver.prior_cost.parameters(), "lr": lr / 2},
        ], weight_decay=1e-3
    )
    return {
        "optimizer": opt,
        "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max),
    }


def triang_lr_adam(lit_mod, lr_min=5e-5, lr_max=3e-3, nsteps=200):
    opt = torch.optim.Adam(
        [
            {"params": lit_mod.solver.grad_mod.parameters(), "lr": lr_max},
            {"params": lit_mod.solver.prior_cost.parameters(), "lr": lr_max / 2},
        ],
    )

    return {
        "optimizer": opt,
        "lr_scheduler": torch.optim.lr_scheduler.CyclicLR(
            opt,
            base_lr=lr_min,
            max_lr=lr_max,
            step_size_up=nsteps,
            step_size_down=nsteps,
            gamma=0.95,
            cycle_momentum=False,
            mode="exp_range",
        ),
    }

def remove_nan(da):
    da["lon"] = da.lon.assign_attrs(units="degrees_east")
    da["lat"] = da.lat.assign_attrs(units="degrees_north")
    
    da.transpose("lon", "lat", "time")[:, :] = pyinterp.fill.gauss_seidel(
         pyinterp.backends.xarray.Grid3D(da)
    )[1]
    return da

def mask(da, sampling_rate = 0.1):
    time_dim = da.time.size
    lat_dim = da.lat.size
    lon_dim = da.lon.size

    random_mask = np.random.choice([0, 1], size=(time_dim, lat_dim, lon_dim), p=[1 - sampling_rate, sampling_rate])
    mask_data_array = xr.DataArray(random_mask, dims=['time', 'lat', 'lon'])
    masked_data_array = da.where(mask_data_array == 1, other=np.nan)
    return masked_data_array

def threshold_xarray(da):
    threshold = 10**3
    da = xr.where(da > threshold, 1, da)
    return da

def get_constant_crop(patch_dims, crop, dim_order=["time", "lat", "lon"]):
        # Check if depth is present in the data
    if "z" in patch_dims:
        dim_order = ["time", "z", "lat", "lon"]

    patch_weight = np.zeros([patch_dims[d] for d in dim_order], dtype="float32")
    mask = tuple(
        slice(crop[d], -crop[d]) if crop.get(d, 0) > 0 else slice(None, None)
        for d in dim_order
    )
    patch_weight[mask] = 1.0
    return patch_weight

def get_cropped_hanning_mask(patch_dims, crop, **kwargs):
    pw = get_constant_crop(patch_dims, crop)

    t_msk = kornia.filters.get_hanning_kernel1d(patch_dims["time"])

    patch_weight = t_msk[:, None, None] * pw
    return patch_weight.cpu().numpy()


def get_triang_time_wei(patch_dims, offset=0, **crop_kw):
    pw = get_constant_crop(patch_dims, **crop_kw)
    return np.fromfunction(
        lambda t, *a: (
            (1 - np.abs(offset + 2 * t - patch_dims["time"]) / patch_dims["time"]) * pw
        ),
        patch_dims.values(),
    )

def load_enatl(*args, obs_from_tgt=False, **kwargs):
    ssh = xr.open_zarr('../sla-data-registry/enatl_preproc/truth_SLA_SSH_NATL60.zarr/').ssh
    nadirs = xr.open_zarr('../sla-data-registry/enatl_preproc/SLA_SSH_5nadirs.zarr/').ssh
    ssh = ssh.interp(
        lon=np.arange(ssh.lon.min(), ssh.lon.max(), 1/20),
        lat=np.arange(ssh.lat.min(), ssh.lat.max(), 1/20)
    )
    nadirs = nadirs.interp(time=ssh.time, method='nearest').interp(lat=ssh.lat, lon=ssh.lon, method='nearest')
    ds =  xr.Dataset(dict(input=nadirs, tgt=(ssh.dims, ssh.values)), nadirs.coords)

    if obs_from_tgt:
        ds = ds.assign(input=ds.tgt.transpose(*ds.input.dims).where(np.isfinite(ds.input), np.nan))
    return ds.transpose('time', 'lat', 'lon').to_array().load()

#def load_enatl_ecs(*args, obs_from_tgt=False, **kwargs):
#    ssh = xr.open_dataset('/DATASET/eNATL/eNATL60_BLB002_cutoff_freq_0_1000m_regrid.nc').ecs
#    nadirs = xr.open_dataset('/DATASET/eNATL/eNATL60_BLB002_cutoff_freq_0_1000m_regrid.nc').ecs
#    ssh = ssh.interp(
#        lon=np.arange(ssh.lon.min(), ssh.lon.max(), 1/20),
#        lat=np.arange(ssh.lat.min(), ssh.lat.max(), 1/20)
#    )
#    nadirs = nadirs.interp(time=ssh.time, method='nearest').interp(lat=ssh.lat, lon=ssh.lon, method='nearest')
#    ds =  xr.Dataset(dict(input=nadirs, tgt=(ssh.dims, ssh.values)), nadirs.coords)
#
#    if obs_from_tgt:
#        ds = ds.assign(input=ds.tgt.transpose(*ds.input.dims).where(np.isfinite(ds.input), np.nan))
#    return ds.transpose('time', 'lat', 'lon').to_array().load()

def load_altimetry_data(path, obs_from_tgt=False):
    ds =  (
        xr.open_dataset(path)
        # .assign(ssh=lambda ds: ds.ssh.coarsen(lon=2, lat=2).mean().interp(lat=ds.lat, lon=ds.lon))
        .load()
        .assign(
            input=lambda ds: ds.nadir_obs,
            tgt=lambda ds: remove_nan(ds.ssh),
        )    
    )

    if obs_from_tgt:
        ds = ds.assign(input=ds.tgt.where(np.isfinite(ds.input), np.nan))
    
    return (
        ds[[*src.data.TrainingItem._fields]]
        .transpose('time', 'lat', 'lon')
        .to_array()
    )

def load_celerity_data(path, obs_from_tgt=False):
    ds =  (
        xr.open_dataset(path)
        # .assign(ssh=lambda ds: ds.ssh.coarsen(lon=2, lat=2).mean().interp(lat=ds.lat, lon=ds.lon))
        .load()
        .assign(input=lambda ds: ds.celerity,
                tgt=lambda ds: ds.celerity.fillna(0))    
    )
    return (
        ds[[*src.data.TrainingItem._fields]]
        #.transpose("time", "z", "y", "x")
        .transpose("time", "z", "lat", "lon")
        .to_array()
    )

def load_cutoff_freq(path, obs_from_tgt=False):
    ds =  (
        xr.open_dataset(path)
        # .assign(ssh=lambda ds: ds.ssh.coarsen(lon=2, lat=2).mean().interp(lat=ds.lat, lon=ds.lon))
        .load()
        .assign(input=lambda ds: (threshold_xarray(ds.ecs)),
                tgt=lambda ds: remove_nan(threshold_xarray(ds.ecs)))   
        # .assign(input=lambda ds: mask(threshold_xarray(ds.cutoff_freq)),
        #         tgt=lambda ds: remove_nan(threshold_xarray(ds.cutoff_freq)))   
    )
    da = ds[[*src.data.TrainingItem._fields]].transpose("time", "lat", "lon").to_array()
    return (
        ds[[*src.data.TrainingItem._fields]]
        .transpose("time", "lat", "lon")
        .to_array()
    )

#def load_cutoff_freq_mask(path, obs_from_tgt=False):
#    ds =  (
#        xr.open_dataset(path)
#        # .assign(ssh=lambda ds: ds.ssh.coarsen(lon=2, lat=2).mean().interp(lat=ds.lat, lon=ds.lon))
#        .load()
#        # .assign(input=lambda ds: (threshold_xarray(ds.ecs)),
#        #         tgt=lambda ds: remove_nan(threshold_xarray(ds.ecs)))   
#        .assign(input=lambda ds: mask(threshold_xarray(ds.cutoff_freq)),
#                tgt=lambda ds: remove_nan(threshold_xarray(ds.cutoff_freq)))   
#    )
#    da = ds[[*src.data.TrainingItem._fields]].transpose("time", "lat", "lon").to_array()
#    return (
#        ds[[*src.data.TrainingItem._fields]]
#        .transpose("time", "lat", "lon")
#        .to_array()
#    )

def load_full_natl_data(
        path_obs="../sla-data-registry/CalData/cal_data_new_errs.nc",
        path_gt="../sla-data-registry/NATL60/NATL/ref_new/NATL60-CJM165_NATL_ssh_y2013.1y.nc",
        obs_var='five_nadirs',
        gt_var='ssh',
        **kwargs
    ):
    inp = xr.open_dataset(path_obs)[obs_var]
    gt = (
        xr.open_dataset(path_gt)[gt_var]
        .isel(time=slice(0, -1))
        .sel(lat=inp.lat, lon=inp.lon, method="nearest")
    )

    return xr.Dataset(dict(input=inp, tgt=(gt.dims, gt.values)), inp.coords).to_array().sortby('variable')


def rmse_based_scores_from_ds(ds, ref_variable='tgt', study_variable='out'):
    #mask = ~np.isnan(ds['input'])
    try:
        return rmse_based_scores(ds[ref_variable], ds[study_variable])[2:]
    except:
        return [np.nan, np.nan]

def psd_based_scores_from_ds(ds, ref_variable='tgt', study_variable='out'):
    print(ds)
    try:
        return psd_based_scores(ds[ref_variable], ds[study_variable])[1:]
    except:
        return [np.nan, np.nan]

def rmse_based_scores(da_rec, da_ref, mask = None):
    if mask is not None:
        # Apply the mask to da_rec and da_ref
        da_rec = da_rec.where(mask)
        da_ref = da_ref.where(mask)
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
        1.0 - (((da_rec - da_ref) ** 2).mean()) ** 0.5 / (((da_ref) ** 2).mean()) ** 0.5
    )
    return (
        rmse_t,
        rmse_xy,
        np.round(leaderboard_rmse.values, 5).item(),
        np.round(reconstruction_error_stability_metric, 5).item(),
    )


def psd_based_scores(da_rec, da_ref, mask = None):
    if mask is not None:
        # Apply the mask to da_rec and da_ref
        da_rec = da_rec.where(mask)
        da_ref = da_ref.where(mask)
    print('hello')
    #print(da_rec)
    #print(da_ref)
    err = da_rec - da_ref
    #print(err)
    err["time"] = (err.time - err.time[0]) / np.timedelta64(1, "D")
    signal = da_ref
    signal["time"] = (signal.time - signal.time[0]) / np.timedelta64(1, "D")
    psd_err = xrft.power_spectrum(
        err, dim=["time", "lon"], detrend="constant", window="hann"
    ).compute()
    psd_signal = xrft.power_spectrum(
        signal, dim=["time", "lon"], detrend="constant", window="hann"
    ).compute()
    # print(psd_signal)
    # psd_signal.to_netcdf('/DATASET/envs/oscar/4dvarnet-starter/outputs/psd_base_signal.nc')
    # psd_err.to_netcdf('/DATASET/envs/oscar/4dvarnet-starter/outputs/psd_base_err.nc')

    mean_psd_signal = psd_signal.mean(dim="lat").where(
        (psd_signal.freq_lon > 0.0) & (psd_signal.freq_time > 0), drop=True
    )
    
    mean_psd_err = psd_err.mean(dim="lat").where(
        (psd_err.freq_lon > 0.0) & (psd_err.freq_time > 0), drop=True
    )
    print(mean_psd_err)
    psd_based_score = 1.0 - mean_psd_err / mean_psd_signal
    level = [0.5]
    cs = plt.contour(
        1.0 / psd_based_score.freq_lon.values,
        1.0 / psd_based_score.freq_time.values,
        psd_based_score,
        level,
    )
    x05, y05 = cs.collections[0].get_paths()[0].vertices.T
    plt.close()

    shortest_spatial_wavelength_resolved = np.min(x05)
    shortest_temporal_wavelength_resolved = np.min(y05)
    psd_da = 1.0 - mean_psd_err / mean_psd_signal
    psd_da.name = "psd_score"
    return (
        psd_da.to_dataset(),
        np.round(shortest_spatial_wavelength_resolved, 3).item(),
        np.round(shortest_temporal_wavelength_resolved, 3).item(),
    )


def diagnostics(lit_mod, test_domain):
    test_data = lit_mod.test_data.sel(test_domain)
    return diagnostics_from_ds(test_data, test_domain)


def diagnostics_from_ds(test_data, test_domain):
    test_data = test_data.sel(test_domain)
    metrics = {
        "RMSE (m)": test_data.pipe(lambda ds: (ds.out - ds.tgt))
        .pipe(lambda da: da**2)
        .mean()
        .pipe(np.sqrt)
        .item(),
        **dict(
            zip(
                ["λx", "λt"],
                test_data.pipe(lambda ds: psd_based_scores(ds.out, ds.tgt)[1:]),
            )
        ),
        **dict(
            zip(
                ["μ", "σ"],
                test_data.pipe(lambda ds: rmse_based_scores(ds.out, ds.tgt)[2:]),
            )
        ),
    }
    return pd.Series(metrics, name="osse_metrics")


def test_osse(trainer, lit_mod, osse_dm, osse_test_domain, ckpt, diag_data_dir=None):
    lit_mod.norm_stats = osse_dm.norm_stats()
    trainer.test(lit_mod, datamodule=osse_dm, ckpt_path=ckpt)
    osse_tdat = lit_mod.test_data[['out', 'ssh']]
    osse_metrics = diagnostics_from_ds(
        osse_tdat, test_domain=osse_test_domain
    )

    print(osse_metrics.to_markdown())

    if diag_data_dir is not None:
        osse_metrics.to_csv(diag_data_dir / "osse_metrics.csv")
        if (diag_data_dir / "osse_test_data.nc").exists():
            xr.open_dataset(diag_data_dir / "osse_test_data.nc").close()
        osse_tdat.to_netcdf(diag_data_dir / "osse_test_data.nc")

    return osse_metrics


def ensemble_metrics(trainer, lit_mod, ckpt_list, dm, save_path):
    metrics = []
    test_data = xr.Dataset()
    for i, ckpt in enumerate(ckpt_list):
        trainer.test(lit_mod, ckpt_path=ckpt, datamodule=dm)
        rmse = (
            lit_mod.test_data.pipe(lambda ds: (ds.out - ds.ssh))
            .pipe(lambda da: da**2)
            .mean()
            .pipe(np.sqrt)
            .item()
        )
        lx, lt = psd_based_scores(lit_mod.test_data.out, lit_mod.test_data.ssh)[1:]
        mu, sig = rmse_based_scores(lit_mod.test_data.out, lit_mod.test_data.ssh)[2:]

        metrics.append(dict(ckpt=ckpt, rmse=rmse, lx=lx, lt=lt, mu=mu, sig=sig))

        if i == 0:
            test_data = lit_mod.test_data
            test_data = test_data.rename(out=f"out_{i}")
        else:
            test_data = test_data.assign(**{f"out_{i}": lit_mod.test_data.out})
        test_data[f"out_{i}"] = test_data[f"out_{i}"].assign_attrs(
            ckpt=str(ckpt)
        )

    metric_df = pd.DataFrame(metrics)
    print(metric_df.to_markdown())
    print(metric_df.describe().to_markdown())
    metric_df.to_csv(save_path + "/metrics.csv")
    test_data.to_netcdf(save_path + "ens_out.nc")


def add_geo_attrs(da):
    da["lon"] = da.lon.assign_attrs(units="degrees_east")
    da["lat"] = da.lat.assign_attrs(units="degrees_north")
    return da


def vort(da):
    return mpcalc.vorticity(
        *mpcalc.geostrophic_wind(
            da.pipe(add_geo_attrs).assign_attrs(units="m").metpy.quantify()
        )
    ).metpy.dequantify()


def geo_energy(da):
    return np.hypot(*mpcalc.geostrophic_wind(da.pipe(add_geo_attrs))).metpy.dequantify()


def best_ckpt(xp_dir):
    _, xpn = load_cfg(xp_dir)
    print(Path(xp_dir) / xpn / 'checkpoints')
    ckpt_last = max(
        (Path(xp_dir) / xpn / 'checkpoints').glob("*.ckpt"), key=lambda p: p.stat().st_mtime
    )
    cbs = torch.load(ckpt_last)["callbacks"]
    ckpt_cb = cbs[next(k for k in cbs.keys() if "ModelCheckpoint" in k)]
    return ckpt_cb["best_model_path"]


def load_cfg(xp_dir):
    hydra_cfg = OmegaConf.load(Path(xp_dir) / ".hydra/hydra.yaml").hydra
    cfg = OmegaConf.load(Path(xp_dir) / ".hydra/config.yaml")
    OmegaConf.register_new_resolver(
        "hydra", lambda k: OmegaConf.select(hydra_cfg, k), replace=True
    )
    try:
        OmegaConf.resolve(cfg)
        OmegaConf.resolve(cfg)
    except Exception as e:
        return None, None

    return cfg, OmegaConf.select(hydra_cfg, "runtime.choices.xp")


