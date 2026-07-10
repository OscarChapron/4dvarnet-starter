# SSP 4DVarNet + Comp_residual Workflow

This workflow uses 4DVarNet as the deterministic SSP reconstruction/backbone and
residual flow matching for probabilistic residual sampling.

It combines the two paper ideas used in this workspace:

- The depth-aware reconstruction paper motivates 8-level SSP/depth windows and depth conditioning.
- The 4DVarNet-FM preprint motivates training a residual conditional-expectation operator and sampling with the CFM ODE.

## 1. Train 4DVarNet on 8 contiguous depth levels

```bash
python main.py xp=fdv_lazy_CTS_z_val +params='[direct_inversion_unet_z_val,ssp_8depth_reconstruction]'
```

The overlay sets `datamodule.xrds_kw.patch_dims.component=8` with stride `1`, so
each batch has shape `(time, 8, lat, lon)`. The model flattens this to channels
internally, and the existing `comp_gloss` term penalizes vertical/depth-gradient
errors between adjacent SSP levels.

## 2. Test and export residual targets

After testing, export the deterministic background `xb`, target SSP, and residual:

```bash
python scripts/export_comp_residual_inputs.py /path/to/test_data.nc \
  --output /path/to/comp_residual_inputs.nc
```

The output contains:

- `xb`: 4DVarNet SSP reconstruction/background.
- `input`: alias of `xb` for this repository's datamodules.
- `target`: reference SSP.
- `tgt`: alias of `target` for this repository's datamodules.
- `residual`: `target - xb`, used as the residual target for Comp_residual.
- `obs`: sparse observations if present in `test_data.nc`.

## 3. Train residual CFM in this repository

Direct residual flow:

```bash
python main.py xp=fdv_lazy_CTS_z_val +params='[ssp_8depth_reconstruction,residual_flow_matching]' \
  train_dm.input_da.tgt_path=/path/to/train_comp_residual_inputs.nc \
  train_dm.input_da.inp_path=/path/to/train_comp_residual_inputs.nc \
  test_dm.input_da.tgt_path=/path/to/test_comp_residual_inputs.nc \
  test_dm.input_da.inp_path=/path/to/test_comp_residual_inputs.nc \
  model.method=flow
```

Two-residual method: deterministic mean residual plus stochastic anomaly flow:

```bash
python main.py xp=fdv_lazy_CTS_z_val +params='[ssp_8depth_reconstruction,residual_flow_matching]' \
  train_dm.input_da.tgt_path=/path/to/train_comp_residual_inputs.nc \
  train_dm.input_da.inp_path=/path/to/train_comp_residual_inputs.nc \
  test_dm.input_da.tgt_path=/path/to/test_comp_residual_inputs.nc \
  test_dm.input_da.inp_path=/path/to/test_comp_residual_inputs.nc \
  model.method=mean_flow
```

In both cases the datamodule reads `input=xb` and `tgt=target`. The learned
model predicts residual samples and returns:

```text
x_k = xb + r_k
```

## 4. Alternative: train Comp_residual on the cleanup branch

```bash
git clone --branch cleanup/docs-and-gitignore https://github.com/OscarChapron/Comp_residual.git
```

Use `/path/to/comp_residual_inputs.nc` as the residual dataset in Comp_residual.
The conditioning should include at least `xb` and can also include sparse
SSH/SST/SSP observations or 4DVarNet features if those are exported later.

At inference for arbitrary query depth `zq`:

```text
xb(t,zq,y,x) + r_k(t,zq,y,x) -> ensemble member k
```

The ensemble mean, variance, and quantiles are then computed over the sampled
residual members.
