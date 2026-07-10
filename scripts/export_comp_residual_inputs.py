#!/usr/bin/env python
"""Export 4DVarNet reconstruction residuals for Comp_residual.

Expected input is a `test_data.nc` produced by the Lightning test hook. The
script writes a NetCDF with:
  - xb: 4DVarNet background/reconstruction
  - target: reference SSP
  - residual: target - xb
  - obs: optional sparse input if present
"""

import argparse
from pathlib import Path

import xarray as xr


def export_residuals(input_path, output_path, background_var="out", target_var="tgt"):
    ds = xr.open_dataset(input_path)
    missing = [name for name in (background_var, target_var) if name not in ds]
    if missing:
        raise KeyError(f"Missing required variables {missing} in {input_path}")

    out = xr.Dataset(
        {
            "xb": ds[background_var],
            "input": ds[background_var],
            "target": ds[target_var],
            "tgt": ds[target_var],
            "residual": ds[target_var] - ds[background_var],
        }
    )
    for obs_name in ("inp", "input"):
        if obs_name in ds:
            out["obs"] = ds[obs_name]
            break

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Path to 4DVarNet test_data.nc")
    parser.add_argument("--output", "-o", default=None, help="Output NetCDF path")
    parser.add_argument("--background-var", default="out", help="Background/reconstruction variable name")
    parser.add_argument("--target-var", default="tgt", help="Target/reference variable name")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = args.output or input_path.with_name("comp_residual_inputs.nc")
    written = export_residuals(input_path, output_path, args.background_var, args.target_var)
    print(written)


if __name__ == "__main__":
    main()
