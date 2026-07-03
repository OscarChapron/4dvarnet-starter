# Paper Workflow Experiments

These workflows are derived from the two supplied papers:

- `2604.02850v1.pdf`: depth-aware probabilistic 3D ocean reconstruction from sparse SSH/SST.
- `preprint_4dvarnet_fm_2025 (36).pdf`: physics-informed neural ensemble DA via conditional flow matching.

This repository exposes 4DVarNet-style depth, direct-inversion, UNet-solver Hydra configs, and a conditional-flow-matching training overlay at `+params=conditional_flow_matching`. The Slurm workflow still runs reproducible analogues by default; use the overlay to train the CFM baseline directly.

## Workflow A: Depth-Aware Reconstruction

Goal: test whether explicit depth and time conditioning help reconstruction under sparse observations, following the depth-aware DDPM paper.

| Label | Hydra setup | Paper motivation |
| --- | --- | --- |
| `p1_depth_zaware_c1` | `xp=fdv_lazy_CTS_z_val +params=direct_inversion_unet_z_val`, one component, depth/time enabled | Baseline depth-aware conditioning. |
| `p1_depth_no_z_c1` | Same, `prior_cost_unet_depth.ignore_depth=true` | Ablate depth identifier. |
| `p1_depth_no_t_c1` | Same, `prior_cost_unet_depth.ignore_time=true` | Ablate time/step embedding. |
| `p1_depth_dropout_c1` | Same, `prior_cost_unet_depth.dropout=0.1` | Robustness/regularization under sparsity. |
| `p1_depth_zaware_c8` | Same, `datamodule.xrds_kw.patch_dims.component=8` | Multi-component/depth-like scaling. |

## Workflow B: 4DVarNet-FM Solver Analogues

Goal: test solver depth, initialization/background, and UNet residual choices, following the 4DVarNet-FM paper's focus on unrolled neural DA, Gaussian-vs-non-Gaussian structure, and one-step vs multi-step flow-style operators.

| Label | Hydra setup | Paper motivation |
| --- | --- | --- |
| `p2_unroll_k1` | `xp=fdv_lazy_CTS`, `model.solver.n_step=1` | One-step operator, analogous to vanilla CFM with `K=1`. |
| `p2_unroll_k10` | `xp=fdv_lazy_CTS`, `model.solver.n_step=10` | Default unrolled 4DVarNet-style solver. |
| `p2_unroll_k20` | `xp=fdv_lazy_CTS`, `model.solver.n_step=20` | Deeper unrolling/scaling test. |
| `p2_unroll_k10_zero_init` | `xp=fdv_lazy_CTS`, `model.solver.init_mode=zeros` | Background/initialization ablation. |
| `p2_unet_residual` | `xp=fdv_lazy_CTS_Unet` | UNet residual update analogue. |
| `p2_unetsolver_unetout` | `xp=fdv_lazy_CTS_UnetSolver_UnetOut` | UNet solver plus UNet output variant. |

## Running

Smoke validation is the default:

```bash
sbatch sript_slurm/paper_workflow_experiments.sbatch
```

Full runs use the config defaults unless you set overrides:

```bash
RUN_MODE=full sbatch sript_slurm/paper_workflow_experiments.sbatch
```

Run only one workflow group or one label:

```bash
EXPERIMENT_FILTER=paper1_depth sbatch sript_slurm/paper_workflow_experiments.sbatch
EXPERIMENT_FILTER=p2_unroll_k10 RUN_MODE=full sbatch sript_slurm/paper_workflow_experiments.sbatch
```

Useful optional overrides:

```bash
MAX_EPOCHS=150 LIMIT_TRAIN_BATCHES=100 LIMIT_VAL_BATCHES=100 RUN_MODE=full sbatch sript_slurm/paper_workflow_experiments.sbatch
PROJECT_DIR=/Odyssey/private/ochapron/4dvarnet-starter sbatch sript_slurm/paper_workflow_experiments.sbatch
```

Run the new conditional-flow-matching training overlay directly with:

```bash
python main.py xp=fdv_lazy_CTS_z_val +params=conditional_flow_matching
```

Outputs are written under `outputs/paper_workflows/<job-id>_<timestamp>/`, with one log per experiment in `logs/`.
