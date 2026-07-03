#!/bin/bash
#SBATCH --partition=Odyssey
#SBATCH --job-name=paper_workflows
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=250G
#SBATCH --output=job/paper_workflows_%j.log
#SBATCH --time=72:00:00

set -uo pipefail

# ==========================================================
# ==== ENVIRONMENT SETUP ====
# ==========================================================
export HOME="/Odyssey/private/ochapron"
if [ -f /Odyssey/private/ochapron/start_conda.sh ]; then
  source /Odyssey/private/ochapron/start_conda.sh
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda is not available; cannot activate the fdv environment." >&2
  exit 127
fi
conda activate fdv

PROJECT_DIR="${PROJECT_DIR:-/Odyssey/private/ochapron/4dvarnet-starter}"
if [ ! -d "${PROJECT_DIR}" ]; then
  PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
fi
cd "${PROJECT_DIR}" || exit 2

mkdir -p job

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_GPU:-12}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_GPU:-12}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${SLURM_CPUS_PER_GPU:-12}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${SLURM_CPUS_PER_GPU:-12}}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128,garbage_collection_threshold:0.8}"

# ==========================================================
# ==== RUN CONTROLS ====
# ==========================================================
RUN_MODE="${RUN_MODE:-smoke}"
EXPERIMENT_FILTER="${EXPERIMENT_FILTER:-all}"
STAMP="$(date +%Y%m%d_%H%M%S)"
JOB_ID="${SLURM_JOB_ID:-local}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-${PROJECT_DIR}/outputs/paper_workflows/${JOB_ID}_${STAMP}}"
mkdir -p "${BASE_OUTPUT_DIR}/logs"

RUN_OVERRIDES=()
if [ "${RUN_MODE}" = "smoke" ]; then
  RUN_OVERRIDES+=("trainer.max_epochs=${SMOKE_EPOCHS:-1}")
  RUN_OVERRIDES+=("++trainer.limit_train_batches=${SMOKE_TRAIN_BATCHES:-2}")
  RUN_OVERRIDES+=("++trainer.limit_val_batches=${SMOKE_VAL_BATCHES:-1}")
elif [ "${RUN_MODE}" = "full" ]; then
  if [ -n "${MAX_EPOCHS:-}" ]; then
    RUN_OVERRIDES+=("trainer.max_epochs=${MAX_EPOCHS}")
  fi
  if [ -n "${LIMIT_TRAIN_BATCHES:-}" ]; then
    RUN_OVERRIDES+=("++trainer.limit_train_batches=${LIMIT_TRAIN_BATCHES}")
  fi
  if [ -n "${LIMIT_VAL_BATCHES:-}" ]; then
    RUN_OVERRIDES+=("++trainer.limit_val_batches=${LIMIT_VAL_BATCHES}")
  fi
else
  echo "[ERROR] RUN_MODE must be 'smoke' or 'full'. Got: ${RUN_MODE}" >&2
  exit 2
fi

echo "====================================================="
echo "Paper workflow experiments"
echo "Project dir: ${PROJECT_DIR}"
echo "Run mode: ${RUN_MODE}"
echo "Experiment filter: ${EXPERIMENT_FILTER}"
echo "Output dir: ${BASE_OUTPUT_DIR}"
echo "====================================================="

FAILED=()
PASSED=()
SKIPPED=()

should_run() {
  local group="$1"
  local label="$2"
  [ "${EXPERIMENT_FILTER}" = "all" ] || [ "${EXPERIMENT_FILTER}" = "${group}" ] || [ "${EXPERIMENT_FILTER}" = "${label}" ]
}

run_experiment() {
  local group="$1"
  local label="$2"
  local xp_name="$3"
  local params_name="$4"
  shift 4
  local extra_overrides=("$@")

  if ! should_run "${group}" "${label}"; then
    SKIPPED+=("${label}")
    return 0
  fi

  local log_path="${BASE_OUTPUT_DIR}/logs/${label}.log"
  local cmd=(srun --ntasks=1 python main.py "xp=${xp_name}")
  if [ "${params_name}" != "-" ]; then
    cmd+=("+params=${params_name}")
  fi
  cmd+=("trainer.logger.save_dir=${BASE_OUTPUT_DIR}")
  cmd+=("trainer.logger.name=${label}")
  cmd+=("trainer.logger.version=")
  cmd+=("${RUN_OVERRIDES[@]}")
  cmd+=("${extra_overrides[@]}")

  echo "====================================================="
  echo "[RUN] ${label}"
  echo "[GROUP] ${group}"
  echo "[XP] ${xp_name}"
  echo "[PARAMS] ${params_name}"
  printf '[CMD]'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  echo "[LOG] ${log_path}"
  echo "====================================================="

  "${cmd[@]}" 2>&1 | tee "${log_path}"
  local status=${PIPESTATUS[0]}

  if [ ${status} -eq 0 ]; then
    PASSED+=("${label}")
    echo "[OK] ${label}"
  elif [ ${status} -eq 137 ]; then
    FAILED+=("${label}:exit137_oom_or_killed")
    echo "[ERROR] ${label} failed with exit 137, likely OOM or scheduler kill." >&2
  else
    FAILED+=("${label}:exit${status}")
    echo "[ERROR] ${label} failed with exit ${status}." >&2
  fi
}

# ==========================================================
# ==== WORKFLOW A: DEPTH-AWARE RECONSTRUCTION ====
# ==========================================================
run_experiment \
  "paper1_depth" \
  "p1_depth_zaware_c1" \
  "fdv_lazy_CTS_z_val" \
  "direct_inversion_unet_z_val" \
  "datamodule.xrds_kw.patch_dims.component=1" \
  "prior_cost_unet_depth.ignore_time=false" \
  "prior_cost_unet_depth.ignore_depth=false" \
  "prior_cost_unet_depth.dropout=0.0"

run_experiment \
  "paper1_depth" \
  "p1_depth_no_z_c1" \
  "fdv_lazy_CTS_z_val" \
  "direct_inversion_unet_z_val" \
  "datamodule.xrds_kw.patch_dims.component=1" \
  "prior_cost_unet_depth.ignore_time=false" \
  "prior_cost_unet_depth.ignore_depth=true" \
  "prior_cost_unet_depth.dropout=0.0"

run_experiment \
  "paper1_depth" \
  "p1_depth_no_t_c1" \
  "fdv_lazy_CTS_z_val" \
  "direct_inversion_unet_z_val" \
  "datamodule.xrds_kw.patch_dims.component=1" \
  "prior_cost_unet_depth.ignore_time=true" \
  "prior_cost_unet_depth.ignore_depth=false" \
  "prior_cost_unet_depth.dropout=0.0"

run_experiment \
  "paper1_depth" \
  "p1_depth_dropout_c1" \
  "fdv_lazy_CTS_z_val" \
  "direct_inversion_unet_z_val" \
  "datamodule.xrds_kw.patch_dims.component=1" \
  "prior_cost_unet_depth.ignore_time=false" \
  "prior_cost_unet_depth.ignore_depth=false" \
  "prior_cost_unet_depth.dropout=0.1"

run_experiment \
  "paper1_depth" \
  "p1_depth_zaware_c8" \
  "fdv_lazy_CTS_z_val" \
  "direct_inversion_unet_z_val" \
  "datamodule.xrds_kw.patch_dims.component=8" \
  "prior_cost_unet_depth.ignore_time=false" \
  "prior_cost_unet_depth.ignore_depth=false" \
  "prior_cost_unet_depth.dropout=0.0"

# ==========================================================
# ==== WORKFLOW B: 4DVARNET-FM SOLVER ANALOGUES ====
# ==========================================================
run_experiment \
  "paper2_fm_proxy" \
  "p2_unroll_k1" \
  "fdv_lazy_CTS" \
  "-" \
  "model.solver.n_step=1" \
  "model.solver.init_mode=OI"

run_experiment \
  "paper2_fm_proxy" \
  "p2_unroll_k10" \
  "fdv_lazy_CTS" \
  "-" \
  "model.solver.n_step=10" \
  "model.solver.init_mode=OI"

run_experiment \
  "paper2_fm_proxy" \
  "p2_unroll_k20" \
  "fdv_lazy_CTS" \
  "-" \
  "model.solver.n_step=20" \
  "model.solver.init_mode=OI"

run_experiment \
  "paper2_fm_proxy" \
  "p2_unroll_k10_zero_init" \
  "fdv_lazy_CTS" \
  "-" \
  "model.solver.n_step=10" \
  "model.solver.init_mode=zeros"

run_experiment \
  "paper2_fm_proxy" \
  "p2_unet_residual" \
  "fdv_lazy_CTS_Unet" \
  "-" \
  "model.solver.n_step=15" \
  "model.solver.init_mode=OI"

run_experiment \
  "paper2_fm_proxy" \
  "p2_unetsolver_unetout" \
  "fdv_lazy_CTS_UnetSolver_UnetOut" \
  "-" \
  "model.solver.n_step=15" \
  "model.solver.init_mode=OI"

echo "====================================================="
echo "Workflow summary"
echo "Passed (${#PASSED[@]}): ${PASSED[*]:-none}"
echo "Failed (${#FAILED[@]}): ${FAILED[*]:-none}"
echo "Skipped (${#SKIPPED[@]}): ${SKIPPED[*]:-none}"
echo "Output dir: ${BASE_OUTPUT_DIR}"
echo "====================================================="

if [ ${#FAILED[@]} -ne 0 ]; then
  exit 1
fi
