#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-fdv}"
if ! command -v conda >/dev/null 2>&1; then
  if [[ -f /Odyssey/private/ochapron/start_conda.sh ]]; then
    # shellcheck disable=SC1091
    source /Odyssey/private/ochapron/start_conda.sh
  elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  elif [[ -f "/usr/site/data/RED/pythonred2/miniforge3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "/usr/site/data/RED/pythonred2/miniforge3/etc/profile.d/conda.sh"
  elif [[ -x "/usr/site/data/RED/pythonred2/miniforge3/bin/conda" ]]; then
    export PATH="/usr/site/data/RED/pythonred2/miniforge3/bin:${PATH}"
  fi
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda is not available on PATH; cannot run the required '${CONDA_ENV}' environment." >&2
  exit 127
fi
PYTHON_CMD=(conda run --no-capture-output -n "${CONDA_ENV}" python)

export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128,garbage_collection_threshold:0.8}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export FDV_RECONSTRUCT_DTYPE="${FDV_RECONSTRUCT_DTYPE:-float16}"

DATA_ROOT="${DATA_ROOT:-/mnt/data/ochapron/data}"
TRAIN_CEL="${TRAIN_CEL:-${DATA_ROOT}/enatl60/celerity/eNATL60_BLB002_sound_speed_regrid_0_btm.nc}"
TEST_CEL="${TEST_CEL:-${DATA_ROOT}/natl60/celerity/NATL60GULF-CJM165_sound_speed_regrid_0_btm.nc}"

RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/outputs/oi_init_depth_sweep_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${RUN_ROOT}/logs"

SAMPLING_RATES="${SAMPLING_RATES:-0.001 0.005 0.01 0.05 0.1}"
MODELS="${MODELS:-4dvn_unet openai_unet_zt}"
MAX_EPOCHS="${MAX_EPOCHS:-150}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1.0}"
LIMIT_TEST_BATCHES="${LIMIT_TEST_BATCHES:-1.0}"
FIRST_N_COMPONENTS="${FIRST_N_COMPONENTS:-107}"
N_STEP_4DVN_UNET="${N_STEP_4DVN_UNET:-10}"
POSTPROCESS="${POSTPROCESS:-1}"
POSTPROCESS_EXTREMA_METHOD="${POSTPROCESS_EXTREMA_METHOD:-raw_filtered}"
MAX_PLOTS_TIMES="${MAX_PLOTS_TIMES:-0}"
MAX_PLOTS_DEPTHS="${MAX_PLOTS_DEPTHS:-0}"

echo "[CONFIG] conda env: ${CONDA_ENV}"
echo "[CONFIG] eNATL60 train: ${TRAIN_CEL}"
echo "[CONFIG] NATL60 test: ${TEST_CEL}"
echo "[CONFIG] sampling rates: ${SAMPLING_RATES}"
echo "[CONFIG] models: ${MODELS}"
echo "[CONFIG] output root: ${RUN_ROOT}"
echo "[CONFIG] postprocess raw/filtered F1 metrics: ${POSTPROCESS}"
echo "[CONFIG] postprocess extrema method: ${POSTPROCESS_EXTREMA_METHOD}"

common_overrides=(
  "entrypoints.1._target_=contrib.fdv_lazy.run_fit_test"
  "paths.inp_cel.train=${TRAIN_CEL}"
  "paths.inp_cel.test=${TEST_CEL}"
  "paths.tgt_cel.train=${TRAIN_CEL}"
  "paths.tgt_cel.test=${TEST_CEL}"
  "trainer.max_epochs=${MAX_EPOCHS}"
  "trainer.limit_train_batches=${LIMIT_TRAIN_BATCHES}"
  "++trainer.limit_val_batches=${LIMIT_VAL_BATCHES}"
  "+trainer.limit_test_batches=${LIMIT_TEST_BATCHES}"
  "+train_dm.input_da.first_n_component=${FIRST_N_COMPONENTS}"
  "+test_dm.input_da.first_n_component=${FIRST_N_COMPONENTS}"
)

run_one() {
  local model_name="$1"
  local sampling_rate="$2"
  local xp=""
  local params=""
  local model_overrides=()

  case "${model_name}" in
    4dvn_unet)
      xp="fdv_lazy_CTS_Unet"
      params="l40s_cpu50_depth_sweep"
      model_overrides=("model.solver.n_step=${N_STEP_4DVN_UNET}")
      ;;
    openai_unet_zt)
      xp="fdv_lazy_CTS_z"
      params="[direct_inversion_unet_z,l40s_cpu50_depth_sweep]"
      model_overrides=()
      ;;
    *)
      echo "[ERROR] Unknown model '${model_name}'. Use 4dvn_unet and/or openai_unet_zt." >&2
      return 2
      ;;
  esac

  local sr_label
  sr_label="$(LC_ALL=C printf "%g" "${sampling_rate}" | tr '.' 'p')"
  local label="${model_name}_sr${sr_label}"
  local log_path="${RUN_ROOT}/logs/${label}.log"

  echo "============================================================"
  echo "[RUN] ${label}"
  echo "[RUN] xp=${xp} params=${params}"
  echo "============================================================"

  "${PYTHON_CMD[@]}" main.py \
    "xp=${xp}" \
    "+params=${params}" \
    "model.sampling_rate=${sampling_rate}" \
    "trainer.logger.save_dir=${RUN_ROOT}" \
    "trainer.logger.name=${label}" \
    "${common_overrides[@]}" \
    "${model_overrides[@]}" \
    2>&1 | tee "${log_path}"

  local test_data="${RUN_ROOT}/${label}/test_data.nc"
  if [[ "${POSTPROCESS}" == "1" ]]; then
    if [[ -s "${test_data}" ]]; then
      local metrics_dir="${RUN_ROOT}/metrics/${label}"
      mkdir -p "${metrics_dir}"
      "${PYTHON_CMD[@]}" scripts/postprocess_three_inferences.py \
        --out-dir "${metrics_dir}" \
        --max-times "${MAX_PLOTS_TIMES}" \
        --max-depths "${MAX_PLOTS_DEPTHS}" \
        --extrema-method "${POSTPROCESS_EXTREMA_METHOD}" \
        --run "${label}=${test_data}" \
        2>&1 | tee "${RUN_ROOT}/logs/postprocess_${label}.log"
    else
      echo "[WARN] Missing ${test_data}; skipping postprocess for ${label}" >&2
    fi
  fi
}

for sampling_rate in ${SAMPLING_RATES}; do
  for model_name in ${MODELS}; do
    run_one "${model_name}" "${sampling_rate}"
  done
done

if [[ "${POSTPROCESS}" == "1" ]]; then
  "${PYTHON_CMD[@]}" - "${RUN_ROOT}" <<'PY'
from pathlib import Path
import sys
import pandas as pd

run_root = Path(sys.argv[1])
summary_dir = run_root / "metrics"
frames_by_name = {
    "summary_metrics.csv": [],
    "all_metrics_per_depth.csv": [],
    "profile_metrics.csv": [],
    "extrema_f1_metrics.csv": [],
}

for metrics_dir in sorted(p for p in summary_dir.glob("*") if p.is_dir()):
    label = metrics_dir.name
    for filename, frames in frames_by_name.items():
        path = metrics_dir / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "run_label", label)
        frames.append(frame)

for filename, frames in frames_by_name.items():
    if frames:
        out = summary_dir / f"sweep_{filename}"
        pd.concat(frames, ignore_index=True).to_csv(out, index=False)
        print(f"[INFO] Wrote {out}")

extrema_frames = frames_by_name["extrema_f1_metrics.csv"]
if extrema_frames:
    extrema = pd.concat(extrema_frames, ignore_index=True)
    keep = extrema[extrema["kind"].isin(["min", "max", "both", "combined_min_max"])]
    cols = [c for c in ["run_label", "model", "method", "kind", "f1", "precision", "recall", "tp", "fp", "fn", "gt_count", "pred_count"] if c in keep.columns]
    compact = keep[cols]
    compact.to_csv(summary_dir / "sweep_extrema_f1_raw_filtered_compact.csv", index=False)
    print(compact.to_markdown(index=False))
PY
fi

echo "[DONE] Sweep outputs are under ${RUN_ROOT}"
