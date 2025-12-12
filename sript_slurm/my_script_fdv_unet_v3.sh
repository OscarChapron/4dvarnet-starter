#!/bin/bash
#SBATCH --partition=Odyssey
#SBATCH --job-name=fdv_lazy_z_exps
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=250G
#SBATCH --output=job/job_%j.log

# ==========================================================
# ==== ENVIRONMENT SETUP ====
# ==========================================================
export HOME="/Odyssey/private/ochapron"
source /Odyssey/private/ochapron/start_conda.sh
conda activate fdv

# ==========================================================
# ==== COMMON CONFIG ====
# ==========================================================
XP_NAME="fdv_lazy_CTS_z_val"
PARAMS_BASE="direct_inversion_unet_z_val"
BASE_OUTDIR="/Odyssey/private/ochapron/4dvarnet-starter/outputs"

# ==========================================================
# ==== EXPERIMENT MATRIX ====
# ==========================================================
EXPERIMENTS=(
  "false false 0.0 1"
  "true  false 0.0 1"
  "false true  0.0 1"
  "true  true  0.1 1"
  "false false 0.1 1"
  "false false 0.0 8"
  "true  false 0.0 8"
  "false true  0.0 8"
)

# ==========================================================
# ==== LOOP OVER EXPERIMENTS ====
# ==========================================================
for EXP in "${EXPERIMENTS[@]}"; do
  read IGNORE_TIME IGNORE_DEPTH DROPOUT NCOMP <<< "${EXP}"

  echo "====================================================="
  echo "Running: ignore_time=${IGNORE_TIME}, ignore_depth=${IGNORE_DEPTH}, dropout=${DROPOUT}, component=${NCOMP}"
  echo "====================================================="

  TAG="it_${IGNORE_TIME}_id_${IGNORE_DEPTH}_do_${DROPOUT}_c${NCOMP}"
  LOGGER_NAME="fdv_lazy_CTS_z_${TAG}"

  # --- Test configuration first ---
  echo "Testing configuration..."
  srun python main.py \
    xp="${XP_NAME}" \
    +params="${PARAMS_BASE}" \
    +logger.name="${LOGGER_NAME}_test" \
    datamodule.xrds_kw.patch_dims.component=${NCOMP} \
    prior_cost_unet_depth.ignore_time=${IGNORE_TIME} \
    prior_cost_unet_depth.ignore_depth=${IGNORE_DEPTH} \
    prior_cost_unet_depth.dropout=${DROPOUT} \
    trainer.max_epochs=1 \
    trainer.limit_train_batches=1 \
    trainer.limit_val_batches=1 \

  TEST_EXIT_CODE=$?
  if [ $TEST_EXIT_CODE -ne 0 ]; then
    echo "[ERROR] Configuration test failed with exit code ${TEST_EXIT_CODE} for ${LOGGER_NAME}"
    continue
  fi
  
  # --- Train phase using base config with parameter overrides ---
  echo "Starting training with parameters:"
  echo "  Config: ${XP_NAME}"
  echo "  Params: ${PARAMS_BASE}"
  echo "  Logger: ${LOGGER_NAME}"
  echo "  Component: ${NCOMP}"
  echo "  Ignore Time: ${IGNORE_TIME}"
  echo "  Ignore Depth: ${IGNORE_DEPTH}"
  echo "  Dropout: ${DROPOUT}"
  
  srun python main.py \
    xp="${XP_NAME}" \
    +params="${PARAMS_BASE}" \
    +logger.name="${LOGGER_NAME}" \
    datamodule.xrds_kw.patch_dims.component=${NCOMP} \
    prior_cost_unet_depth.ignore_time=${IGNORE_TIME} \
    prior_cost_unet_depth.ignore_depth=${IGNORE_DEPTH} \
    prior_cost_unet_depth.dropout=${DROPOUT} \
    trainer.max_epochs=600

  # Check if training was successful
  TRAIN_EXIT_CODE=$?
  if [ $TRAIN_EXIT_CODE -eq 137 ]; then
    echo "[ERROR] Training killed (likely OOM) for ${LOGGER_NAME}. Consider reducing batch size or model size."
    continue
  elif [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "[ERROR] Training failed with exit code ${TRAIN_EXIT_CODE} for ${LOGGER_NAME}"
    continue
  fi

  # --- Find best checkpoint automatically ---
  echo "Looking for checkpoints in: ${BASE_OUTDIR}/*/${LOGGER_NAME}/checkpoints/"
  CKPT_PATH=$(find ${BASE_OUTDIR} -type f -path "*/${LOGGER_NAME}/checkpoints/val_mse=*.ckpt" | head -n 1)

  if [ -n "$CKPT_PATH" ] && [ -f "$CKPT_PATH" ]; then
    # Use base config for forecast too
    echo "-----------------------------------------------------"
    echo "[FORECAST] Running forecast for ${LOGGER_NAME}"
    echo "           Checkpoint: ${CKPT_PATH}"
    echo "-----------------------------------------------------"
    
    srun python main.py \
      xp="${XP_NAME}" \
      +params="${PARAMS_BASE}" \
      ckpt="${CKPT_PATH}" \
      trainer.max_epochs=0 \
      trainer.limit_test_batches=1.0 \
      +logger.name="${LOGGER_NAME}_forecast"
  else
    echo "[WARNING] No valid checkpoint found for ${LOGGER_NAME}, skipping forecast."
    echo "          Searched in: ${BASE_OUTDIR}/*/${LOGGER_NAME}/checkpoints/"
  fi
done

# ==========================================================
# ==== FINAL RUN: UNetSolver + UNetOut VARIANT ====
# ==========================================================
echo "====================================================="
echo "Running final experiment: fdv_lazy_CTS_UnetSolver"
echo "====================================================="

srun python main.py xp='fdv_lazy_CTS_UnetSolver_UnetOut'
srun python main.py xp='fdv_lazy_CTS_Unet'

echo "====================================================="
echo "All experiments completed!"
echo "====================================================="
echo "Results can be found in: ${BASE_OUTDIR}"
echo "Experiment logs and checkpoints are organized by logger name."
