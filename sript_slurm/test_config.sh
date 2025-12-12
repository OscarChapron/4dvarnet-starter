#!/bin/bash
#SBATCH --partition=Odyssey
#SBATCH --job-name=fdv_lazy_z_test
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --output=job/job_test_%j.log
#SBATCH --time=00:30:00

# ==========================================================
# ==== ENVIRONMENT SETUP ====
# ==========================================================
export HOME="/Odyssey/private/ochapron"
source /Odyssey/private/ochapron/start_conda.sh
conda activate fdv

echo "Testing fdv_lazy_CTS_z configuration..."

# Test with single experiment configuration
srun python main.py \
  xp="fdv_lazy_CTS_z" \
  +params="direct_inversion_unet_z" \
  logger.name="test_config" \
  datamodule.xrds_kw.patch_dims.component=1 \
  prior_cost_unet_depth.ignore_time=false \
  prior_cost_unet_depth.ignore_depth=false \
  prior_cost_unet_depth.dropout=0.0 \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=2 \
  trainer.limit_val_batches=1 \
  trainer.fast_dev_run=true

TEST_EXIT_CODE=$?
if [ $TEST_EXIT_CODE -eq 0 ]; then
  echo "Configuration test PASSED! The script should work correctly."
else
  echo "Configuration test FAILED with exit code ${TEST_EXIT_CODE}"
  echo "Please check the configuration before running the full experiment script."
fi