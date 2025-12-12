#!/bin/bash
#SBATCH --partition=Odyssey # Selected partition
#SBATCH --job-name=unet_t # Name for the job
#SBATCH --gres=gpu:a100:1 # Resources asked
#SBATCH --cpus-per-gpu=12    # 12 CPUs for each GPU
#SBATCH --mem=102G
#SBATCH --output=job/job_%j.log # %j for jobid

export HOME='/Odyssey/private/ochapron/'
source /Odyssey/private/ochapron/start_conda.sh
conda activate fdv
srun python main.py xp='fdv_lazy_CTS' +params='direct_inversion_unet'
