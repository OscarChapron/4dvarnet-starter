#!/bin/bash
#SBATCH --partition=Odyssey # Selected partition
#SBATCH --job-name=fdv_unet++ # Name for the job
#SBATCH --gres=gpu:a100:1 # Resources asked
#SBATCH --cpus-per-gpu=12    # 12 CPUs for each GPU
#SBATCH --mem=102G
#SBATCH --output=job/job_%j.log # %j for jobid

export HOME='/Odyssey/private/ochapron/'
source /Odyssey/private/ochapron/start_conda.sh
conda activate fdv
srun python main.py xp='fdv_lazy_CTS_UnetSolver_UnetOut' 
# srun python main.py xp='fdv_lazy_CTS_season' +params='direct_inversion_unet'
