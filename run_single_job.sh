#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --job-name=sim
#SBATCH --array=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=10G
#SBATCH --time=30-00:00:00
#SBATCH --output=./out/CPparameter_%A.txt
#SBATCH --no-requeue

set -euo pipefail

#srun stdbuf -oL -eL "${HOME}/GPC_cleanup/build/SISO" 3.5 4 5 --seed "${SLURM_ARRAY_TASK_ID}"
srun stdbuf -oL -eL python3 "${HOME}/Chase-Pyndiah-decoding-demo/CP_parameter_opt.py" 4 5 3000 3.5 4.1 3.75
#python3 ${HOME}/GPC_cleanup/DRSD_OFEC_opt.py
#srun "${HOME}/GPC_cleanup/build/iBDDs" --seed "$SLURM_ARRAY_TASK_ID"