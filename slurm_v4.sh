#!/bin/bash
#SBATCH --job-name=lnn-v4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=longq7-eng
set -euo pipefail

REPO_DIR="/mnt/onefs/home/bneeluri2024/ondemand/LNN/repo_inspect"
VENV_DIR="/mnt/beegfs/home/bneeluri2024/lnn-gpu-venv"

mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"

source /etc/profile.d/lmod.sh
module purge
module load python3/3.9.16
module load cuda/12.4.0

# If your cluster registers the GPU type as xa100 instead of a100,
# change the directive above manually before submission.

# Reuse an existing environment if available. Replace this with a shared
# persistent venv path if /tmp is not visible on compute nodes.
source "${VENV_DIR}/bin/activate"

echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Host: $(hostname)"
echo "Working dir: $(pwd)"
echo "Python: $(which python)"
echo "Torch check:"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
PY

python -u train_solar_v4.py
