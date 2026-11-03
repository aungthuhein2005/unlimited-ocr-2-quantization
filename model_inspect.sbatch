#!/bin/bash
#SBATCH --job-name=inspect_ocr
#SBATCH --output=logs/inspect_%j.out
#SBATCH --error=logs/inspect_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
# inspect_model.py runs on CPU, so no GPU is requested. Add --partition=<name> if
# your cluster's default partition isn't suitable (list them with: sinfo -s).
# Submit from the project root (logs/ comes with the repo).

set -eo pipefail

# Ignore packages in ~/.local so only the llm env's own packages are used
export PYTHONNOUSERSITE=1

source /cm/shared/apps/miniconda3/etc/profile.d/conda.sh
conda activate llm
# Enabled after activation: conda's activate scripts can reference unset variables
set -u

cd "$SLURM_SUBMIT_DIR"

# Compute nodes often have no internet: download the model on the login node first,
# then uncomment these. Point HF_HOME at scratch if your home quota is small.
# export HF_HOME=/path/to/scratch/hf_cache
# export HF_HUB_OFFLINE=1

python scripts/inspect_model.py
