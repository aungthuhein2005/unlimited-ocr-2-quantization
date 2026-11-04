#!/bin/bash
#SBATCH --job-name=export_nf4
#SBATCH --output=logs/export_%j.out
#SBATCH --error=logs/export_%j.err
#SBATCH --partition=defq
#SBATCH --gres=gpu:3g.71gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
# Usage (from the project root): sbatch export_nf4.sbatch [--output ...] [--repo-id ...]
# Writes export/Unlimited-OCR-2-bnb-nf4 (weights, model code, model card) and test-loads it.

set -eo pipefail

# Ignore packages in ~/.local so only the llm env's own packages are used
export PYTHONNOUSERSITE=1

source /cm/shared/apps/miniconda3/etc/profile.d/conda.sh
conda activate llm
# Enabled after activation: conda's activate scripts can reference unset variables
set -u

cd "$SLURM_SUBMIT_DIR"

python scripts/export_nf4.py "$@"
