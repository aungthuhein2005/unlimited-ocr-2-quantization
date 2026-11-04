#!/bin/bash
#SBATCH --job-name=ocr_bf16
#SBATCH --output=logs/ocr_%j.out
#SBATCH --error=logs/ocr_%j.err
#SBATCH --partition=defq
#SBATCH --gres=gpu:3g.71gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
# Usage (from the project root): sbatch run_ocr.sbatch [--images data/*.png] [--output results/bf16.json]
# The GPU node is split into MIG slices (sinfo -o "%P %G"): 1g.18gb, 1g.35gb, 3g.71gb, 7g.141gb.
# Keep the same slice for every run so bf16 vs quantized timings are comparable.

set -eo pipefail

# Ignore packages in ~/.local so only the llm env's own packages are used
export PYTHONNOUSERSITE=1

source /cm/shared/apps/miniconda3/etc/profile.d/conda.sh
conda activate llm
# Enabled after activation: conda's activate scripts can reference unset variables
set -u

cd "$SLURM_SUBMIT_DIR"

# Log GPU and driver
nvidia-smi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"

python scripts/run_ocr.py "$@"
