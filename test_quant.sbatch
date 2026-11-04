#!/bin/bash
#SBATCH --job-name=ocr_quant_test
#SBATCH --output=logs/quant_%j.out
#SBATCH --error=logs/quant_%j.err
#SBATCH --partition=defq
#SBATCH --gres=gpu:3g.71gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
# Usage (from the project root): sbatch test_quant.sbatch scripts/int8_quantize.py
#                                sbatch test_quant.sbatch scripts/int4_quantize.py [--images ...] [--output ...]

set -eo pipefail

SCRIPT="${1:?Usage: sbatch test_quant.sbatch scripts/<script>.py [args]}"
shift

# Ignore packages in ~/.local so only the llm env's own packages are used
export PYTHONNOUSERSITE=1

source /cm/shared/apps/miniconda3/etc/profile.d/conda.sh
conda activate llm
# Enabled after activation: conda's activate scripts can reference unset variables
set -u

cd "$SLURM_SUBMIT_DIR"

python -c "import torch, bitsandbytes as bnb; print('torch', torch.__version__, 'bnb', bnb.__version__, 'gpu', torch.cuda.get_device_name())"

python "$SCRIPT" "$@"
