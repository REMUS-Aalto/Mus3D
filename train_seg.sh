#!/bin/bash
#SBATCH --job-name=seg
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=1-23:59:00
#SBATCH --signal=TERM@600
#SBATCH --output=seg-%j.log
#
# Train the stem embedding generator (SEG): 12 x 384 recursive flow-matching DiT,
# batch 8, 400k iterations in the paper.
#
#   bash train_seg.sh                                   # SEG
#   MODEL=SEG_resemb bash train_seg.sh                  # SEG (with res. embedding)
#   sbatch train_seg.sh trainer.batch_size=4            # any hydra override
#   bash train_seg.sh trainer.ckpt_dir=$OUT/ckpt/latest # resume
#
# Evaluate `ckpt/latest`, not `ckpt/best` (best-ckpt tracking is off).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}"

MODEL=${MODEL:-SEG}
RUN=${RUN:-seg-${MODEL}}
OUT=${OUT:-runs/${RUN}}

export MALLOC_ARENA_MAX=2
export HYDRA_FULL_ERROR=1
export PYTHONPATH=src

mkdir -p "${OUT}"
exec uv run --no-sync python -u src/train_seg.py \
    model="${MODEL}" \
    trainer.output_dir="${OUT}" \
    trainer.logger.run_name="${RUN}" \
    "$@"
