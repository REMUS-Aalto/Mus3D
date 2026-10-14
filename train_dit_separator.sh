#!/bin/bash
#SBATCH --job-name=dit-separator
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=1-23:59:00
#SBATCH --signal=TERM@600
#SBATCH --output=dit-separator-%j.log
#
# Fine-tune the query-conditioned DiT separator from SAM-Audio large (2.27B,
# cross-attention removed, query + level via adaLN modulation). Batch 16.
#
# The weights of facebook/sam-audio-large are downloaded (Hugging Face access
# required) and converted to this backbone when the model is built. A local
# checkpoint.pt can be given with INIT=/path/to/checkpoint.pt.
#
#   sbatch train_dit_separator.sh
#   sbatch train_dit_separator.sh trainer.ckpt_dir=$OUT/ckpt/latest   # resume
#
# The model is mono: the DAC-VAE has no stereo mode.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}"

RUN=${RUN:-dit-separator}
OUT=${OUT:-runs/${RUN}}
INIT=${INIT:-facebook/sam-audio-large}

export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=131072
export HYDRA_FULL_ERROR=1
export PYTHONPATH=src

mkdir -p "${OUT}"
exec uv run --no-sync python -u src/train_dit_separator.py \
    trainer.output_dir="${OUT}" \
    trainer.logger.run_name="${RUN}" \
    model.backbone.init_from="${INIT}" \
    "$@"
