#!/bin/bash
#SBATCH --job-name=bslocoformer
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-23:59:00
#SBATCH --signal=TERM@600
#SBATCH --output=bslocoformer-%j.log
#
# Train the query-conditioned BS-Locoformer separator (medium), batch 4,
# ~200k iterations in the paper.
#
#   sbatch train_bslocoformer_separator.sh
#   sbatch train_bslocoformer_separator.sh trainer.ckpt_dir=$OUT/ckpt/latest   # resume
#   CONFIG=BSLocoformer_static sbatch train_bslocoformer_separator.sh          # static VDBGPO baseline
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}"

CONFIG=${CONFIG:-BSLocoformer_separator}
RUN=${RUN:-${CONFIG}}
OUT=${OUT:-runs/${RUN}}

export HYDRA_FULL_ERROR=1
export PYTHONPATH=src

mkdir -p "${OUT}"
exec uv run --no-sync python -u src/train_bslocoformer_separator.py --config "${CONFIG}" \
    trainer.output_dir="${OUT}" \
    trainer.logger.run_name="${RUN}" \
    "$@"
