#!/bin/bash
# MuS3D inference on audio files: SEG discovers the stems, a query-conditioned
# separator extracts them. Any extra flags go to src/inference.py (see --help).
#
#   bash inference.sh song.wav --out output/inference
#   bash inference.sh song.wav --separator bslocoformer --level track
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH=src
exec uv run --no-sync python -u src/inference.py "$@"
