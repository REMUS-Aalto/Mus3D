#!/bin/bash
#
# Build the MedleyDB-derived multitrack dataset at
# $DATASET_ROOT/medleydb-stems
# (clean/ and bleeding/ at the top level; no train/test split).
#
# All of MedleyDB minus Classical, Musical Theatre and the MusicDelta_Chinese*
# excerpts (MUSDB25 overlap is no longer excluded -- MUSDB25 is not used).
# Requires dataset/medleydb_songs.csv -- run list_medleydb_songs.py first.
#
# CPU/IO only, no GPU needed. Resumable: re-running skips songs already
# present in songs.json / skipped.json.
#
# Usage (from anywhere):
#   bash dataset/run_prepare_medleydb_stems.sh
#   bash dataset/run_prepare_medleydb_stems.sh --verify

set -euo pipefail

cd "$(dirname "$0")/.."

uv run --no-sync python dataset/prepare_medleydb_stems.py "$@"
