#!/bin/bash
#
# Build the AlbumDB-derived multitrack dataset at
# $DATASET_ROOT/albumdb-stems
# (no train/test split -- the whole album is one dataset dir).
#
# CPU/IO only, no GPU needed. Resumable: re-running skips songs already
# present in songs.json.
#
# Usage (from anywhere):
#   bash dataset/run_prepare_albumdb_stems.sh

set -euo pipefail

# `uv run` needs the project root, so this works regardless of the caller's cwd.
cd "$(dirname "$0")/.."

uv run --no-sync python dataset/prepare_albumdb_stems.py
