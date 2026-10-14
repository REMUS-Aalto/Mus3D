#!/bin/bash
#
# Prepare every dataset the paper uses from the raw corpora: the training and
# validation data, and the test set.
#
#   bash dataset/prepare_datasets.sh            # everything
#   bash dataset/prepare_datasets.sh train      # training/validation only
#   bash dataset/prepare_datasets.sh test       # test set only (MoisesDB is enough)
#
# The raw corpora are read from $DATASET_ROOT (default data/):
#
#   MoisesDB/moisesdb_v0.1/      MoisesDB v0.1
#   MedleyDB/all_v1v2/           MedleyDB v1 and v2 (train)
#   AlbumDB/stems_mixed/         AlbumDB (train)
#
# train writes, under $DATASET_ROOT,
#   moisesdb-stems/trimmed/      silence-trimmed MoisesDB tracks (substitution pool)
#   medleydb-stems/, albumdb-stems/   stems organized to the MoisesDB taxonomy
# the audio listed by the two files `dset_info` names in configs/data/*.yaml:
#   dataset/splits/moisesdb_medleydb_albumdb_train.json   279 songs
#   dataset/splits/moisesdb-val-no_bleed.json             26 songs
# (built with make_mss_dataset_info.py from the MoisesDB splits and these stems).
#
# test writes the test set (1065 segments of the 48 test songs, about 49 GB) to
# $MUS3D_TEST_SET (default test_set/), checked against the manifests in
# results_paper/.
#
# The MoisesDB splits are Banquet's (github.com/kwatcharasupat/query-bandit,
# reproducibility/splits.csv): folds 1-3 train, 4 validation, 5 test. They are
# fixed by dataset/splits/. Every step skips what is already written.
set -euo pipefail

WHAT=${1:-all}
JOBS=${JOBS:-16}
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT=$(realpath "${DATASET_ROOT:-data}")
PY="uv run --no-sync python"

need() {
    [ -d "$1" ] || { echo "missing $1 ($2)" >&2; exit 1; }
}
need "${ROOT}/MoisesDB/moisesdb_v0.1" "MoisesDB v0.1"

if [ "${WHAT}" = all ] || [ "${WHAT}" = train ]; then
    need "${ROOT}/MedleyDB/all_v1v2" "MedleyDB v1 + v2"
    need "${ROOT}/AlbumDB/stems_mixed" "AlbumDB"
    export DATASET_ROOT="${ROOT}"

    ${PY} dataset/trim_moisesdb_pool.py --root "${ROOT}" --jobs "${JOBS}"
    ${PY} dataset/prepare_medleydb_stems.py --workers "${JOBS}"
    ${PY} dataset/prepare_albumdb_stems.py
    # The two scripts above fill `trimmed/` with links to the untrimmed files;
    # the trimmed copies replace them.
    find "${ROOT}/albumdb-stems" -maxdepth 1 -name trimmed -type l -delete
    find "${ROOT}/medleydb-stems/trimmed" -type l -delete
    ${PY} dataset/trim_stems.py --root "${ROOT}" \
        --corpus medleydb-stems --corpus albumdb-stems --jobs "${JOBS}"

    echo "training data ready in ${ROOT}"
fi

if [ "${WHAT}" = all ] || [ "${WHAT}" = test ]; then
    TEST_SET=${MUS3D_TEST_SET:-test_set}
    ${PY} dataset/prepare_test_set.py --root "${ROOT}" --out "${TEST_SET}" --workers "${JOBS}"
    echo "test set ready in ${TEST_SET}"
fi
