# Evaluation: reproducing the paper tables

The scripts read the paper data: `results_paper/` (in this repository) and
`estimates_paper/` (the separated audio, regenerated with `generate_estimates.py`).
Set other locations with `RESULTS_PAPER` and `ESTIMATES_PAPER`.

    uv run python evaluation/reproduce_tables.py
        # tab:discovery and tab:separation from the stored per-item results
        # (SEG labels are the recorded ones); checks every cell
    uv run python evaluation/reproduce_separation_from_audio.py {snr,mert,judge,table}
        # tab:separation recomputed from the separated audio (hours; Judge runs in
        # the SAM-Audio environment, see the script's docstring)

`mert_mse.py` is the MERT-MSE metric used by the second script.

Regenerating the model outputs themselves (not needed to reproduce the tables):

    uv run python evaluation/generate_estimates.py {seg,oracle_dit,oracle_loco,mus3d_dit,mus3d_loco} --out <dir>
    uv run python evaluation/build_release.py --release <dir>

writes a new `results_paper/` + `estimates_paper/` tree that the two scripts above
can score (point `RESULTS_PAPER` / `ESTIMATES_PAPER` at it).
