"""Score tab:separation (Judge, SNR, MERT-MSE) from the audio in estimates_paper/.

Nothing is read from stored scores. Inputs are the population files under
`RESULTS_PAPER/populations/` and the audio under `ESTIMATES_PAPER/`; each
manifest row lists every system's wav, the mixture, the reference and the text
prompt the Judge is given (identical for every system on that item).

Four stages; `snr`, `mert` and `judge` write per-item csvs under --work
(default `RESULTS_PAPER/separation`, where `reproduce_tables.py` reads them),
`table` aggregates them:

    uv run python evaluation/reproduce_separation_from_audio.py snr
    uv run python evaluation/reproduce_separation_from_audio.py mert
    uv run python evaluation/reproduce_separation_from_audio.py judge   # SAM-Audio environment
    uv run python evaluation/reproduce_separation_from_audio.py table

The Judge stage needs facebook/sam-audio-judge and so runs in a SAM-Audio
environment, with `SAM_AUDIO_REPO` pointing at that checkout. Every stage takes
--block {groups,all_instr}, --rows <Row> ... and --limit N to restrict the work
(`table` then reports the reduced n).

CONVENTIONS -- the same ones that produced the published numbers:
  SNR       mono SNR (clipped to +-100 dB). An estimate whose stereo RMS is
            more than 40 dB below the target's is scored at -30 dB.
            Mixture row: the mixture as the estimate.
            Ground truth: infinite by definition, not computed.
  MERT-MSE  `evaluation/mert_mse.py` (MERT-v1-95M layer 12, gensvs recipe).
            Mixture row: mixture vs reference. Ground truth: 0.
  Judge     JudgeOverall of facebook/sam-audio-judge on (estimate, mixture,
            prompt), 48 kHz mono, batch 8 -- as sam-audio/benchmark_metrics.py.
            Mixture row scores the mixture, Ground truth the reference.
  Table     mean +- population sd over the manifest's items.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HERE = Path(os.environ.get("RESULTS_PAPER", REPO / "results_paper")).resolve()
EST = Path(os.environ.get("ESTIMATES_PAPER", HERE.parent / "estimates_paper")).resolve()
SAM_REPO = Path(os.environ.get("SAM_AUDIO_REPO", REPO.parent / "sam-audio"))
FLOOR, MARGIN = -30.0, 40.0

LABEL = {  # manifest column -> (setting, paper label)
    "Ref_Mixture": ("Ref.", "Mixture"),
    "Ref_GroundTruth": ("Ref.", "Ground truth"),
    "Ref_GroundTruth-AE": ("Ref.", "Ground truth (AE)"),
    "Ref_IdealRatioMask": ("Ref.", "Ideal Ratio Mask"),
    "Static_HTDemucs": ("Static", "HTDemucs"),
    "Static_BS-Locoformer": ("Static", "BS-Locoformer"),
    "Oracle_SAM-A": ("Oracle", "SAM-A."),
    "Oracle_BS-Locoformer": ("Oracle", "BS-Locoformer"),
    "Oracle_DiT": ("Oracle", "DiT"),
    "Blind_Banquet-other-song": ("Blind", "Banquet (other song)"),
    "Blind_MusicFlamingo+SAM-A": ("Blind", "MusicFlam. + SAM-A."),
    "Blind_MuS3D-BS-Locoformer": ("Blind", "MuS3D (BS-Locoformer)"),
    "Blind_MuS3D-DiT": ("Blind", "MuS3D (DiT)"),
}
BLOCKS = {"groups": "groups_vdbgp.csv", "all_instr": "all_instr.csv"}
META = ("song", "segment", "stem", "track", "prompt")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reproduce_tables import SEP_EXPECTED as EXPECTED, same  # noqa: E402


def load_manifest(block: str, limit: int | None):
    rows = list(csv.DictReader(open(HERE / "populations" / BLOCKS[block])))
    systems = [c for c in rows[0] if c not in META]
    return (rows[:limit] if limit else rows), systems


def item_key(r: dict) -> tuple:
    return (r["song"], r["segment"], r.get("track") or r.get("stem"))


def write(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ------------------------------------------------------------------- SNR ----
def stage_snr(args, block):
    import numpy as np
    import soundfile as sf

    def _snr(est, ref, eps=1e-8):
        num, den = float(np.sum(ref ** 2)), float(np.sum((ref - est) ** 2))
        return float(np.clip(10 * np.log10((num + eps) / (den + eps)), -100, 100))

    def _dbrms(x):
        return float(10 * np.log10(np.mean(x ** 2) + 1e-12))

    def rd(p):
        return sf.read(str(EST / p), dtype="float32", always_2d=True)[0]

    rows, systems = load_manifest(block, args.limit)
    for sysname in systems:
        if sysname == "Ref_GroundTruth" or (args.rows and sysname not in args.rows):
            continue
        out = []
        for r in rows:
            tgt, mix = rd(r["Ref_GroundTruth"]), rd(r["Ref_Mixture"])
            if sysname == "Ref_Mixture":
                n = min(tgt.shape[0], mix.shape[0])
                v, floored = _snr(mix[:n].mean(1), tgt[:n].mean(1)), False
            else:
                est = rd(r[sysname])
                n = min(est.shape[0], tgt.shape[0], mix.shape[0])
                e, t = est[:n], tgt[:n]
                if e.shape[1] == 1:
                    e = np.repeat(e, t.shape[1], axis=1)
                floored = _dbrms(e) < _dbrms(t) - MARGIN
                v = FLOOR if floored else _snr(e.mean(1), t.mean(1))
            out.append([*item_key(r), f"{v:.6f}", str(floored)])
        write(Path(args.work) / block / "snr" / f"{sysname}.csv",
              ["song", "segment", "item", "snr_mono", "silent_floored"], out)
        print(f"[snr] {block}/{sysname}: {len(out)} items")


# ------------------------------------------------------------------ MERT ----
def stage_mert(args, block):
    sys.path.insert(0, str(REPO / "evaluation"))
    from mert_mse import Mert, mse_mert

    rows, systems = load_manifest(block, args.limit)
    mert = Mert(args.device)
    targets = {item_key(r): mert.embed(EST / r["Ref_GroundTruth"]) for r in rows}
    for sysname in systems:
        if args.rows and sysname not in args.rows:
            continue
        out = []
        for r in rows:
            k = item_key(r)
            if sysname == "Ref_GroundTruth":
                v = 0.0
            else:
                e = mert.embed(EST / r[sysname])
                # a silent estimate has no embedding; the published column
                # contains none (every item here produced audio), so refuse
                if e is None or targets[k] is None:
                    raise SystemExit(f"silent audio for {sysname} {k}")
                v = mse_mert(e, targets[k])
            out.append([*k, f"{v:.6f}"])
        write(Path(args.work) / block / "mert" / f"{sysname}.csv",
              ["song", "segment", "item", "MertMSE"], out)
        print(f"[mert] {block}/{sysname}: {len(out)} items")


# ----------------------------------------------------------------- Judge ----
def stage_judge(args, block):
    # sam-audio's `eval/metrics` package, NOT evaluation/metrics.py: the
    # benchmarks directory is deliberately kept off sys.path in this stage.
    sys.path.insert(0, str(SAM_REPO / "eval"))
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    from metrics import Judge

    fs_out = 48_000

    def load(p):
        data, fs = sf.read(str(EST / p), dtype="float32", always_2d=True)
        wav = torch.from_numpy(np.ascontiguousarray(data.T)).mean(0)
        return torchaudio.functional.resample(wav, fs, fs_out) if fs != fs_out else wav

    rows, systems = load_manifest(block, args.limit)
    judge = Judge(device=torch.device(args.device))
    for sysname in systems:
        if args.rows and sysname not in args.rows:
            continue
        col = sysname
        out = []
        for i in range(0, len(rows), args.batch):
            batch = rows[i:i + args.batch]
            scores = judge(target_wavs=[load(r[col]) for r in batch],
                           input_wavs=[load(r["Ref_Mixture"]) for r in batch],
                           descriptions=[r["prompt"] for r in batch],
                           reference_wavs=[load(r["Ref_GroundTruth"]) for r in batch],
                           target_wavs_sample_rate=fs_out, videos=None)
            for j, r in enumerate(batch):
                out.append([*item_key(r), f"{float(scores['JudgeOverall'][j]):.6f}"])
        write(Path(args.work) / block / "judge" / f"{sysname}.csv",
              ["song", "segment", "item", "JudgeOverall"], out)
        print(f"[judge] {block}/{sysname}: {len(out)} items")


# ----------------------------------------------------------------- table ----
def stage_table(args, block):
    work = Path(args.work) / block
    rows, systems = load_manifest(block, args.limit)
    col = {"judge": "JudgeOverall", "snr": "snr_mono", "mert": "MertMSE"}
    print(f"\n{block}  (n={len(rows)} items)")
    print(f"{'':32s} {'Judge':>16s} {'SNR':>18s} {'MERT-MSE':>16s}   vs paper")
    mismatch = 0
    for sysname in systems:
        setting, label = LABEL[sysname]
        key = f"{label} ({setting})" if label == "BS-Locoformer" else label
        cells, got = [], []
        for m in ("judge", "snr", "mert"):
            p = work / m / f"{sysname}.csv"
            if m == "snr" and sysname == "Ref_GroundTruth":
                cells.append("inf"); got += [None, None]; continue
            if not p.exists():
                cells.append("(not run)"); got += [None, None]; continue
            v = [float(r[col[m]]) for r in csv.DictReader(open(p))]
            mean, sd = st.mean(v), st.pstdev(v)
            cells.append(f"{mean:.4f}+-{sd:.4f}"); got += [mean, sd]
        exp = EXPECTED[block].get(key)
        verdict = ""
        if exp and not args.limit:
            bad = [i for i, (g, e) in enumerate(zip(got, exp))
                   if g is not None and not same(g, e)]
            mismatch += bool(bad)
            verdict = "OK" if not bad else "DIFF " + ", ".join(
                f"{['Judge','Judge sd','SNR','SNR sd','MERT','MERT sd'][i]} "
                f"{got[i]:.3f} vs {exp[i]}" for i in bad)
        print(f"{label + ' [' + setting + ']':32s} {cells[0]:>16s} {cells[1]:>18s} {cells[2]:>16s}   {verdict}")
    if not args.limit:
        print(f"-> {block}: {'all published cells reproduced' if not mismatch else f'{mismatch} row(s) differ'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["snr", "mert", "judge", "table"])
    ap.add_argument("--block", choices=list(BLOCKS), nargs="*", default=list(BLOCKS))
    ap.add_argument("--rows", nargs="*", default=None, help="manifest columns, e.g. Oracle_DiT")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--work", default=str(HERE / "separation"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    fn = {"snr": stage_snr, "mert": stage_mert, "judge": stage_judge, "table": stage_table}[args.stage]
    for block in args.block:
        fn(args, block)


if __name__ == "__main__":
    main()
