"""MuS3D inference: an audio file -> the stems SEG discovers in it.

    PYTHONPATH=src uv run python src/inference.py song.wav --out out/
    PYTHONPATH=src uv run python src/inference.py song.wav --separator bslocoformer

A 10 s segment (the middle of the file by default) is decomposed recursively by
the stem embedding generator; each generated query is separated from the
mixture by the query-conditioned separator and named by the stem classifier.

Defaults are the paper's blind MuS3D rows:
  SEG           raw (non-EMA) weights, 32 ODE steps, churn 0.1, mono input
  DiT           EMA weights, 32 ODE steps, churn 0.5
  BS-Locoformer raw weights

Written to `--out/<input name>/`:
    mixture.wav   the segment the models were given
    <name>.wav    one per discovered source (`<name>_2.wav` for repeats)
    trace.txt     extraction order and each step's stop cosine
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

REPO_ROOT = Path(__file__).resolve().parent.parent
#: the released `checkpoints/` directory
CKPTS = Path(os.environ.get("MUS3D_CHECKPOINTS", REPO_ROOT / "checkpoints"))
SAMPLE_RATE = 44100

SEG_CKPT = CKPTS / "SEG_mixemb"
SEPARATORS = {
    "dit": dict(ckpt=CKPTS / "DiT", weights="ema", steps=32, churn=0.5),
    "bslocoformer": dict(ckpt=CKPTS / "BS-Locoformer_query",
                         weights="model", steps=None, churn=0.0),
}
CLASSIFIER = CKPTS / "classifiers/stem_classifier_pcatrain.pkl"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="audio files")
    p.add_argument("--out", default="output/inference")
    p.add_argument("--separator", default="dit", choices=sorted(SEPARATORS))
    p.add_argument("--seg_ckpt", default=str(SEG_CKPT))
    p.add_argument("--sep_ckpt", default=None, help="default: per --separator")
    p.add_argument("--classifier", default=str(CLASSIFIER))
    p.add_argument("--level", default="stem", choices=["stem", "track"],
                   help="stem: instrument groups; track: individual instruments")
    p.add_argument("--start", type=float, default=None,
                   help="segment start in seconds (default: centred)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seg_steps", type=int, default=32)
    p.add_argument("--seg_churn", type=float, default=0.1)
    p.add_argument("--sep_steps", type=int, default=None)
    p.add_argument("--sep_churn", type=float, default=None)
    p.add_argument("--max_steps", type=int, default=16,
                   help="cap on the recursion; the stop symbol normally ends it")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    preset = SEPARATORS[args.separator]
    args.sep_ckpt = args.sep_ckpt or str(preset["ckpt"])
    args.sep_steps = preset["steps"] if args.sep_steps is None else args.sep_steps
    args.sep_churn = preset["churn"] if args.sep_churn is None else args.sep_churn
    args.sep_weights = preset["weights"]
    return args


def load_segment(path: Path, start_s: float | None, dur_s: float = 10.0):
    """-> ([1, 2, T] stereo at 44.1 kHz, start in seconds)."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    wav = wav[:2] if wav.shape[0] >= 2 else wav.repeat(2, 1)
    total_s = wav.shape[-1] / SAMPLE_RATE
    if start_s is None:
        start_s = max(0.0, (total_s - dur_s) / 2)
    n, i = int(dur_s * SAMPLE_RATE), int(start_s * SAMPLE_RATE)
    wav = wav[:, i:i + n]
    wav = torch.nn.functional.pad(wav, (0, n - wav.shape[-1]))
    return wav.unsqueeze(0), start_s


def slug(name: str) -> str:
    keep = [c if (c.isalnum() or c in "-_") else "_" for c in name]
    return "".join(keep).strip("_").replace("__", "_") or "source"


def unique(path: Path) -> Path:
    """`drums.wav` -> `drums_2.wav` when the recursion names two sources alike."""
    cand, k = path, 2
    while cand.exists():
        cand, k = path.with_name(f"{path.stem}_{k}{path.suffix}"), k + 1
    return cand


def write(path: Path, audio: torch.Tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.detach().float().cpu().numpy().T, SAMPLE_RATE)


@torch.no_grad()
def main():
    args = parse_args()
    from source_embedding_generator import protocol
    from source_embedding_generator.separator_io import load_separator, load_seg
    from classifiers.stem_classifier import StemClassifier
    from utils.torch_common import set_seed

    t0 = time.time()
    seg = load_seg(args.seg_ckpt, device=args.device, use_ema=False)
    seg.separator = load_separator(args.sep_ckpt, device=args.device,
                                   use_ema=(args.sep_weights == "ema"))
    if args.sep_steps is not None:
        seg.sep_steps = args.sep_steps
    seg.sep_churn = args.sep_churn
    clf = StemClassifier.load(args.classifier)
    clf.check_basis(seg)
    level = protocol.LEVEL_TRACK if args.level == "track" else protocol.LEVEL_STEM
    print(f"[load] SEG {args.seg_ckpt}\n[load] {args.separator} {args.sep_ckpt} "
          f"({time.time() - t0:.0f}s)", flush=True)

    for src in map(Path, args.inputs):
        stereo, start_s = load_segment(src, args.start)
        stereo = stereo.to(args.device)
        mono = stereo.mean(1, keepdim=True)
        name = src.parent.name if src.stem.lower() in {"mixture", "mix"} else src.stem
        out_dir = Path(args.out) / slug(name)
        write(out_dir / "mixture.wav", mono[0])

        set_seed(args.seed)
        run = seg.inference(mono, level=level, max_steps=args.max_steps,
                            steps=args.seg_steps, churn=args.seg_churn, audio=False)
        # SEG always sees the mono mixture. The separator gets the stereo
        # segment: the BS-Locoformer is stereo-in, the DiT downmixes itself.
        audio = seg.separate(stereo, run)
        names = list(clf.predict(run["emb"], args.level))

        trace = [f"{src}  start={start_s:.1f}s  level={args.level}  seed={args.seed}",
                 f"seg {args.seg_ckpt}", f"sep {args.sep_ckpt}", ""]
        for i, nm in enumerate(names):
            trace.append(f"step {i}  {nm:<24} stop_cos={float(run['stop_cos'][i]):+.3f}")
            write(unique(out_dir / f"{slug(nm)}.wav"), audio["est"][i])
        (out_dir / "trace.txt").write_text("\n".join(trace) + "\n")
        print(f"{name}: {run['n']} sources -> {', '.join(names)}  [{out_dir}]",
              flush=True)


if __name__ == "__main__":
    main()
