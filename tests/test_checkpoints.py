"""Slow tests with the paper checkpoints and released data.

    MUS3D_SLOW_TESTS=1 uv run python -m unittest tests.test_checkpoints

Skipped unless MUS3D_SLOW_TESTS=1 and the files exist. Frozen encoders (PaSST,
DAC-VAE) are downloaded on first use.
"""

import os
import subprocess
import sys
import unittest

import torch

from tests.common import CKPT, REPO, SLOW


def needs(*keys):
    missing = [k for k in keys if not CKPT[k].exists()]
    return unittest.skipUnless(SLOW and not missing,
                               "set MUS3D_SLOW_TESTS=1 (and have the checkpoints)")


class TestPaperCheckpoints(unittest.TestCase):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    @needs("seg", "seg_resemb")
    def test_seg_loads_and_recurses(self):
        from classifiers.stem_classifier import StemClassifier
        from source_embedding_generator.protocol import LEVEL_STEM
        from source_embedding_generator.separator_io import load_seg
        clf = StemClassifier.load()
        for key in ("seg", "seg_resemb"):
            seg = load_seg(CKPT[key], device=self.device)
            clf.check_basis(seg)
            torch.manual_seed(0)
            mix = 0.1 * torch.randn(1, 1, 441000, device=self.device)
            with torch.no_grad():
                out = seg.inference(mix, level=LEVEL_STEM, max_steps=3, steps=4, audio=False)
            self.assertGreaterEqual(out["n"], 1)
            self.assertEqual(out["emb"].shape[1], 128)

    @needs("loco", "loco_static")
    def test_bslocoformers_load(self):
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from source_embedding_generator.separator_io import checkpoint_config, load_separator
        sep = load_separator(CKPT["loco"], device="cpu", use_ema=False)
        with torch.no_grad():
            est = sep.sample(torch.randn(1, 1, 128), mix=torch.randn(1, 2, 44100),
                             query_is_embedding=True, level=torch.tensor([1]))
        self.assertEqual(tuple(est.shape), (1, 1, 2, 44100))
        cfg = checkpoint_config(CKPT["loco_static"]).model
        cfg.loss._target_ = "query_conditioned_bslocoformer.loss.ThresSNRLossWithInactiveSource"
        static = instantiate(cfg)
        sd = torch.load(CKPT["loco_static"] / "model.pth", map_location="cpu", weights_only=False)
        static.load_state_dict(sd, strict=True)

    @needs("dit")
    def test_dit_loads(self):
        from source_embedding_generator.separator_io import load_separator
        sep = load_separator(CKPT["dit"], device=self.device, use_ema=True)
        self.assertEqual(type(sep).__module__, "query_conditioned_dit.separator")


@unittest.skipUnless(SLOW, "set MUS3D_SLOW_TESTS=1")
class TestDeterminism(unittest.TestCase):
    """Two runs of a generation stage must produce the same audio, bit for bit."""

    @needs("seg", "dit", "loco")
    def test_stage_reruns_are_identical(self):
        import filecmp
        import tempfile
        from pathlib import Path
        env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
        with tempfile.TemporaryDirectory() as tmp:
            runs = []
            for i in ("a", "b"):
                out = Path(tmp) / i
                out.mkdir()
                for stage in ("seg", "mus3d_dit", "mus3d_loco"):
                    r = subprocess.run(
                        [sys.executable, str(REPO / "evaluation/generate_estimates.py"), stage,
                         "--out", str(out), "--limit", "1", "--blocks", "groups"],
                        capture_output=True, text=True, env=env)
                    self.assertEqual(r.returncode, 0, r.stderr[-2000:])
                runs.append(out)
            files = sorted(p.relative_to(runs[0]) for p in runs[0].rglob("*")
                           if p.suffix in (".wav", ".npy"))
            self.assertTrue(files, "the stages produced nothing")
            differ = [str(f) for f in files
                      if not filecmp.cmp(runs[0] / f, runs[1] / f, shallow=False)]
            self.assertEqual(differ, [], f"{len(differ)} of {len(files)} outputs differ")


@unittest.skipUnless(SLOW, "set MUS3D_SLOW_TESTS=1")
class TestReproduction(unittest.TestCase):
    def test_paper_tables(self):
        out = subprocess.run([sys.executable, str(REPO / "evaluation/reproduce_tables.py")],
                             capture_output=True, text=True, env=dict(os.environ))
        self.assertIn("all published cells reproduced", out.stdout, out.stdout[-2000:] + out.stderr[-2000:])


if __name__ == "__main__":
    unittest.main()
