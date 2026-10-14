"""Fast unit tests: no data, no downloads, CPU only.

    uv run python -m unittest discover tests
"""

import unittest

import numpy as np
import torch

from tests.common import REPO


class TestConfigs(unittest.TestCase):
    def test_training_configs_compose(self):
        from hydra import compose, initialize_config_dir
        cases = [("SEG", [], "RecursiveFMGenerator"),
                 ("SEG", ["model=SEG_resemb"], "RecursiveFMGenerator"),
                 ("DiT_separator", [], "LatentFlowSeparator"),
                 ("BSLocoformer_separator", [], "query_conditioned_bslocoformer.regression.Regressor"),
                 ("BSLocoformer_static", [], "static_bslocoformer.regression.Regressor")]
        for name, overrides, target in cases:
            with initialize_config_dir(version_base=None, config_dir=str(REPO / "configs")):
                cfg = compose(config_name=name, overrides=overrides + ["trainer.output_dir=/tmp/x"])
            self.assertIn(target, cfg.model._target_)
            self.assertIn("train", cfg.data)
            self.assertIn("val", cfg.data)

    def test_resemb_variant_sets_both_flags(self):
        from hydra import compose, initialize_config_dir
        with initialize_config_dir(version_base=None, config_dir=str(REPO / "configs")):
            cfg = compose(config_name="SEG", overrides=["model=SEG_resemb", "trainer.output_dir=/tmp/x"])
        self.assertTrue(cfg.model.use_residual_embedding)
        self.assertEqual(cfg.model.backbone.res_emb_dim, 128)

    def test_referenced_files_exist(self):
        from omegaconf import OmegaConf
        for f in ("seg", "separator", "separator_vdbgpo"):
            cfg = OmegaConf.load(REPO / "configs/data" / f"{f}.yaml")
            irs = OmegaConf.select(cfg, "train.impulse_responses") or \
                OmegaConf.select(cfg, "train.dataset.impulse_responses")
            for path in irs.values():
                self.assertTrue((REPO / path).exists(), path)
        for p in ("src/query_conditioned_bslocoformer/pca_params_train.pkl",
                  "src/classifiers/stem_classifier.pkl",
                  "src/classifiers/multilabel_classifier.pkl"):
            self.assertTrue((REPO / p).exists(), p)


class TestProtocol(unittest.TestCase):
    groups = ["drums", "drums", "bass", "vocals", "guitar"]

    def test_sample_recursive_partitions_units(self):
        from source_embedding_generator.protocol import LEVEL_STEM, LEVEL_TRACK, sample_recursive
        rng = np.random.default_rng(0)
        for level in (LEVEL_TRACK, LEVEL_STEM):
            for _ in range(50):
                avail, target, residual, lvl = sample_recursive(len(self.groups), self.groups,
                                                                rng=rng, level=level)
                self.assertEqual(lvl, level)
                used = [i for u in avail for i in u] + list(target) + list(residual)
                self.assertEqual(sorted(used), list(range(len(self.groups))))
                if level == LEVEL_STEM:   # a unit never splits a family
                    for u in avail + [list(target)]:
                        self.assertEqual(len({self.groups[i] for i in u}), 1)

    def test_sample_targets_levels(self):
        from source_embedding_generator.protocol import LEVEL_STEM, sample_targets
        rng = np.random.default_rng(1)
        for _ in range(50):
            targets, residual, _, level = sample_targets(len(self.groups), self.groups,
                                                         rng=rng, level=LEVEL_STEM)
            self.assertEqual(level, LEVEL_STEM)
            used = sorted([i for t in targets for i in t] + list(residual))
            self.assertEqual(used, list(range(len(self.groups))))

    def test_active_tracks_drops_silence(self):
        from source_embedding_generator.protocol import active_tracks
        tracks = torch.zeros(3, 2, 44100)
        tracks[0] = 0.1 * torch.randn(2, 44100)
        tracks[2] = 1e-7 * torch.randn(2, 44100)
        self.assertEqual(active_tracks(tracks).tolist(), [True, False, False])

    def test_collate_recursive(self):
        from source_embedding_generator.protocol import collate_recursive
        batch = [(torch.zeros(3, 2, 10), ["a", "b", "residual"], 0, 4),
                 (torch.zeros(2, 2, 10), ["c", "residual"], 1, 2)]
        audio, counts, labels, levels, n_units = collate_recursive(batch)
        self.assertEqual(tuple(audio.shape), (5, 2, 10))
        self.assertEqual(counts.tolist(), [3, 2])
        self.assertEqual(levels.tolist(), [0, 1])
        self.assertEqual(len(labels), 5)


class TestSEGBackbone(unittest.TestCase):
    def _dit(self, **kw):
        from source_embedding_generator.recursive_dit import RecursiveSetDiT
        torch.manual_seed(0)
        return RecursiveSetDiT(embedding_dim=128, io_channels=64, embed_dim=64, depth=2,
                               num_heads=4, max_groups=16, cond_channels=128,
                               mixture_emb_dim=128, chunk_pos_dim=8, group_tag_dim=32,
                               avail_dim=128, **kw).eval()

    def _inputs(self, b=2):
        g = torch.Generator().manual_seed(0)
        return dict(x=torch.randn(b, 2, 64, 2, generator=g), t=torch.rand(b, generator=g),
                    cond=torch.randn(b, 128, 20, generator=g),
                    mixture_emb=torch.randn(b, 128, generator=g),
                    avail=torch.randn(b, 3, 128, generator=g),
                    avail_mask=torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool),
                    n_avail=torch.tensor([2, 1]), level=torch.tensor([0, 1]),
                    group_tags=torch.randn(b, 2, 32, generator=g))

    def _perturb_zero_init(self, m):
        for p in m.parameters():                   # zero-initialised output layers
            p.data.add_(0.02 * torch.randn_like(p))

    def test_forward_shape_and_cfg(self):
        m = self._dit()
        self._perturb_zero_init(m)
        inp = self._inputs()
        with torch.no_grad():
            out = m(**inp)
            out_cfg = m(**inp, cfg_scale=2.0)
            out_unc = m(**inp, force_uncond=True)
        self.assertEqual(tuple(out.shape), (2, 2, 64, 2))
        self.assertTrue(torch.allclose(out_cfg, out_unc + 2.0 * (out - out_unc), atol=1e-5))

    def test_slot_permutation_equivariance(self):
        m = self._dit()
        self._perturb_zero_init(m)
        inp = self._inputs()
        is_res = torch.tensor([[False, True], [False, True]])
        with torch.no_grad():
            a = m(**inp, is_residual=is_res)
            perm = dict(inp, x=inp["x"].flip(1), group_tags=inp["group_tags"].flip(1))
            b = m(**perm, is_residual=is_res.flip(1))
        self.assertTrue(torch.allclose(a, b.flip(1), atol=1e-5))

    def test_masked_extracted_units_do_not_matter(self):
        m = self._dit()
        self._perturb_zero_init(m)
        inp = self._inputs()
        changed = inp["avail"].clone()
        changed[0, 2] += 10.0                     # masked out for item 0
        with torch.no_grad():
            a = m(**inp)
            b = m(**dict(inp, avail=changed))
        self.assertTrue(torch.allclose(a, b, atol=1e-5))

    def test_residual_embedding_stream(self):
        with self.assertRaises(ValueError):
            self._dit(res_mix_dim=128)
        m = self._dit(res_emb_dim=128)
        with torch.no_grad():
            out = m(**self._inputs(), res_emb=torch.randn(2, 128))
        self.assertEqual(tuple(out.shape), (2, 2, 64, 2))


class TestSeparatorBackbones(unittest.TestCase):
    def test_dit_backbone_forward(self):
        from query_conditioned_dit.modulation_backbone import QueryLevelModulatedSamAudioDiT
        torch.manual_seed(0)
        m = QueryLevelModulatedSamAudioDiT(latent_dim=128, cond_dim=128, dim=64, n_layers=2,
                                           n_heads=2, max_positions=64, flip_time=True).eval()
        with torch.no_grad():
            out = m(torch.randn(2, 25, 256), torch.rand(2), torch.randn(2, 25, 128),
                    torch.randn(2, 128), level=torch.tensor([0, 1]))
        self.assertEqual(tuple(out.shape), (2, 25, 256))
        self.assertFalse(any(k.startswith("transformer.layers.0.cross_attention")
                             for k in m.state_dict()))

    def test_sam_audio_conversion(self):
        from query_conditioned_dit.modulation_backbone import QueryLevelModulatedSamAudioDiT
        from query_conditioned_dit.sam_audio_init import convert, is_released_checkpoint
        m = QueryLevelModulatedSamAudioDiT(latent_dim=128, cond_dim=128, dim=64, n_layers=1,
                                           n_heads=2, max_positions=64)
        ours = m.state_dict()
        released = {k: torch.randn_like(v) for k, v in ours.items()
                    if k.startswith("transformer.") and "y_embedder" not in k and "q_" not in k}
        released["transformer.layers.0.cross_attention.q_proj.weight"] = torch.randn(64, 64)
        released["proj.weight"] = torch.randn(64, 768)
        released["proj.bias"] = torch.randn(64)
        released["align_masked_video.gate"] = torch.tensor([0.3])
        released["align_masked_video.conv.bias"] = torch.randn(64)
        released["align_masked_video.layer_norm.weight"] = torch.ones(64)
        released["align_masked_video.layer_norm.bias"] = torch.zeros(64)
        self.assertTrue(is_released_checkpoint(released))
        out = convert(released, ours)
        w = released["proj.weight"]
        self.assertTrue(torch.equal(out["proj.weight"][:, :256], w[:, :256]))
        self.assertTrue(torch.equal(out["proj.weight"][:, 256:], w[:, 512:640] + w[:, 640:768]))
        self.assertNotIn("transformer.layers.0.cross_attention.q_proj.weight", out)
        missing, unexpected = m.load_state_dict(out, strict=False)
        self.assertEqual(list(unexpected), [])

    def test_bslocoformer_backbone_forward(self):
        from query_conditioned_bslocoformer.backbone import LevelConditionedBanditBSLocoformer
        torch.manual_seed(0)
        m = LevelConditionedBanditBSLocoformer(n_chan=2, masking=True, n_bands=16, emb_dim=32,
                                               mlp_dim=64, transformer_blocks=1,
                                               ffn_hidden_dim=[48, 48], attention_heads=4,
                                               norm_groups=4).eval()
        with torch.no_grad():
            out = m(torch.randn(2, 2, 44100), torch.randn(2, 128), level=torch.tensor([0, 1]))
        self.assertEqual(out.shape[-1], 44100)


class TestUtilities(unittest.TestCase):
    def test_inverse_lr(self):
        from utils.scheduler import InverseLR
        p = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.AdamW([p], lr=1e-4)
        sched = InverseLR(opt, inv_gamma=1000, power=1.0, warmup=0.99)
        for step in range(1, 500):
            opt.step()
            sched.step()
            expected = (1 - 0.99 ** (step + 1)) * 1e-4 * (1 + step / 1000) ** -1
            self.assertAlmostEqual(opt.param_groups[0]["lr"], expected, places=12)

    def test_metrics_logger(self):
        from utils.logging import MetricsLogger
        log = MetricsLogger()
        log.add({"loss": torch.tensor(1.0)})
        log.add({"loss": torch.tensor(3.0)})
        self.assertAlmostEqual(float(log.pop()["loss"]), 2.0)
        self.assertEqual(log.pop(), {})

    def test_stem_classifier(self):
        from classifiers.stem_classifier import StemClassifier
        clf = StemClassifier.load()
        names = clf.predict(np.random.default_rng(0).normal(size=(4, 128)).astype(np.float32), "stem")
        self.assertEqual(len(names), 4)
        from data.moisesdb import STEM_TAX
        self.assertTrue(all(n in STEM_TAX for n in names))

    def test_checkpoint_config_remaps(self):
        """Old checkpoint configs name modules that were moved or renamed."""
        import tempfile
        from pathlib import Path
        from omegaconf import OmegaConf
        import source_embedding_generator.separator_io as io
        cfg = OmegaConf.create({"model": {
            "_target_": "latent_separator.separator.LatentFlowSeparator",
            "backbone": {"_target_": "latent_separator.modulation_backbone.X", "init_from": "x"},
            "loss": {"_target_": "loss.snr.ThresSNRLossWithInactiveSource"},
            "query_adapter": {"_target_": "query_conditioned_bslocoformer.query_model.PcaAdapter"}}})
        captured = {}

        def fake_instantiate(c):
            captured["cfg"] = c
            raise RuntimeError("stop")
        import hydra.utils
        orig = hydra.utils.instantiate
        hydra.utils.instantiate = fake_instantiate
        try:
            with tempfile.TemporaryDirectory() as d:
                OmegaConf.save(cfg, Path(d) / "config.yaml")
                with self.assertRaises(RuntimeError):
                    io.load_separator(d)
        finally:
            hydra.utils.instantiate = orig
        m = captured["cfg"]
        self.assertTrue(m._target_.startswith("query_conditioned_dit."))
        self.assertTrue(m.backbone._target_.startswith("query_conditioned_dit."))
        self.assertIsNone(m.backbone.init_from)
        self.assertEqual(m.loss._target_, "query_conditioned_bslocoformer.loss.ThresSNRLossWithInactiveSource")
        self.assertTrue(str(m.query_adapter.pca_pkl).endswith("pca_params_train.pkl"))


class TestEvaluation(unittest.TestCase):
    def test_discovery_scoring(self):
        import reproduce_tables as rt
        true = {"s/seg000": {"drums", "bass"}, "s/seg001": {"vocals"}}
        pred = {"s/seg000": ["drums", "drums", "guitar"], "s/seg001": ["vocals"]}
        (f1, p, r, dk), n = rt.score(pred, true, track=False)
        self.assertEqual(n, 2)
        self.assertAlmostEqual(p[0], (1 / 3 + 1) / 2)       # a duplicate costs precision
        self.assertAlmostEqual(r[0], (1 / 2 + 1) / 2)
        self.assertAlmostEqual(f1[0], (2 * 1 / 5 + 1) / 2)
        self.assertAlmostEqual(dk[0], (1 + 0) / 2)

    def test_published_rounding(self):
        import reproduce_tables as rt
        self.assertTrue(rt.same(1.246, "1.3"))    # rounded from 1.25
        self.assertTrue(rt.same(0.0745, "0.07"))
        self.assertTrue(rt.same(10.485, "10"))
        self.assertFalse(rt.same(4.10, "4.2"))
