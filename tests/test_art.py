"""Tests for ART."""

import numpy as np
from da4bci.methods.art import domain_adaptation_art


class TestART:
    def test_structure(self, src_tgt_large_42):
        src, tgt = src_tgt_large_42
        res = domain_adaptation_art(src, tgt)
        assert isinstance(res, dict)
        assert set(res.keys()) == {"weighted_source_data", "target_data", "transformation_matrix"}

    def test_preserves_dimensions(self):
        rng = np.random.RandomState(42)
        src = rng.randn(40, 5)
        tgt = rng.randn(30, 5) + 1
        res = domain_adaptation_art(src, tgt)
        assert res["weighted_source_data"].shape == (40, 5)
        assert res["target_data"].shape == (30, 5)
        assert res["transformation_matrix"].shape == (5, 5)

    def test_target_unchanged(self, src_tgt_large_42):
        src, tgt = src_tgt_large_42
        res = domain_adaptation_art(src, tgt)
        np.testing.assert_array_equal(res["target_data"], tgt)

    def test_mean_alignment(self):
        rng = np.random.RandomState(42)
        n, p = 200, 5
        src = rng.randn(n, p)
        tgt = rng.randn(n, p) + 3
        res = domain_adaptation_art(src, tgt)
        mean_aligned = res["weighted_source_data"].mean(axis=0)
        mean_target = tgt.mean(axis=0)
        mean_src = src.mean(axis=0)
        dist_after = np.sum((mean_aligned - mean_target) ** 2)
        dist_before = np.sum((mean_src - mean_target) ** 2)
        assert dist_after < dist_before

    def test_deterministic(self):
        def run():
            rng = np.random.RandomState(123)
            s = rng.randn(40, 5)
            t = rng.randn(40, 5) + 1
            return domain_adaptation_art(s, t)
        r1 = run()
        r2 = run()
        np.testing.assert_array_equal(r1["weighted_source_data"], r2["weighted_source_data"])

    def test_reference(self):
        rng = np.random.RandomState(2024)
        src = rng.randn(10, 5)
        tgt = rng.randn(10, 5) + 1
        res = domain_adaptation_art(src, tgt)
        assert np.all(np.isfinite(res["weighted_source_data"]))
        assert np.all(np.isfinite(res["transformation_matrix"]))
