"""Tests for CORAL."""

import numpy as np
from da4bci.methods.coral import domain_adaptation_coral


class TestCORAL:
    def test_structure(self, src_tgt_large_42):
        src, tgt = src_tgt_large_42
        res = domain_adaptation_coral(src, tgt, lam=1e-5)
        assert isinstance(res, dict)
        assert set(res.keys()) == {"weighted_source_data", "target_data"}

    def test_preserves_dimensions(self):
        rng = np.random.RandomState(42)
        src = rng.randn(40, 5)
        tgt = rng.randn(30, 5) + 2
        res = domain_adaptation_coral(src, tgt)
        assert res["weighted_source_data"].shape == (40, 5)
        assert res["target_data"].shape == (30, 5)

    def test_target_unchanged(self, src_tgt_large_42):
        src, tgt = src_tgt_large_42
        res = domain_adaptation_coral(src, tgt)
        np.testing.assert_array_equal(res["target_data"], tgt)

    def test_aligns_covariance(self):
        rng = np.random.RandomState(42)
        n, p = 200, 5
        src = rng.randn(n, p)
        L = rng.randn(p, p)
        tgt = rng.randn(n, p) @ L
        res = domain_adaptation_coral(src, tgt, lam=1e-5)
        cov_before = np.cov(src, rowvar=False)
        cov_after = np.cov(res["weighted_source_data"], rowvar=False)
        cov_tgt = np.cov(tgt, rowvar=False)
        dist_before = np.linalg.norm(cov_before - cov_tgt, "fro")
        dist_after = np.linalg.norm(cov_after - cov_tgt, "fro")
        assert dist_after < dist_before

    def test_deterministic(self):
        def run():
            rng = np.random.RandomState(123)
            s = rng.randn(40, 5)
            t = rng.randn(40, 5) * 2
            return domain_adaptation_coral(s, t, lam=1e-5)
        r1 = run()
        r2 = run()
        np.testing.assert_array_equal(r1["weighted_source_data"], r2["weighted_source_data"])

    def test_reference(self):
        rng = np.random.RandomState(2024)
        src = rng.randn(10, 5)
        tgt = rng.randn(10, 5) * 2 + 1
        res = domain_adaptation_coral(src, tgt, lam=1e-5)
        assert np.all(np.isfinite(res["weighted_source_data"]))
        assert res["weighted_source_data"].shape == (10, 5)
