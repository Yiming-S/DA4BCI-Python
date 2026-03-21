"""Tests for PT."""

import numpy as np
from da4bci.methods.pt import domain_adaptation_pt


class TestPT:
    def test_structure(self, src_tgt_large_42):
        src, tgt = src_tgt_large_42
        res = domain_adaptation_pt(src, tgt)
        assert isinstance(res, dict)
        assert set(res.keys()) == {"weighted_source_data", "target_data", "transformation_matrix"}

    def test_preserves_dimensions(self):
        rng = np.random.RandomState(42)
        src = rng.randn(40, 5)
        tgt = rng.randn(30, 5) + 1
        res = domain_adaptation_pt(src, tgt)
        assert res["weighted_source_data"].shape == (40, 5)
        assert res["target_data"].shape == (30, 5)
        assert res["transformation_matrix"].shape == (5, 5)

    def test_target_unchanged(self, src_tgt_large_42):
        src, tgt = src_tgt_large_42
        res = domain_adaptation_pt(src, tgt)
        np.testing.assert_array_equal(res["target_data"], tgt)

    def test_covariance_alignment(self):
        rng = np.random.RandomState(42)
        n, p = 500, 4
        src = rng.randn(n, p)
        L = np.array([[2, 0.5, 0, 0], [0.5, 1, 0, 0],
                       [0, 0, 3, 0.3], [0, 0, 0.3, 1.5]])
        tgt = rng.randn(n, p) @ L + 2
        res = domain_adaptation_pt(src, tgt)
        cov_aligned = np.cov(res["weighted_source_data"], rowvar=False)
        cov_tgt = np.cov(tgt, rowvar=False)
        cov_src = np.cov(src, rowvar=False)
        dist_before = np.linalg.norm(cov_src - cov_tgt, "fro")
        dist_after = np.linalg.norm(cov_aligned - cov_tgt, "fro")
        assert dist_after < dist_before

    def test_deterministic(self):
        def run():
            rng = np.random.RandomState(123)
            s = rng.randn(40, 5)
            t = rng.randn(40, 5) + 1
            return domain_adaptation_pt(s, t)
        r1 = run()
        r2 = run()
        np.testing.assert_array_equal(r1["weighted_source_data"], r2["weighted_source_data"])

    def test_reference(self):
        rng = np.random.RandomState(2024)
        src = rng.randn(10, 5)
        tgt = rng.randn(10, 5) + 1
        res = domain_adaptation_pt(src, tgt)
        assert np.all(np.isfinite(res["weighted_source_data"]))
        assert np.all(np.isfinite(res["transformation_matrix"]))
