"""Tests for RD (Riemannian)."""

import numpy as np
from da4bci.methods.riemannian import domain_adaptation_riemannian


class TestRD:
    def test_structure(self, src_tgt_large_42):
        src, tgt = src_tgt_large_42
        res = domain_adaptation_riemannian(src, tgt)
        assert isinstance(res, dict)
        for key in ["weighted_source_data", "target_data", "rotation_matrix",
                     "cov_source_aligned", "riemannian_distance"]:
            assert key in res

    def test_preserves_dimensions(self):
        rng = np.random.RandomState(42)
        src = rng.randn(40, 5)
        tgt = rng.randn(30, 5) * 2
        res = domain_adaptation_riemannian(src, tgt)
        assert res["weighted_source_data"].shape == (40, 5)
        assert res["target_data"].shape == (30, 5)
        assert res["rotation_matrix"].shape == (5, 5)

    def test_target_unchanged(self, src_tgt_large_42):
        src, tgt = src_tgt_large_42
        res = domain_adaptation_riemannian(src, tgt)
        np.testing.assert_array_equal(res["target_data"], tgt)

    def test_rotation_orthogonal(self, src_tgt_large_42):
        src, tgt = src_tgt_large_42
        res = domain_adaptation_riemannian(src, tgt)
        R = res["rotation_matrix"]
        np.testing.assert_allclose(R.T @ R, np.eye(5), atol=1e-8)

    def test_distance_non_negative(self, src_tgt_large_42):
        src, tgt = src_tgt_large_42
        res = domain_adaptation_riemannian(src, tgt)
        assert res["riemannian_distance"] >= 0

    def test_distance_zero_identical(self):
        rng = np.random.RandomState(42)
        data = rng.randn(40, 5)
        res = domain_adaptation_riemannian(data, data)
        assert abs(res["riemannian_distance"]) < 1e-6

    def test_deterministic(self):
        def run():
            rng = np.random.RandomState(123)
            s = rng.randn(40, 5)
            t = rng.randn(40, 5) * 2
            return domain_adaptation_riemannian(s, t)
        r1 = run()
        r2 = run()
        np.testing.assert_array_equal(r1["weighted_source_data"], r2["weighted_source_data"])

    def test_reference(self):
        rng = np.random.RandomState(2024)
        src = rng.randn(10, 5)
        tgt = rng.randn(10, 5) * 2 + 1
        res = domain_adaptation_riemannian(src, tgt)
        assert np.all(np.isfinite(res["weighted_source_data"]))
        assert np.isfinite(res["riemannian_distance"])
