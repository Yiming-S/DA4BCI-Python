"""Tests for SA."""

import numpy as np
from da4bci.methods.sa import domain_adaptation_sa


class TestSA:
    def test_structure(self, src_tgt_42):
        src, tgt = src_tgt_42
        res = domain_adaptation_sa(src, tgt, k=3)
        assert isinstance(res, dict)
        assert set(res.keys()) == {"weighted_source_data", "target_data", "eigenvalue"}

    def test_dimensions(self):
        rng = np.random.RandomState(42)
        src = rng.randn(40, 5)
        tgt = rng.randn(40, 5) + 2
        for k in [2, 3, 5]:
            res = domain_adaptation_sa(src, tgt, k=k)
            assert res["weighted_source_data"].shape[1] == k
            assert res["target_data"].shape[1] == k

    def test_k_capped(self):
        rng = np.random.RandomState(42)
        src = rng.randn(20, 3)
        tgt = rng.randn(20, 3) + 1
        res = domain_adaptation_sa(src, tgt, k=10)
        assert res["weighted_source_data"].shape[1] == 3

    def test_W_shape(self, src_tgt_42):
        src, tgt = src_tgt_42
        res = domain_adaptation_sa(src, tgt, k=3)
        assert res["eigenvalue"].shape == (3, 3)

    def test_deterministic(self):
        def run():
            rng = np.random.RandomState(123)
            s = rng.randn(20, 5)
            t = rng.randn(20, 5) + 1
            return domain_adaptation_sa(s, t, k=3)
        r1 = run()
        r2 = run()
        np.testing.assert_array_equal(r1["weighted_source_data"], r2["weighted_source_data"])

    def test_reference(self):
        rng = np.random.RandomState(2024)
        src = rng.randn(10, 5)
        tgt = rng.randn(10, 5) + 1
        res = domain_adaptation_sa(src, tgt, k=3)
        assert np.all(np.isfinite(res["weighted_source_data"]))
        assert res["weighted_source_data"].shape == (10, 3)
