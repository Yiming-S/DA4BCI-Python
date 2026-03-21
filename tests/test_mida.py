"""Tests for MIDA."""

import numpy as np
from da4bci.methods.mida import domain_adaptation_mida


class TestMIDA:
    def test_structure(self, src_tgt_42):
        src, tgt = src_tgt_42
        res = domain_adaptation_mida(src, tgt, k=3, sigma=1, mu=0.1, maximize=True)
        assert isinstance(res, dict)
        assert set(res.keys()) == {"weighted_source_data", "target_data", "eigenvalue"}

    def test_dimensions(self, src_tgt_42):
        src, tgt = src_tgt_42
        for k in [2, 3, 5]:
            res = domain_adaptation_mida(src, tgt, k=k, sigma=1, mu=0.1, maximize=True)
            assert res["weighted_source_data"].shape[1] == k
            assert res["target_data"].shape[1] == k

    def test_max_min_differ(self, src_tgt_42):
        src, tgt = src_tgt_42
        res_max = domain_adaptation_mida(src, tgt, k=3, sigma=1, mu=0.1, maximize=True)
        res_min = domain_adaptation_mida(src, tgt, k=3, sigma=1, mu=0.1, maximize=False)
        assert np.all(np.isfinite(res_max["weighted_source_data"]))
        assert np.all(np.isfinite(res_min["weighted_source_data"]))
        assert not np.allclose(res_max["weighted_source_data"], res_min["weighted_source_data"])

    def test_deterministic(self):
        def run():
            rng = np.random.RandomState(123)
            s = rng.randn(20, 5)
            t = rng.randn(20, 5) + 1
            return domain_adaptation_mida(s, t, k=3, sigma=1, mu=0.1, maximize=True)
        r1 = run()
        r2 = run()
        np.testing.assert_array_equal(r1["weighted_source_data"], r2["weighted_source_data"])

    def test_reference(self):
        rng = np.random.RandomState(2024)
        src = rng.randn(10, 5)
        tgt = rng.randn(10, 5) + 1
        res = domain_adaptation_mida(src, tgt, k=3, sigma=1, mu=0.1, maximize=True)
        assert np.all(np.isfinite(res["weighted_source_data"]))
        assert res["weighted_source_data"].shape == (10, 3)
