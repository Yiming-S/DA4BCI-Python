"""Tests for M3D."""

import numpy as np
from da4bci.methods.m3d import domain_adaptation_m3d


class TestM3D:
    def test_structure(self):
        rng = np.random.RandomState(42)
        src = rng.randn(100, 10)
        tgt = rng.randn(100, 10) + 1
        labels = rng.choice([1, 2, 3], size=100)
        res = domain_adaptation_m3d(src, labels, tgt, l_iter=3, max_dim=10)
        assert isinstance(res, dict)
        assert "weighted_source_data" in res
        assert "target_data" in res

    def test_row_counts(self):
        rng = np.random.RandomState(42)
        n_s, n_t, p = 60, 40, 10
        src = rng.randn(n_s, p)
        tgt = rng.randn(n_t, p) + 1
        labels = rng.choice([1, 2], size=n_s)
        res = domain_adaptation_m3d(src, labels, tgt, l_iter=3, max_dim=8)
        assert res["weighted_source_data"].shape[0] == n_s
        assert res["target_data"].shape[0] == n_t
        assert res["weighted_source_data"].shape[1] == res["target_data"].shape[1]

    def test_different_stages(self):
        rng = np.random.RandomState(42)
        src = rng.randn(80, 8)
        tgt = rng.randn(80, 8) + 1
        labels = rng.choice([1, 2], size=80)

        res1 = domain_adaptation_m3d(
            src, labels, tgt,
            stage1={"method": "tca", "control": {"k": 5, "sigma": 1}},
            stage2={"method": "sa", "control": {"k": 5}},
            l_iter=2, max_dim=8,
        )
        assert np.all(np.isfinite(res1["weighted_source_data"]))

        res2 = domain_adaptation_m3d(
            src, labels, tgt,
            stage1={"method": "sa", "control": {"k": 5}},
            stage2={"method": "sa", "control": {"k": 5}},
            l_iter=2, max_dim=8,
        )
        assert np.all(np.isfinite(res2["weighted_source_data"]))

    def test_deterministic(self):
        def run():
            rng = np.random.RandomState(123)
            src = rng.randn(80, 8)
            tgt = rng.randn(80, 8) + 1
            labels = rng.choice([1, 2, 3], size=80)
            return domain_adaptation_m3d(src, labels, tgt, l_iter=3, max_dim=8)
        r1 = run()
        r2 = run()
        np.testing.assert_allclose(r1["weighted_source_data"],
                                   r2["weighted_source_data"], atol=1e-10)

    def test_reference(self):
        rng = np.random.RandomState(2024)
        src = rng.randn(50, 8)
        tgt = rng.randn(50, 8) + 1
        labels = rng.choice([1, 2], size=50)
        res = domain_adaptation_m3d(src, labels, tgt, l_iter=3, max_dim=6)
        assert np.all(np.isfinite(res["weighted_source_data"]))
        assert np.all(np.isfinite(res["target_data"]))
        assert res["weighted_source_data"].shape[0] == 50
