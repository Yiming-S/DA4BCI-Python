"""Tests for TCA."""

import numpy as np
from da4bci.methods.tca import domain_adaptation_tca
from da4bci.metrics.distance import compute_mmd


class TestTCA:
    def test_structure(self, src_tgt_42):
        src, tgt = src_tgt_42
        res = domain_adaptation_tca(src, tgt, k=3, sigma=1, mu=0.1)
        assert isinstance(res, dict)
        assert set(res.keys()) == {"weighted_source_data", "target_data", "eigenvalue"}

    def test_dimensions(self, src_tgt_42):
        src, tgt = src_tgt_42
        for k in [2, 3, 5]:
            res = domain_adaptation_tca(src, tgt, k=k, sigma=1, mu=1)
            assert res["weighted_source_data"].shape[1] == k
            assert res["target_data"].shape[1] == k
            assert res["weighted_source_data"].shape[0] == 20
            assert res["target_data"].shape[0] == 20

    def test_reduces_mmd(self):
        rng = np.random.RandomState(42)
        src = rng.randn(40, 5)
        tgt = rng.randn(40, 5) + 3
        res = domain_adaptation_tca(src, tgt, k=5, sigma=1, mu=0.1)
        mmd_before = compute_mmd(src, tgt, sigma=1)
        mmd_after = compute_mmd(res["weighted_source_data"], res["target_data"], sigma=1)
        assert mmd_after < mmd_before

    def test_deterministic(self):
        def run():
            rng = np.random.RandomState(123)
            s = rng.randn(20, 5)
            t = rng.randn(20, 5) + 1
            return domain_adaptation_tca(s, t, k=3, sigma=1, mu=1)
        r1 = run()
        r2 = run()
        # Eigenvectors may have sign flips; check absolute values
        np.testing.assert_allclose(np.abs(r1["weighted_source_data"]),
                                   np.abs(r2["weighted_source_data"]), atol=1e-10)

    def test_reference(self):
        rng = np.random.RandomState(2024)
        src = rng.randn(10, 5)
        tgt = rng.randn(10, 5) + 1
        res = domain_adaptation_tca(src, tgt, k=3, sigma=1, mu=0.5)
        assert np.all(np.isfinite(res["weighted_source_data"]))
        assert res["weighted_source_data"].shape == (10, 3)
