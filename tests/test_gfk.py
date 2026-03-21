"""Tests for GFK."""

import numpy as np
from da4bci.methods.gfk import domain_adaptation_gfk


class TestGFK:
    def test_structure(self):
        rng = np.random.RandomState(42)
        src = rng.randn(20, 10)
        tgt = rng.randn(20, 10) + 2
        res = domain_adaptation_gfk(src, tgt, dim_subspace=5)
        assert isinstance(res, dict)
        assert set(res.keys()) == {"weighted_source_data", "target_data", "G"}

    def test_preserves_rows_and_cols(self):
        rng = np.random.RandomState(42)
        src = rng.randn(20, 10)
        tgt = rng.randn(15, 10) + 1
        res = domain_adaptation_gfk(src, tgt, dim_subspace=5)
        assert res["weighted_source_data"].shape == (20, 10)
        assert res["target_data"].shape == (15, 10)
        assert res["G"].shape == (10, 10)

    def test_dim_subspace_capped(self):
        rng = np.random.RandomState(42)
        src = rng.randn(20, 3)
        tgt = rng.randn(20, 3) + 1
        res = domain_adaptation_gfk(src, tgt, dim_subspace=10)
        assert res["G"].shape == (3, 3)

    def test_G_symmetric(self):
        rng = np.random.RandomState(42)
        src = rng.randn(20, 10)
        tgt = rng.randn(20, 10) + 2
        res = domain_adaptation_gfk(src, tgt, dim_subspace=5)
        np.testing.assert_allclose(res["G"], res["G"].T, atol=1e-10)

    def test_deterministic(self):
        def run():
            rng = np.random.RandomState(123)
            s = rng.randn(20, 10)
            t = rng.randn(20, 10) + 1
            return domain_adaptation_gfk(s, t, dim_subspace=5)
        r1 = run()
        r2 = run()
        np.testing.assert_array_equal(r1["G"], r2["G"])

    def test_reference(self):
        rng = np.random.RandomState(2024)
        src = rng.randn(10, 8)
        tgt = rng.randn(10, 8) + 1
        res = domain_adaptation_gfk(src, tgt, dim_subspace=4)
        assert np.all(np.isfinite(res["weighted_source_data"]))
        assert res["weighted_source_data"].shape == (10, 8)
