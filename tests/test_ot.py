"""Tests for OT (Sinkhorn)."""

import numpy as np
from da4bci.methods.ot import domain_adaptation_ot


class TestOT:
    def test_structure(self, src_tgt_42):
        src, tgt = src_tgt_42
        res = domain_adaptation_ot(src, tgt, eps=0.05, maxit=500)
        assert isinstance(res, dict)
        for key in ["weighted_source_data", "target_data", "ot_plan", "cost",
                     "epsilon", "iterations", "converged", "residual"]:
            assert key in res

    def test_preserves_dimensions(self):
        rng = np.random.RandomState(42)
        src = rng.randn(20, 5)
        tgt = rng.randn(15, 5) + 2
        res = domain_adaptation_ot(src, tgt, eps=0.1)
        assert res["weighted_source_data"].shape == (20, 5)
        assert res["target_data"].shape == (15, 5)
        assert res["ot_plan"].shape == (20, 15)
        assert res["cost"].shape == (20, 15)

    def test_plan_marginals(self):
        rng = np.random.RandomState(42)
        n_s, n_t = 20, 15
        src = rng.randn(n_s, 5)
        tgt = rng.randn(n_t, 5) + 1
        res = domain_adaptation_ot(src, tgt, eps=0.05, maxit=1000, tol=1e-8)
        if res["converged"]:
            np.testing.assert_allclose(res["ot_plan"].sum(axis=1), 1.0 / n_s, atol=1e-5)
            np.testing.assert_allclose(res["ot_plan"].sum(axis=0), 1.0 / n_t, atol=1e-5)

    def test_plan_non_negative(self, src_tgt_42):
        src, tgt = src_tgt_42
        res = domain_adaptation_ot(src, tgt, eps=0.1)
        assert np.all(res["ot_plan"] >= 0)

    def test_cost_non_negative(self, src_tgt_42):
        src, tgt = src_tgt_42
        res_sq = domain_adaptation_ot(src, tgt, eps=0.1, cost="sqeuclidean")
        assert np.all(res_sq["cost"] >= 0)
        res_euc = domain_adaptation_ot(src, tgt, eps=0.1, cost="euclidean")
        assert np.all(res_euc["cost"] >= 0)

    def test_different_costs_differ(self, src_tgt_42):
        src, tgt = src_tgt_42
        res_sq = domain_adaptation_ot(src, tgt, eps=0.1, cost="sqeuclidean")
        res_euc = domain_adaptation_ot(src, tgt, eps=0.1, cost="euclidean")
        assert not np.allclose(res_sq["weighted_source_data"], res_euc["weighted_source_data"])

    def test_epsilon_stored(self, src_tgt_42):
        src, tgt = src_tgt_42
        res = domain_adaptation_ot(src, tgt, eps=0.42)
        assert res["epsilon"] == 0.42

    def test_deterministic(self):
        def run():
            rng = np.random.RandomState(123)
            s = rng.randn(20, 5)
            t = rng.randn(20, 5) + 1
            return domain_adaptation_ot(s, t, eps=0.05, maxit=500)
        r1 = run()
        r2 = run()
        np.testing.assert_array_equal(r1["weighted_source_data"], r2["weighted_source_data"])

    def test_reference(self):
        rng = np.random.RandomState(2024)
        src = rng.randn(10, 5)
        tgt = rng.randn(10, 5) + 1
        res = domain_adaptation_ot(src, tgt, eps=0.1, maxit=500, tol=1e-7)
        assert np.all(np.isfinite(res["weighted_source_data"]))
        assert np.all(np.isfinite(res["ot_plan"]))
        assert res["weighted_source_data"].shape == (10, 5)
