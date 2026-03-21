"""Tests for utility functions."""

import numpy as np
from da4bci.preprocessing.label_shift import label_shift_em
from da4bci.preprocessing.alignment import euclidean_alignment
from da4bci.metrics.evaluation import proxy_a_distance, evaluate_shift
from da4bci.detection.page_hinkley import ph_init, ph_update


class TestLabelShiftEM:
    def test_structure(self):
        rng = np.random.RandomState(42)
        P = rng.rand(10, 5)
        P = P / P.sum(axis=1, keepdims=True)
        out = label_shift_em(P, np.ones(5) / 5)
        assert isinstance(out, dict)
        assert "pi_t" in out
        assert "P_adj" in out
        assert "iter" in out

    def test_priors_sum_to_one(self):
        rng = np.random.RandomState(42)
        P = rng.rand(10, 5)
        P = P / P.sum(axis=1, keepdims=True)
        out = label_shift_em(P, np.ones(5) / 5)
        np.testing.assert_allclose(out["pi_t"].sum(), 1.0, atol=1e-10)

    def test_adjusted_posteriors_valid(self):
        rng = np.random.RandomState(42)
        P = rng.rand(10, 5)
        P = P / P.sum(axis=1, keepdims=True)
        out = label_shift_em(P, np.ones(5) / 5)
        assert np.all(out["P_adj"] >= 0)
        np.testing.assert_allclose(out["P_adj"].sum(axis=1), 1.0, atol=1e-10)


class TestEuclideanAlignment:
    def test_trial_count(self):
        rng = np.random.RandomState(42)
        trials = [rng.randn(32, 100) for _ in range(5)]
        out = euclidean_alignment(trials)
        assert len(out) == 5

    def test_dimensions(self):
        rng = np.random.RandomState(42)
        trials = [rng.randn(32, 100) for _ in range(5)]
        out = euclidean_alignment(trials)
        for t in out:
            assert t.shape == (32, 100)

    def test_whitens_mean_covariance(self):
        rng = np.random.RandomState(42)
        trials = [rng.randn(8, 50) for _ in range(10)]
        out = euclidean_alignment(trials)
        covs = []
        for X in out:
            S = X @ X.T / max(1, X.shape[1] - 1)
            covs.append(S)
        R = np.mean(covs, axis=0)
        np.testing.assert_allclose(R, np.eye(8), atol=0.3)


class TestProxyADistance:
    def test_returns_pad_err(self):
        rng = np.random.RandomState(42)
        Xs = rng.randn(20, 10)
        Xt = rng.randn(20, 10) + 1
        out = proxy_a_distance(Xs, Xt, seed=42)
        assert isinstance(out, dict)
        assert "pad" in out
        assert "err" in out

    def test_pad_range(self):
        rng = np.random.RandomState(42)
        Xs = rng.randn(20, 10)
        Xt = rng.randn(20, 10) + 1
        out = proxy_a_distance(Xs, Xt, seed=42)
        assert -2 <= out["pad"] <= 2

    def test_err_range(self):
        rng = np.random.RandomState(42)
        Xs = rng.randn(20, 10)
        Xt = rng.randn(20, 10) + 1
        out = proxy_a_distance(Xs, Xt, seed=42)
        assert 0 <= out["err"] <= 0.5

    def test_high_for_very_different(self):
        rng = np.random.RandomState(42)
        Xs = rng.randn(20, 10)
        Xt = rng.randn(20, 10) + 10
        out = proxy_a_distance(Xs, Xt, seed=42)
        assert out["pad"] > 1.0


class TestEvaluateShift:
    def test_returns_dataframe_like_dict(self):
        rng = np.random.RandomState(42)
        A = rng.randn(20, 10)
        B = rng.randn(20, 10) + 2
        As = rng.randn(20, 10) + 1
        Bs = rng.randn(20, 10) + 1.5
        result = evaluate_shift(A, B, As, Bs)
        # Matches R's data.frame return: dict with Metric, Before, After
        assert isinstance(result, dict)
        assert "Metric" in result
        assert "Before" in result
        assert "After" in result
        assert len(result["Metric"]) == 2
        assert "MMD" in result["Metric"]
        assert "Wasserstein" in result["Metric"]


class TestPageHinkley:
    def test_init_structure(self):
        s = ph_init()
        assert isinstance(s, dict)
        for key in ["mean", "cum", "min_cum", "delta", "lambda", "alpha"]:
            assert key in s
        assert s["mean"] == 0
        assert s["cum"] == 0

    def test_custom_params(self):
        s = ph_init(delta=0.01, lambda_=100, alpha=0.99)
        assert s["delta"] == 0.01
        assert s["lambda"] == 100
        assert s["alpha"] == 0.99

    def test_update_returns_state_change(self):
        s = ph_init()
        out = ph_update(s, 1.0)
        assert isinstance(out, dict)
        assert "state" in out
        assert "change" in out
        assert isinstance(out["change"], bool)

    def test_detects_large_shift(self):
        s = ph_init(delta=0.005, lambda_=5, alpha=0.999)
        for z in np.random.RandomState(1).normal(0, 0.1, 100):
            out = ph_update(s, z)
            s = out["state"]
        for z in np.random.RandomState(2).normal(10, 0.1, 50):
            out = ph_update(s, z)
            s = out["state"]
        assert out["change"] is True

    def test_no_false_alarm(self):
        s = ph_init(delta=0.005, lambda_=50, alpha=0.999)
        any_change = False
        for z in np.random.RandomState(1).normal(0, 0.1, 200):
            out = ph_update(s, z)
            s = out["state"]
            if out["change"]:
                any_change = True
        assert not any_change
