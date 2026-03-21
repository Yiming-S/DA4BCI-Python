"""Tests verifying all six review findings are fixed."""

import numpy as np
import pytest


class TestMIDAEigenOrdering:
    """Finding 1: MIDA must use correct eigenvalue ordering."""

    def test_output_scale_reasonable(self):
        """MIDA output values should be in a reasonable range, not thousands."""
        from da4bci.methods.mida import domain_adaptation_mida
        rng = np.random.RandomState(42)
        src = rng.randn(20, 5)
        tgt = rng.randn(20, 5) + 2
        res = domain_adaptation_mida(src, tgt, k=3, sigma=1, mu=0.1, maximize=True)
        # If eigenvalue ordering is wrong, values explode to thousands
        assert np.all(np.abs(res["weighted_source_data"]) < 100), \
            "MIDA output values suspiciously large — eigenvalue ordering may be wrong"

    def test_max_min_produce_different_subspaces(self):
        from da4bci.methods.mida import domain_adaptation_mida
        rng = np.random.RandomState(42)
        src = rng.randn(30, 5)
        tgt = rng.randn(30, 5) + 2
        res_max = domain_adaptation_mida(src, tgt, k=3, sigma=1, mu=0.1, maximize=True)
        res_min = domain_adaptation_mida(src, tgt, k=3, sigma=1, mu=0.1, maximize=False)
        assert not np.allclose(res_max["weighted_source_data"],
                               res_min["weighted_source_data"])


class TestM3DDimension:
    """Finding 2: M3D output dimension must match R when using defaults."""

    def test_auto_dim_subspace_always_set(self):
        """stage2 dim_subspace is auto-set even when key is absent from control."""
        from da4bci.methods.m3d import domain_adaptation_m3d
        rng = np.random.RandomState(42)
        src = rng.randn(80, 8)
        tgt = rng.randn(80, 8) + 1
        labels = rng.choice([1, 2], size=80)

        # stage2 control has 'k' but NOT 'dim_subspace' —
        # Python must still set dim_subspace automatically.
        res = domain_adaptation_m3d(
            src, labels, tgt,
            stage1={"method": "tca", "control": {"k": 5, "sigma": 1}},
            stage2={"method": "sa", "control": {"k": 5}},
            l_iter=2, max_dim=8,
        )
        # R would output ncol = min(stage1$control$k, max_dim) = min(5, 8) = 5
        assert res["weighted_source_data"].shape[1] == 5
        assert res["target_data"].shape[1] == 5

    def test_auto_dim_subspace_default_stages(self):
        """With fully default stages, dim is auto-determined."""
        from da4bci.methods.m3d import domain_adaptation_m3d
        rng = np.random.RandomState(42)
        src = rng.randn(80, 8)
        tgt = rng.randn(80, 8) + 1
        labels = rng.choice([1, 2], size=80)
        res = domain_adaptation_m3d(src, labels, tgt, l_iter=2, max_dim=8)
        # Both source and target columns should be equal
        assert res["weighted_source_data"].shape[1] == res["target_data"].shape[1]
        # And the dimension should be <= max_dim
        assert res["weighted_source_data"].shape[1] <= 8


class TestWassersteinImportRobustness:
    """Finding 3: compute_wasserstein should not crash if POT import fails."""

    def test_wasserstein_returns_finite(self):
        from da4bci.metrics.distance import compute_wasserstein
        rng = np.random.RandomState(42)
        X = rng.randn(10, 3)
        Y = rng.randn(10, 3) + 1
        w = compute_wasserstein(X, Y)
        assert np.isfinite(w)
        assert w >= 0


class TestAPICompatibility:
    """Finding 4: distanceSummary alias must exist."""

    def test_distanceSummary_alias(self):
        import da4bci
        assert hasattr(da4bci, "distanceSummary")
        assert da4bci.distanceSummary is da4bci.distance_summary

    def test_distance_summary_format_table(self):
        from da4bci.metrics.evaluation import distance_summary
        rng = np.random.RandomState(42)
        X = rng.randn(20, 5)
        Y = rng.randn(20, 5) + 1
        out = distance_summary(X, Y, include=["MMD2", "Energy"], format="table")
        assert "Metric" in out
        assert "Value" in out
        assert "sigma_used" in out
        assert len(out["Metric"]) == 2


class TestParameterNaming:
    """Finding 5: max/lambda parameter compat through unified interface."""

    def test_unified_interface_max_param(self):
        """control={'max': False} should work in the unified interface."""
        from da4bci.methods import domain_adaptation
        rng = np.random.RandomState(42)
        src = rng.randn(20, 5)
        tgt = rng.randn(20, 5) + 2
        # This should NOT raise TypeError
        res = domain_adaptation(src, tgt, method="mida",
                                control={"k": 3, "sigma": 1, "mu": 0.1, "max": False})
        assert "weighted_source_data" in res

    def test_ph_init_lambda_kwarg(self):
        """ph_init should accept 'lambda' as a keyword argument."""
        from da4bci.detection.page_hinkley import ph_init
        # Using **dict to pass 'lambda' (Python keyword)
        s = ph_init(**{"lambda": 100})
        assert s["lambda"] == 100

    def test_ph_init_lambda_underscore(self):
        """ph_init should accept lambda_ (Python-friendly name)."""
        from da4bci.detection.page_hinkley import ph_init
        s = ph_init(lambda_=200)
        assert s["lambda"] == 200


class TestReturnStructures:
    """Finding 6: Return structures must match R's contracts."""

    def test_evaluate_shift_dataframe_like(self):
        from da4bci.metrics.evaluation import evaluate_shift
        rng = np.random.RandomState(42)
        A = rng.randn(20, 5)
        B = rng.randn(20, 5) + 2
        As = rng.randn(20, 5) + 1
        Bs = rng.randn(20, 5) + 1.5
        result = evaluate_shift(A, B, As, Bs)
        # R returns data.frame with Metric, Before, After columns
        assert isinstance(result, dict)
        assert result["Metric"] == ["MMD", "Wasserstein"]
        assert len(result["Before"]) == 2
        assert len(result["After"]) == 2

    def test_plot_returns_p1_p2(self):
        from da4bci.plotting import plot_data_comparison
        rng = np.random.RandomState(42)
        src = rng.randn(20, 5)
        tgt = rng.randn(20, 5) + 1
        Z_s = rng.randn(20, 5) + 0.5
        Z_t = rng.randn(20, 5) + 0.5
        result = plot_data_comparison(src, tgt, Z_s, Z_t)
        # R returns list(p1=..., p2=...)
        assert "p1" in result
        assert "p2" in result

    def test_plot_only_before(self):
        from da4bci.plotting import plot_data_comparison
        rng = np.random.RandomState(42)
        src = rng.randn(20, 5)
        tgt = rng.randn(20, 5) + 1
        result = plot_data_comparison(src, tgt)
        assert "p1" in result
        assert "p2" not in result

    def test_distance_summary_list_format(self):
        from da4bci.metrics.evaluation import distance_summary
        rng = np.random.RandomState(42)
        X = rng.randn(20, 5)
        Y = rng.randn(20, 5) + 1
        out = distance_summary(X, Y, include=["MMD2", "Energy"], format="list")
        assert isinstance(out, dict)
        assert "MMD2" in out
        assert "Energy" in out
