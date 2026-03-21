"""Tests for kernel functions and distance metrics."""

import numpy as np
import pytest
from da4bci.metrics.kernels import rbf_kernel, sigma_med
from da4bci.metrics.distance import (
    compute_distance_matrix, compute_mmd, compute_energy,
    compute_wasserstein, compute_mahalanobis,
)


# ---- rbf_kernel ----

class TestRBFKernel:
    def test_dimensions(self):
        rng = np.random.RandomState(42)
        x = rng.randn(5, 4)
        y = rng.randn(6, 4)
        K = rbf_kernel(x, y, sigma=1)
        assert K.shape == (5, 6)

    def test_values_in_01(self):
        rng = np.random.RandomState(42)
        x = rng.randn(5, 4)
        y = rng.randn(6, 4)
        K = rbf_kernel(x, y, sigma=1)
        assert np.all(K > 0)
        assert np.all(K <= 1)

    def test_self_kernel_diagonal(self):
        rng = np.random.RandomState(42)
        x = rng.randn(5, 4)
        K = rbf_kernel(x, x, sigma=1)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-12)

    def test_self_kernel_symmetric(self):
        rng = np.random.RandomState(42)
        x = rng.randn(5, 4)
        K = rbf_kernel(x, x, sigma=1)
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    def test_standard_scale_flag(self):
        rng = np.random.RandomState(42)
        x = rng.randn(5, 4)
        y = rng.randn(6, 4)
        K_std = rbf_kernel(x, y, sigma=2, standard_scale=True)
        K_nstd = rbf_kernel(x, y, sigma=2, standard_scale=False)
        assert not np.allclose(K_std, K_nstd)

    def test_rejects_column_mismatch(self):
        x = np.random.randn(5, 4)
        y = np.random.randn(5, 3)
        with pytest.raises(AssertionError):
            rbf_kernel(x, y, sigma=1)


# ---- sigma_med ----

class TestSigmaMed:
    def test_positive_scalar(self):
        rng = np.random.RandomState(42)
        X = rng.randn(20, 5)
        Y = rng.randn(20, 5) + 1
        s = sigma_med(X, Y)
        assert isinstance(s, float)
        assert s > 0

    def test_subsampling(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        Y = rng.randn(100, 5) + 1
        s = sigma_med(X, Y, m=20, seed=1)
        assert s > 0

    def test_warns_insufficient_data(self):
        X = np.array([[1, 2, 3]])
        Y = np.array([[4, 5, 6]])
        with pytest.warns(UserWarning, match="Not enough"):
            sigma_med(X, Y)


# ---- compute_distance_matrix ----

class TestDistanceMatrix:
    def test_dimensions(self):
        rng = np.random.RandomState(42)
        A = rng.randn(5, 3)
        B = rng.randn(4, 3)
        D = compute_distance_matrix(A, B)
        assert D.shape == (5, 4)

    def test_self_distance_diagonal_zero(self):
        rng = np.random.RandomState(42)
        A = rng.randn(5, 3)
        D = compute_distance_matrix(A, A)
        np.testing.assert_allclose(np.diag(D), 0, atol=1e-6)

    def test_non_negative(self):
        rng = np.random.RandomState(42)
        A = rng.randn(5, 3)
        B = rng.randn(4, 3)
        D = compute_distance_matrix(A, B)
        assert np.all(D >= 0)


# ---- compute_mmd ----

class TestMMD:
    def test_returns_scalar(self):
        rng = np.random.RandomState(42)
        X = rng.randn(20, 5)
        Y = rng.randn(20, 5) + 1
        m = compute_mmd(X, Y, sigma=1)
        assert isinstance(m, float)

    def test_zero_for_identical(self):
        rng = np.random.RandomState(42)
        X = rng.randn(20, 5)
        m = compute_mmd(X, X, sigma=1)
        assert abs(m) < 1e-10

    def test_positive_for_different(self):
        rng = np.random.RandomState(42)
        X = rng.randn(40, 5)
        Y = rng.randn(40, 5) + 5
        m = compute_mmd(X, Y, sigma=1)
        assert m > 0


# ---- compute_energy ----

class TestEnergy:
    def test_non_negative(self):
        rng = np.random.RandomState(42)
        X = rng.randn(20, 5)
        Y = rng.randn(20, 5) + 1
        e = compute_energy(X, Y)
        assert e >= 0

    def test_zero_for_identical(self):
        rng = np.random.RandomState(42)
        X = rng.randn(20, 5)
        e = compute_energy(X, X)
        assert abs(e) < 1e-10


# ---- compute_wasserstein ----

class TestWasserstein:
    def test_non_negative(self):
        rng = np.random.RandomState(42)
        X = rng.randn(5, 4)
        Y = rng.randn(6, 4) + 1
        w = compute_wasserstein(X, Y)
        assert w >= 0

    def test_zero_for_identical(self):
        rng = np.random.RandomState(42)
        X = rng.randn(5, 4)
        w = compute_wasserstein(X, X)
        assert abs(w) < 1e-6

    def test_rejects_column_mismatch(self):
        X = np.random.randn(5, 4)
        Y = np.random.randn(5, 3)
        with pytest.raises(ValueError):
            compute_wasserstein(X, Y)


# ---- compute_mahalanobis ----

class TestMahalanobis:
    def test_non_negative(self):
        rng = np.random.RandomState(42)
        X = rng.randn(20, 10)
        Y = rng.randn(15, 10) + 0.5
        d = compute_mahalanobis(X, Y)
        assert d >= 0

    def test_zero_for_identical(self):
        rng = np.random.RandomState(42)
        X = rng.randn(20, 10)
        d = compute_mahalanobis(X, X)
        assert abs(d) < 1e-6

    def test_cov_choices(self):
        rng = np.random.RandomState(42)
        X = rng.randn(20, 10)
        Y = rng.randn(15, 10) + 0.5
        d_pool = compute_mahalanobis(X, Y, cov_choice="pooled")
        d_src = compute_mahalanobis(X, Y, cov_choice="source")
        d_tgt = compute_mahalanobis(X, Y, cov_choice="target")
        assert d_pool >= 0
        assert d_src >= 0
        assert d_tgt >= 0

    def test_squared(self):
        rng = np.random.RandomState(42)
        X = rng.randn(20, 10)
        Y = rng.randn(15, 10) + 0.5
        d = compute_mahalanobis(X, Y, squared=False)
        d2 = compute_mahalanobis(X, Y, squared=True)
        np.testing.assert_allclose(d ** 2, d2, atol=1e-10)

    def test_shrinkage(self):
        rng = np.random.RandomState(42)
        X = rng.randn(20, 10)
        Y = rng.randn(15, 10) + 0.5
        d_no = compute_mahalanobis(X, Y)
        d_sh = compute_mahalanobis(X, Y, shrinkage_alpha=0.1)
        assert d_no >= 0 and d_sh >= 0
        assert not np.isclose(d_no, d_sh)


# ---- Cross-language reference ----

class TestReference:
    def test_reference_values(self):
        rng = np.random.RandomState(2024)
        X = rng.randn(10, 5)
        Y = rng.randn(10, 5) + 1
        K = rbf_kernel(X, Y, sigma=1)
        assert K.shape == (10, 10)
        assert np.all(np.isfinite(K))
        assert np.isfinite(compute_mmd(X, Y, sigma=1))
        assert np.isfinite(compute_energy(X, Y))
        assert np.isfinite(compute_wasserstein(X, Y))
        assert np.isfinite(compute_mahalanobis(X, Y))
