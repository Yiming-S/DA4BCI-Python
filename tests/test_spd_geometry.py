"""Tests for SPD geometry functions."""

import numpy as np
import pytest
from da4bci.geometry.spd import (
    orthonormal_complement, LW_covariance, matrix_power,
    riemannian_mean, log_map, exp_map, compute_geodesic,
)


class TestOrthonormalComplement:
    def test_dimensions(self):
        Q, _ = np.linalg.qr(np.random.RandomState(42).randn(4, 3))
        U = Q[:, :3]
        Uperp = orthonormal_complement(U)
        assert Uperp.shape == (4, 1)

    def test_orthogonal_to_U(self):
        rng = np.random.RandomState(42)
        Q, _ = np.linalg.qr(rng.randn(5, 2))
        U = Q[:, :2]
        Uperp = orthonormal_complement(U)
        cross = U.T @ Uperp
        np.testing.assert_allclose(cross, 0, atol=1e-10)

    def test_unit_columns(self):
        rng = np.random.RandomState(42)
        Q, _ = np.linalg.qr(rng.randn(5, 2))
        U = Q[:, :2]
        Uperp = orthonormal_complement(U)
        for j in range(Uperp.shape[1]):
            np.testing.assert_allclose(np.sum(Uperp[:, j] ** 2), 1, atol=1e-10)

    def test_full_rank_empty(self):
        U = np.eye(5)
        Uperp = orthonormal_complement(U)
        assert Uperp.shape[1] == 0

    def test_empty_columns_identity(self):
        U = np.zeros((4, 0))
        Uperp = orthonormal_complement(U)
        assert Uperp.shape == (4, 4)


class TestLWCovariance:
    def test_shape(self):
        rng = np.random.RandomState(42)
        X = rng.randn(20, 10)
        S = LW_covariance(X)
        assert S.shape == (10, 10)

    def test_symmetric(self):
        rng = np.random.RandomState(42)
        X = rng.randn(20, 10)
        S = LW_covariance(X)
        np.testing.assert_allclose(S, S.T, atol=1e-12)

    def test_positive_definite(self):
        rng = np.random.RandomState(42)
        X = rng.randn(20, 10)
        S = LW_covariance(X)
        vals = np.linalg.eigvalsh(S)
        assert np.all(vals > 0)


class TestMatrixPower:
    def test_identity_cases(self, spd_pair_42):
        A, _ = spd_pair_42
        np.testing.assert_allclose(matrix_power(A, 1), (A + A.T) / 2, atol=1e-10)
        np.testing.assert_allclose(matrix_power(A, 0), np.eye(5), atol=1e-10)

    def test_inverse(self, spd_pair_42):
        A, _ = spd_pair_42
        A_inv = matrix_power(A, -1)
        np.testing.assert_allclose(A @ A_inv, np.eye(5), atol=1e-6)

    def test_sqrt(self, spd_pair_42):
        A, _ = spd_pair_42
        A_half = matrix_power(A, 0.5)
        np.testing.assert_allclose(A_half @ A_half, (A + A.T) / 2, atol=1e-6)

    def test_neg_sqrt(self, spd_pair_42):
        A, _ = spd_pair_42
        A_neg_half = matrix_power(A, -0.5)
        A_sym = (A + A.T) / 2
        result = A_neg_half @ A_sym @ A_neg_half
        np.testing.assert_allclose(result, np.eye(5), atol=1e-5)


class TestRiemannianMean:
    def test_identical_matrices(self, spd_pair_42):
        A, _ = spd_pair_42
        arr = np.stack([A, A, A], axis=-1)
        R = riemannian_mean(arr)
        np.testing.assert_allclose(R, A, atol=1e-4)

    def test_returns_spd(self):
        rng = np.random.RandomState(42)
        # Generate well-conditioned SPD matrices
        mats = []
        for _ in range(5):
            M = rng.randn(4, 4)
            mats.append(M.T @ M + 5 * np.eye(4))
        arr = np.stack(mats, axis=-1)
        R = riemannian_mean(arr)
        np.testing.assert_allclose(R, R.T, atol=1e-6)
        assert np.all(np.linalg.eigvalsh(R) > 0)

    def test_list_input(self):
        rng = np.random.RandomState(42)
        mats = [rng.randn(4, 4).T @ rng.randn(4, 4) + np.eye(4) for _ in range(4)]
        R = riemannian_mean(mats)
        assert R.shape == (4, 4)


class TestLogExpMap:
    def test_roundtrip(self, spd_pair_42):
        P, Q = spd_pair_42
        L = log_map(P, Q)
        Q_rec = exp_map(P, L)
        np.testing.assert_allclose(Q_rec, Q, atol=1e-6)

    def test_log_at_identity(self, spd_pair_42):
        _, Q = spd_pair_42
        I = np.eye(5)
        L = log_map(I, Q)
        vals, vecs = np.linalg.eigh(L)
        Q_from_L = vecs @ np.diag(np.exp(vals)) @ vecs.T
        np.testing.assert_allclose(Q_from_L, Q, atol=1e-6)

    def test_precomputed_sqrt(self, spd_pair_42):
        P, Q = spd_pair_42
        vals, vecs = np.linalg.eigh(P)
        P_dict = {
            "sqrt": vecs @ np.diag(np.sqrt(vals)) @ vecs.T,
            "inv_sqrt": vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T,
        }
        L_mat = log_map(P, Q)
        L_dict = log_map(P_dict, Q)
        np.testing.assert_allclose(L_mat, L_dict, atol=1e-10)


class TestGeodesic:
    def test_non_negative(self):
        rng = np.random.RandomState(42)
        X = rng.randn(60, 5)
        Y = rng.randn(60, 5) + 0.5
        g = compute_geodesic(X, Y)
        assert g >= 0

    def test_zero_for_identical(self):
        rng = np.random.RandomState(42)
        X = rng.randn(60, 5)
        g = compute_geodesic(X, X)
        assert abs(g) < 1e-6

    def test_d_parameter(self):
        rng = np.random.RandomState(42)
        X = rng.randn(60, 5)
        Y = rng.randn(60, 5) + 0.5
        g3 = compute_geodesic(X, Y, d=3)
        g5 = compute_geodesic(X, Y, d=5)
        assert g3 >= 0
        assert g5 >= 0


class TestSPDReference:
    def test_reference_values(self):
        rng = np.random.RandomState(2024)
        M = rng.randn(5, 5)
        A = M.T @ M + np.eye(5)
        M2 = rng.randn(5, 5)
        B = M2.T @ M2 + np.eye(5)
        Ahalf = matrix_power(A, 0.5)
        assert np.all(np.isfinite(Ahalf))
        L = log_map(A, B)
        B_rec = exp_map(A, L)
        np.testing.assert_allclose(B_rec, B, atol=1e-6)
        X = rng.randn(20, 5)
        Slw = LW_covariance(X)
        assert Slw.shape == (5, 5)
        assert np.all(np.isfinite(Slw))
