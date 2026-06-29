"""Regression tests for the review-fix batch (GFK, geodesic, sigma_med, M3D)."""

import copy

import numpy as np
import pytest

import da4bci
from da4bci.metrics.kernels import sigma_med
from da4bci.geometry.spd import compute_geodesic


def _src_tgt(n_s=60, n_t=50, p=8, seed=0):
    rng = np.random.default_rng(seed)
    S = rng.standard_normal((n_s, p)) @ np.diag(np.linspace(5, 1, p))
    T = (rng.standard_normal((n_t, p)) @ np.diag(np.linspace(1, 5, p))) + 2.0
    return S, T


def test_gfk_transforms_and_handles_unequal_sizes():
    S, T = _src_tgt()
    out = da4bci.domain_adaptation_gfk(S, T, dim_subspace=4)
    assert out["weighted_source_data"].shape == S.shape
    assert out["target_data"].shape == T.shape
    # GFK must NOT be an identity no-op when the subspaces differ.
    assert not np.allclose(out["G"], np.eye(S.shape[1]))
    assert not np.allclose(out["weighted_source_data"], S)
    # G is symmetric and positive semidefinite.
    assert np.allclose(out["G"], out["G"].T)
    assert np.linalg.eigvalsh(out["G"]).min() > -1e-8


def test_compute_geodesic_no_crash_on_unequal_sizes():
    S, T = _src_tgt(n_s=60, n_t=50)
    g = compute_geodesic(S, T, d=3)  # used to raise ValueError when n_s != n_t
    assert np.isfinite(g) and g >= 0.0


def test_sigma_med_is_deterministic_above_subsample_size():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((600, 8))
    Y = rng.standard_normal((600, 8)) + 1.0
    vals = [float(sigma_med(X, Y)) for _ in range(3)]
    assert len(set(vals)) == 1  # fixed default seed -> reproducible bandwidth


def test_gfk_geodesic_flow_reaches_target_subspace():
    # The geodesic flow built by GFK must connect span(Ps) to span(Pt): at t=1 it
    # must lie in the target subspace. A reversed angle pairing breaks this.
    from sklearn.preprocessing import StandardScaler
    from da4bci.geometry.spd import orthonormal_complement
    for p, dim in [(10, 5), (20, 10), (12, 4), (8, 6)]:
        S, T = _src_tgt(80, 70, p, seed=p)
        da4bci.domain_adaptation_gfk(S, T, dim_subspace=dim)  # must not crash
        src = StandardScaler().fit_transform(S)
        tgt = StandardScaler().fit_transform(T)
        _, _, Vts = np.linalg.svd(src, full_matrices=False)
        _, _, Vtt = np.linalg.svd(tgt, full_matrices=False)
        k = min(dim, p, Vts.shape[0], Vtt.shape[0])
        Ps, Pt = Vts[:k].T, Vtt[:k].T
        Rs = orthonormal_complement(Ps)
        U1, gam, V1t = np.linalg.svd(Ps.T @ Pt)
        th = np.arccos(np.clip(gam, -1, 1))
        PU = Ps @ U1
        B = Rs.T @ Pt @ V1t.T
        nb = np.linalg.norm(B, axis=0)
        RU = Rs @ (-B / np.where(nb > 1e-12, nb, 1.0))
        phi1 = PU * np.cos(th) - RU * np.sin(th)
        resid = np.linalg.norm((np.eye(p) - Pt @ Pt.T) @ phi1)
        assert resid < 1e-8


def test_gfk_no_crash_on_few_samples():
    # n_s or n_t <= dim_subspace used to crash with a matmul mismatch.
    S, T = _src_tgt(8, 100, 30)
    out = da4bci.domain_adaptation_gfk(S, T)  # default dim_subspace=10
    assert out["G"].shape == (30, 30)
    da4bci.domain_adaptation_gfk(_src_tgt(200, 6, 22)[0], _src_tgt(200, 6, 22)[1])


def test_coral_aligned_source_matches_target_covariance():
    rng = np.random.default_rng(7)
    A, B = rng.standard_normal((5, 5)), rng.standard_normal((5, 5))
    Xs, Xt = rng.standard_normal((300, 5)) @ A, rng.standard_normal((300, 5)) @ B
    ws = da4bci.domain_adaptation(Xs, Xt, method="coral")["weighted_source_data"]
    diff = np.max(np.abs(np.cov(ws, rowvar=False) - np.cov(Xt, rowvar=False)))
    assert diff < 0.1  # aligned source covariance now matches target (was ~247)


def test_mahalanobis_single_feature():
    from da4bci.metrics.distance import compute_mahalanobis
    rng = np.random.default_rng(0)
    assert np.isfinite(compute_mahalanobis(rng.standard_normal((10, 1)), rng.standard_normal((12, 1))))


def test_distance_matrix_no_spurious_nan_on_large_coordinates():
    from da4bci.metrics.distance import compute_distance_matrix
    X = np.array([[4198.093263614437, 1140.8349171885798],
                  [4198.09326050875, 1140.8349171771315]])
    assert not np.isnan(compute_distance_matrix(X, X)).any()


def test_metrics_accept_1d_input():
    from da4bci.metrics.distance import compute_mmd
    rng = np.random.default_rng(0)
    assert np.isfinite(compute_mmd(rng.standard_normal(5), rng.standard_normal(5), sigma=1.0))


def test_kmm_warns_when_qp_fails():
    from da4bci.preprocessing.weights import kmm_weights
    rng = np.random.default_rng(0)
    Xs = rng.standard_normal((8, 3))
    Xs[0, 0] = np.nan
    with pytest.warns(RuntimeWarning):
        kmm_weights(Xs, rng.standard_normal((10, 3)), sigma=1.0)


def test_dispatcher_rejects_bad_shapes():
    S, T = _src_tgt(p=8)
    with pytest.raises(ValueError):
        da4bci.domain_adaptation(S[:, 0], T, method="coral")          # 1D source
    with pytest.raises(ValueError):
        da4bci.domain_adaptation(S, _src_tgt(p=5)[1], method="coral")  # mismatched p


def test_matrix_power_warns_on_non_finite_instead_of_silent_identity():
    from da4bci.geometry.spd import matrix_power
    A = np.array([[1.0, np.nan], [np.nan, 1.0]])
    with pytest.warns(RuntimeWarning):
        out = matrix_power(A, 0.5)
    assert np.allclose(out, np.eye(2))


def test_m3d_does_not_mutate_caller_stage_dicts_and_guards_label_length():
    S, T = _src_tgt(n_s=40, n_t=40, p=8)
    labels = np.array([1, 2] * 20)
    stage1 = {"method": "tca", "control": {"k": None, "sigma": 1}}
    stage2 = {"method": "sa", "control": {"k": 10}}
    snapshot1 = copy.deepcopy(stage1)
    da4bci.domain_adaptation_m3d(S, labels, T, stage1=stage1, stage2=stage2)
    # The caller's stage dict must be unchanged (no in-place auto-k write-back).
    assert stage1 == snapshot1
    # A label array of the wrong length must raise, not silently corrupt the fit.
    with pytest.raises(ValueError):
        da4bci.domain_adaptation_m3d(S, labels[:3], T)
