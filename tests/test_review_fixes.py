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
