"""Euclidean alignment for EEG trials."""

import numpy as np


def euclidean_alignment(trials, target_R=None):
    """Whiten a domain to identity (or recolor to target_R).

    Parameters
    ----------
    trials : list of ndarray (channels, samples)
    target_R : ndarray (channels, channels) or None
        If None, align to identity.

    Returns
    -------
    list of aligned trial matrices.
    """
    assert isinstance(trials, list) and len(trials) > 0

    covs = []
    for X in trials:
        X = np.asarray(X, dtype=float)
        S = X @ X.T
        S = S / max(1, X.shape[1] - 1)
        covs.append(S)

    R = np.mean(covs, axis=0)

    def mat_pow(A, p):
        A = (A + A.T) / 2
        vals, vecs = np.linalg.eigh(A)
        vals = np.maximum(vals, 1e-12) ** p
        return vecs @ np.diag(vals) @ vecs.T

    Rm12 = mat_pow(R, -0.5)
    L = np.eye(R.shape[0]) if target_R is None else mat_pow(target_R, 0.5)

    return [L @ Rm12 @ np.asarray(X, dtype=float) for X in trials]
