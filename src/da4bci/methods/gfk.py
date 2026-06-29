"""Geodesic Flow Kernel (GFK).

Gong et al. (2012), "Geodesic Flow Kernel for Unsupervised Domain Adaptation".
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from da4bci.geometry.spd import orthonormal_complement


def _flow_diagonals(theta):
    """GFK geodesic-flow integral diagonals for principal angles ``theta``.

    Returns (lam1, lam2, lam3) — the closed-form integrals of the geodesic flow,
    with the correct theta -> 0 limits (lam1 -> 2, lam2 -> 0, lam3 -> 0).
    """
    eps = 1e-12
    two_t = 2.0 * theta
    safe = np.where(theta < eps, 1.0, two_t)
    ratio = np.where(theta < eps, 1.0, np.sin(two_t) / safe)
    cross = np.where(theta < eps, 0.0, (np.cos(two_t) - 1.0) / safe)
    return 1.0 + ratio, cross, 1.0 - ratio


def domain_adaptation_gfk(source_data, target_data, dim_subspace=10):
    """GFK: geodesic-flow kernel between the source and target PCA subspaces.

    The kernel ``G`` (p x p) is built from the principal angles between the two
    subspaces; features are then projected by ``G**(1/2)`` so that Euclidean
    inner products in the transformed space equal the GFK kernel.

    Parameters
    ----------
    source_data : ndarray (n_s, p)
    target_data : ndarray (n_t, p)
    dim_subspace : int
        Subspace dimension k (clamped to p). For a meaningful flow use k < p.

    Returns
    -------
    dict with 'weighted_source_data', 'target_data', 'G'.
    """
    source_data = np.asarray(source_data, dtype=float)
    target_data = np.asarray(target_data, dtype=float)
    p = source_data.shape[1]
    k = min(dim_subspace, p)

    # Scaled PCA subspaces (matching R prcomp scale.=TRUE).
    src = StandardScaler().fit_transform(source_data)
    tgt = StandardScaler().fit_transform(target_data)
    _, _, Vt_s = np.linalg.svd(src, full_matrices=False)
    _, _, Vt_t = np.linalg.svd(tgt, full_matrices=False)
    Ps = Vt_s[:k].T                      # (p, k) source subspace
    Pt = Vt_t[:k].T                      # (p, k) target subspace
    Rs = orthonormal_complement(Ps)      # (p, p - k) source complement

    # Principal angles between the source and target subspaces.
    U1, gamma, _ = np.linalg.svd(Ps.T @ Pt)
    theta = np.arccos(np.clip(gamma, -1.0, 1.0))     # (k,)
    lam1, lam2, lam3 = _flow_diagonals(theta)

    PU = Ps @ U1                          # (p, k)
    G = PU @ np.diag(lam1) @ PU.T

    kc = min(k, Rs.shape[1])              # usable complement directions
    if kc > 0:
        U2, _, _ = np.linalg.svd(Rs.T @ Pt)
        RU = Rs @ U2[:, :kc]              # (p, kc)
        PUc = PU[:, :kc]
        L2 = np.diag(lam2[:kc])
        G = G + PUc @ L2 @ RU.T + RU @ L2 @ PUc.T + RU @ np.diag(lam3[:kc]) @ RU.T

    # Symmetric PSD square root: <X G_half, Y G_half> = X G Y^T (the GFK kernel).
    G = 0.5 * (G + G.T)
    vals, vecs = np.linalg.eigh(G)
    G_half = (vecs * np.sqrt(np.maximum(vals, 0.0))) @ vecs.T

    return {
        "weighted_source_data": source_data @ G_half,
        "target_data": target_data @ G_half,
        "G": G,
    }
