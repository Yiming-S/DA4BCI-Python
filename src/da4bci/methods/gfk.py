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
        Requested subspace dimension; clamped to the usable rank of each domain
        so few-trial inputs do not crash. For a meaningful flow use k < p.

    Returns
    -------
    dict with 'weighted_source_data', 'target_data', 'G'.
    """
    source_data = np.asarray(source_data, dtype=float)
    target_data = np.asarray(target_data, dtype=float)
    p = source_data.shape[1]

    # Scaled PCA subspaces (matching R prcomp scale.=TRUE).
    src = StandardScaler().fit_transform(source_data)
    tgt = StandardScaler().fit_transform(target_data)
    _, _, Vt_s = np.linalg.svd(src, full_matrices=False)
    _, _, Vt_t = np.linalg.svd(tgt, full_matrices=False)
    # Clamp k to a dimension both domains actually span (avoids shape mismatches
    # when a domain has <= dim_subspace samples).
    k = min(dim_subspace, p, Vt_s.shape[0], Vt_t.shape[0])
    Ps = Vt_s[:k].T                      # (p, k) source subspace
    Pt = Vt_t[:k].T                      # (p, k) target subspace
    Rs = orthonormal_complement(Ps)      # (p, p - k) source complement

    # Principal angles between the source and target subspaces. The complement
    # rotation U2 MUST be tied to the same right-singular basis V1 as U1 (so the
    # angles are paired consistently); a separate SVD of Rs.T@Pt would order the
    # complement directions by descending sine, reversing the pairing.
    U1, gamma, V1t = np.linalg.svd(Ps.T @ Pt)
    V1 = V1t.T
    theta = np.arccos(np.clip(gamma, -1.0, 1.0))     # (k,)
    lam1, lam2, lam3 = _flow_diagonals(theta)

    PU = Ps @ U1                          # (p, k)
    G = PU @ np.diag(lam1) @ PU.T

    if Rs.shape[1] > 0:
        # Columns of B = Rs.T @ Pt @ V1 are (-sin theta_j) * u2_j, matched to
        # angle j. Normalize per column to recover the complement directions.
        B = Rs.T @ Pt @ V1                # (p - k, k)
        norms = np.linalg.norm(B, axis=0)
        U2 = -B / np.where(norms > 1e-12, norms, 1.0)
        RU = Rs @ U2                      # (p, k), paired column-for-column with PU
        L2, L3 = np.diag(lam2), np.diag(lam3)
        G = G + PU @ L2 @ RU.T + RU @ L2 @ PU.T + RU @ L3 @ RU.T

    # Symmetric PSD square root: <X G_half, Y G_half> = X G Y^T (the GFK kernel).
    G = 0.5 * (G + G.T)
    vals, vecs = np.linalg.eigh(G)
    G_half = (vecs * np.sqrt(np.maximum(vals, 0.0))) @ vecs.T

    return {
        "weighted_source_data": source_data @ G_half,
        "target_data": target_data @ G_half,
        "G": G,
    }
