"""Geodesic Flow Kernel (GFK)."""

import numpy as np
from sklearn.preprocessing import StandardScaler
from da4bci.geometry.spd import orthonormal_complement


def domain_adaptation_gfk(source_data, target_data, dim_subspace=10):
    """GFK: geodesic interpolation on Grassmann manifold.

    Parameters
    ----------
    source_data : ndarray (n_s, p)
    target_data : ndarray (n_t, p)
    dim_subspace : int

    Returns
    -------
    dict with 'weighted_source_data', 'target_data', 'G'.
    """
    source_data = np.asarray(source_data, dtype=float)
    target_data = np.asarray(target_data, dtype=float)
    p = source_data.shape[1]
    dim_subspace = min(dim_subspace, p)

    # PCA with scaling (matching R's prcomp scale.=TRUE)
    scaler_s = StandardScaler()
    src_scaled = scaler_s.fit_transform(source_data)
    scaler_t = StandardScaler()
    tgt_scaled = scaler_t.fit_transform(target_data)

    _, _, Vt_s = np.linalg.svd(src_scaled, full_matrices=False)
    _, _, Vt_t = np.linalg.svd(tgt_scaled, full_matrices=False)

    Us = Vt_s[:dim_subspace].T  # (p, k)
    Ut = Vt_t[:dim_subspace].T  # (p, k)

    d = p
    k = dim_subspace

    # Orthonormal complements
    Qs = orthonormal_complement(Us)
    Qt = orthonormal_complement(Ut)

    # Construct full bases
    U0 = np.hstack([Us, Qs])  # (d, d)
    U1 = np.hstack([Ut, Qt])  # (d, d)

    M0 = U0.T @ U1
    U_m, S_m, Vt_m = np.linalg.svd(M0, full_matrices=True)

    # Clamp and compute angles
    S_clamped = np.minimum(S_m, 1.0)
    angles = np.arccos(S_clamped)

    # GFK diagonal
    def int_cossin(ti):
        if ti < 1e-12:
            return 1.0
        return (ti - np.sin(ti)) / ti

    g_vec = np.ones(d)
    for i in range(k):
        if i < len(angles):
            g_vec[i] = int_cossin(angles[i])

    diag_g = np.diag(g_vec)

    G = U0 @ U_m @ diag_g @ U_m.T @ U0.T

    source_gfk = source_data @ G
    target_gfk = target_data @ G

    return {
        "weighted_source_data": source_gfk,
        "target_data": target_gfk,
        "G": G,
    }
