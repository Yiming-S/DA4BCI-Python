"""Subspace Alignment (SA)."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def domain_adaptation_sa(source_data, target_data, k=10):
    """Subspace Alignment via PCA.

    Parameters
    ----------
    source_data : ndarray (n_s, p)
    target_data : ndarray (n_t, p)
    k : int, number of principal components.

    Returns
    -------
    dict with 'weighted_source_data', 'target_data', 'eigenvalue' (W matrix).
    """
    source_data = np.asarray(source_data, dtype=float)
    target_data = np.asarray(target_data, dtype=float)
    k = min(k, source_data.shape[1], target_data.shape[1])

    # R's prcomp(X, scale.=TRUE) standardizes columns then does SVD
    # PCA rotation = loadings (p x k), x = scores (n x k)
    scaler_s = StandardScaler()
    src_scaled = scaler_s.fit_transform(source_data)
    scaler_t = StandardScaler()
    tgt_scaled = scaler_t.fit_transform(target_data)

    # SVD-based PCA to match R's prcomp
    # prcomp returns rotation (p x k) and x = centered_scaled_data @ rotation
    U_s, S_s, Vt_s = np.linalg.svd(src_scaled, full_matrices=False)
    U_t, S_t, Vt_t = np.linalg.svd(tgt_scaled, full_matrices=False)

    Z_s = Vt_s[:k].T  # (p, k) loadings
    Z_t = Vt_t[:k].T  # (p, k) loadings

    # Scores
    scores_s = src_scaled @ Z_s  # (n_s, k)
    scores_t = tgt_scaled @ Z_t  # (n_t, k)

    # Alignment matrix
    W = Z_s.T @ Z_t  # (k, k)

    # Project source into target subspace
    weighted_source = scores_s @ W

    return {
        "weighted_source_data": weighted_source,
        "target_data": scores_t,
        "eigenvalue": W,
    }
