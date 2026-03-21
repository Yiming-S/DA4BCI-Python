"""Maximum/Minimum Independence Domain Adaptation (MIDA)."""

import numpy as np
from scipy.linalg import eigh, eig


def domain_adaptation_mida(source_data, target_data, k=10, sigma=1, mu=0.1,
                           maximize=True):
    """MIDA: HSIC-based domain adaptation.

    Parameters
    ----------
    source_data : ndarray (n_s, p)
    target_data : ndarray (n_t, p)
    k : int, components to keep.
    sigma : float, RBF bandwidth.
    mu : float, regularization.
    maximize : bool, maximize (True) or minimize (False) domain dependence.
        Corresponds to R's ``max`` parameter.

    Returns
    -------
    dict with 'weighted_source_data', 'target_data', 'eigenvalue'.
    """
    source_data = np.asarray(source_data, dtype=float)
    target_data = np.asarray(target_data, dtype=float)

    n_s = source_data.shape[0]
    n_t = target_data.shape[0]
    n = n_s + n_t

    # Domain labels
    domain_features = np.concatenate([np.zeros(n_s), np.ones(n_t)])
    data_combined = np.vstack([source_data, target_data])
    augmented_data = np.column_stack([data_combined, domain_features])

    # RBF kernel on augmented data
    nrm = np.sum(augmented_data ** 2, axis=1)
    K_x = np.exp(-0.5 / sigma ** 2 * (
        nrm[:, None] + nrm[None, :] - 2.0 * augmented_data @ augmented_data.T
    ))

    # Domain kernel (equality)
    K_d = (domain_features[:, None] == domain_features[None, :]).astype(float)

    # Centering matrix
    H = np.eye(n) - np.ones((n, n)) / n

    # Build M matrix
    KxH = K_x @ H
    if maximize:
        M = KxH @ K_d @ H @ K_x + mu * KxH @ K_x
    else:
        M = KxH @ K_d @ H @ K_x - mu * KxH @ K_x

    KH = KxH @ K_x

    # Symmetrize
    M = (M + M.T) / 2
    KH = (KH + KH.T) / 2

    # Generalized eigenvalue problem: M w = lambda KH w
    #
    # R uses geigen(M, KH, TRUE), falling back to geigen(M, KH, FALSE).
    # geigen(symmetric=TRUE) calls LAPACK dsygv (requires B positive definite).
    # geigen(symmetric=FALSE) calls LAPACK dggev (handles singular B).
    # R then takes vectors[, 1:k] — the first k columns in LAPACK return order.
    #
    # KH = K H K is typically rank (n-1) because H is rank (n-1),
    # so KH is NOT positive definite and eigh will fail.
    # We mirror R's fallback: eigh → eig, keeping LAPACK column order.
    eigenvalues = None
    eigenvectors = None
    try:
        eigenvalues, eigenvectors = eigh(M, KH)
    except Exception:
        pass

    if eigenvalues is None:
        # Non-symmetric generalized eigenproblem (matches geigen(M,KH,FALSE)).
        eigenvalues, eigenvectors = eig(M, KH)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
        # Filter out infinite eigenvalues (from null space of KH).
        # Keep columns in LAPACK-returned order (matches R's vectors[,1:k]).
        finite_mask = np.isfinite(eigenvalues)
        eigenvalues = eigenvalues[finite_mask]
        eigenvectors = eigenvectors[:, finite_mask]

    # L2-normalize eigenvector columns for consistent scale.
    # eigh normalizes w^T B w = 1, which blows up for near-null directions
    # of B.  LAPACK dggev (used by R's geigen) normalizes to ||w||=1.
    # We adopt unit-norm columns to match dggev and avoid scale explosion.
    col_norms = np.linalg.norm(eigenvectors, axis=0)
    col_norms[col_norms < 1e-12] = 1.0
    eigenvectors = eigenvectors / col_norms[np.newaxis, :]

    # Take first k columns (matching R's vectors[, 1:k])
    W = eigenvectors[:, :k]

    Z = K_x @ W
    Z = np.real(Z)

    return {
        "weighted_source_data": Z[:n_s],
        "target_data": Z[n_s:],
        "eigenvalue": W,
    }
