"""Correlation Alignment (CORAL)."""

import numpy as np


def domain_adaptation_coral(source_data, target_data, lam=1e-5):
    """CORAL: whiten source, recolor with target covariance.

    Parameters
    ----------
    source_data : ndarray (n_s, p)
    target_data : ndarray (n_t, p)
    lam : float, regularization for covariance.

    Returns
    -------
    dict with 'weighted_source_data', 'target_data'.
    """
    source_data = np.asarray(source_data, dtype=float)
    target_data = np.asarray(target_data, dtype=float)
    p = source_data.shape[1]

    def regularize_cov(C, eps=1e-6):
        return C + np.diag(eps * np.diag(C))

    cov_source = np.cov(source_data, rowvar=False, ddof=1) + lam * np.eye(p)
    cov_target = np.cov(target_data, rowvar=False, ddof=1) + lam * np.eye(p)
    cov_source = regularize_cov(cov_source, lam)
    cov_target = regularize_cov(cov_target, lam)

    # Whiten: chol(inv(cov_source))
    try:
        L_s = np.linalg.cholesky(np.linalg.inv(cov_source))
    except np.linalg.LinAlgError:
        L_s = np.linalg.cholesky(np.linalg.pinv(cov_source))
    # Right-multiply whitening: cov(Xs @ M) = M.T cov_s M = I requires M M.T =
    # inv(cov_s) = L_s L_s.T, i.e. M = L_s (the lower factor) — NOT L_s.T.
    chol_s = L_s

    # Recolor: chol(cov_target)
    try:
        L_t = np.linalg.cholesky(cov_target)
    except np.linalg.LinAlgError:
        L_t = np.linalg.cholesky(np.linalg.pinv(cov_target))
    chol_t = L_t.T  # upper triangular

    source_aligned = source_data @ chol_s @ chol_t

    return {
        "weighted_source_data": source_aligned,
        "target_data": target_data,
    }
