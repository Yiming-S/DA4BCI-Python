"""Riemannian-Distance-Based Alignment (RD)."""

import numpy as np
from scipy.linalg import svd


def domain_adaptation_riemannian(source_data, target_data, ridge=1e-6, pinv_tol=1e-10):
    """Procrustes-like alignment using Riemannian distance.

    Parameters
    ----------
    source_data : ndarray (n_s, p)
    target_data : ndarray (n_t, p)
    ridge : float
    pinv_tol : float

    Returns
    -------
    dict with 'weighted_source_data', 'target_data', 'rotation_matrix',
         'cov_source_aligned', 'riemannian_distance'.
    """
    source_data = np.asarray(source_data, dtype=float)
    target_data = np.asarray(target_data, dtype=float)
    p = source_data.shape[1]

    # Covariance with ridge
    C_source = np.cov(source_data, rowvar=False, ddof=1) + ridge * np.eye(p)
    C_target = np.cov(target_data, rowvar=False, ddof=1) + ridge * np.eye(p)

    # Invert C_source
    try:
        inv_Cs = np.linalg.inv(C_source)
    except np.linalg.LinAlgError:
        U, s, Vt = svd(C_source)
        d_inv = np.where(s > pinv_tol, 1.0 / s, 0.0)
        inv_Cs = Vt.T @ np.diag(d_inv) @ U.T

    # Riemannian distance
    eig_vals = np.linalg.eigvals(inv_Cs @ C_target)
    riem_dist = float(np.sqrt(np.sum(np.log(np.real(eig_vals)) ** 2)))

    # Procrustes alignment
    U_s = svd(C_source)[0]
    U_t = svd(C_target)[0]
    R = U_s @ U_t.T

    C_source_aligned = R @ C_source @ R.T

    return {
        "weighted_source_data": source_data @ R,
        "target_data": target_data,
        "rotation_matrix": R,
        "cov_source_aligned": C_source_aligned,
        "riemannian_distance": riem_dist,
    }
