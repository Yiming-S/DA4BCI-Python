"""Aligned Riemannian Transport (ART)."""

import numpy as np
from da4bci.geometry.spd import LW_covariance, matrix_power, align_riemannian_transport


def domain_adaptation_art(source_data, target_data):
    """ART: align source covariance to target via Riemannian transport.

    Parameters
    ----------
    source_data : ndarray (n_s, p)
    target_data : ndarray (n_t, p)

    Returns
    -------
    dict with 'weighted_source_data', 'target_data', 'transformation_matrix'.
    """
    source_data = np.asarray(source_data, dtype=float)
    target_data = np.asarray(target_data, dtype=float)

    mu_S = source_data.mean(axis=0)
    mu_T = target_data.mean(axis=0)
    Xs_c = source_data - mu_S
    Xt_c = target_data - mu_T

    C_S = LW_covariance(Xs_c)
    C_T = LW_covariance(Xt_c)

    try:
        C_S_aligned = align_riemannian_transport([C_S], [C_T])[0]
    except Exception:
        C_S_aligned = C_T

    C_S_inv_sqrt = matrix_power(C_S, -0.5)
    C_S_aligned_sqrt = matrix_power(C_S_aligned, 0.5)
    M = C_S_inv_sqrt @ C_S_aligned_sqrt

    Xs_aligned = (source_data - mu_S) @ M + mu_T

    return {
        "weighted_source_data": Xs_aligned,
        "target_data": target_data,
        "transformation_matrix": M,
    }
