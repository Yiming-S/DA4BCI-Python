"""Parallel Transport (PT) on the SPD cone manifold."""

import numpy as np
from da4bci.geometry.spd import LW_covariance, matrix_power


def domain_adaptation_pt(source_data, target_data):
    """PT: E = (C_T C_S^{-1})^{1/2}, map source covariance to target.

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

    C_S = LW_covariance(Xs_c)
    C_T = LW_covariance(target_data - mu_T)

    # E = (C_T C_S^{-1})^{1/2}
    E = matrix_power(C_T @ np.linalg.solve(C_S, np.eye(C_S.shape[0])), 0.5)
    M = E.T

    Xs_aligned = Xs_c @ M + mu_T

    return {
        "weighted_source_data": Xs_aligned,
        "target_data": target_data,
        "transformation_matrix": M,
    }
