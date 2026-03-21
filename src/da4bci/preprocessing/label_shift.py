"""Prior adjustment under label shift via EM."""

import numpy as np


def label_shift_em(Ps_yx, pi_s, tol=1e-7, maxit=200):
    """Estimate target class priors and adjust posteriors.

    Parameters
    ----------
    Ps_yx : ndarray (n_t, K), source-model posteriors on target (rows sum to 1).
    pi_s : array-like (K,), source priors.
    tol : float
    maxit : int

    Returns
    -------
    dict with 'pi_t', 'P_adj', 'iter'.
    """
    Ps_yx = np.asarray(Ps_yx, dtype=float)
    pi_s = np.asarray(pi_s, dtype=float)
    pi_s = pi_s / pi_s.sum()
    pi_t = pi_s.copy()

    it = 0
    for it in range(1, maxit + 1):
        # E-step
        R = Ps_yx * (pi_t / (pi_s + 1e-15))
        R = R / (R.sum(axis=1, keepdims=True) + 1e-15)
        # M-step
        pi_new = R.mean(axis=0)
        if np.max(np.abs(pi_new - pi_t)) < tol:
            pi_t = pi_new
            break
        pi_t = pi_new

    # Adjusted posteriors
    P_adj = Ps_yx * (pi_t / (pi_s + 1e-15))
    P_adj = P_adj / (P_adj.sum(axis=1, keepdims=True) + 1e-15)

    return {"pi_t": pi_t, "P_adj": P_adj, "iter": it}
