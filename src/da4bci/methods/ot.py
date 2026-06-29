"""Entropy-Regularized Optimal Transport (Sinkhorn-Knopp) with Barycentric Mapping."""

import numpy as np


def domain_adaptation_ot(source_data, target_data, eps=0.05, maxit=500,
                         tol=1e-7, cost="sqeuclidean"):
    """Sinkhorn OT with barycentric mapping.

    Parameters
    ----------
    source_data : ndarray (n, d)
    target_data : ndarray (m, d)
    eps : float > 0, entropy regularization.
    maxit : int
    tol : float
    cost : str, 'sqeuclidean' or 'euclidean'.

    Returns
    -------
    dict with 'weighted_source_data', 'target_data', 'ot_plan', 'cost',
         'epsilon', 'iterations', 'converged', 'residual'.
    """
    Xs = np.asarray(source_data, dtype=float)
    Xt = np.asarray(target_data, dtype=float)
    assert Xs.shape[1] == Xt.shape[1]
    assert eps > 0

    n = Xs.shape[0]
    m = Xt.shape[0]

    # Cost matrix
    XXs = np.sum(Xs ** 2, axis=1)
    XXt = np.sum(Xt ** 2, axis=1)
    Csq = XXs[:, None] + XXt[None, :] - 2.0 * Xs @ Xt.T
    Csq = np.maximum(Csq, 0.0)
    C = np.sqrt(Csq) if cost == "euclidean" else Csq

    # Sinkhorn
    r = np.ones(n) / n
    c = np.ones(m) / m

    K = np.exp(-C / eps)
    tiny = np.finfo(float).tiny
    K = np.maximum(K, tiny)

    u = np.ones(n)
    v = np.ones(m)

    residual = np.inf
    iters = 0
    for it in range(1, maxit + 1):
        iters = it
        Kv = K @ v
        Kv = np.maximum(Kv, tiny)
        u = r / Kv

        Ktu = K.T @ u
        Ktu = np.maximum(Ktu, tiny)
        v = c / Ktu

        # Marginal residual without materializing the full n*m plan:
        # the row marginal of P is u * (K @ v), the column marginal is v * (K.T @ u).
        residual = max(np.max(np.abs(u * (K @ v) - r)),
                       np.max(np.abs(v * Ktu - c)))
        if not np.isfinite(residual) or residual <= tol:
            break

    converged = np.isfinite(residual) and residual <= tol

    # Transport plan (built once) + barycentric mapping
    P = (u[:, None] * v[None, :]) * K
    rs = P.sum(axis=1)
    rs = np.maximum(rs, tiny)
    Xs_map = (P @ Xt) / rs[:, None]

    return {
        "weighted_source_data": Xs_map,
        "target_data": Xt,
        "ot_plan": P,
        "cost": C,
        "epsilon": eps,
        "iterations": iters,
        "converged": converged,
        "residual": float(residual),
    }
