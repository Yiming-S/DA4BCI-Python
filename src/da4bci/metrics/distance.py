"""Distance metrics: MMD, Energy, Wasserstein, Mahalanobis, distance matrix."""

import numpy as np
from da4bci.metrics.kernels import rbf_kernel


def compute_distance_matrix(source, target, eps=1e-12):
    """Pairwise Euclidean distance matrix between source and target rows.

    Returns
    -------
    ndarray (n_s, n_t)
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)

    cross_term = source @ target.T
    source_norms = np.sum(source ** 2, axis=1)
    target_norms = np.sum(target ** 2, axis=1)

    d2 = source_norms[:, None] + target_norms[None, :] - 2.0 * cross_term
    # Clamp tiny negatives
    d2 = np.where((d2 > -eps) & (d2 < 0), 0.0, d2)
    d2 = np.where(d2 < -eps, np.nan, d2)

    return np.sqrt(d2)


def compute_mmd(source, target, sigma):
    """Squared MMD with RBF kernel.

    Returns
    -------
    float, MMD^2.
    """
    Kss = rbf_kernel(source, source, sigma)
    Ktt = rbf_kernel(target, target, sigma)
    Kst = rbf_kernel(source, target, sigma)

    m = source.shape[0]
    n = target.shape[0]

    return float(np.sum(Kss) / (m * m) + np.sum(Ktt) / (n * n) - 2.0 * np.sum(Kst) / (m * n))


def compute_energy(source, target):
    """Energy distance between empirical distributions.

    Returns
    -------
    float, non-negative.
    """
    ds = compute_distance_matrix(source, source)
    dt = compute_distance_matrix(target, target)
    d_st = compute_distance_matrix(source, target)

    ed2 = 2.0 * np.nanmean(d_st) - np.nanmean(ds) - np.nanmean(dt)
    return float(np.sqrt(max(ed2, 0.0)))


def _wasserstein_scipy(cost, p, q):
    """Solve 1-Wasserstein via scipy linear programming (no POT dependency).

    Minimises sum_{ij} C_{ij} T_{ij}  subject to
        T 1 = p,  T^T 1 = q,  T >= 0
    using scipy.optimize.linprog (revised simplex / HiGHS).
    """
    from scipy.optimize import linprog

    n_s, n_t = cost.shape
    c = cost.ravel()
    n_vars = n_s * n_t

    # Row-sum constraints:  sum_j T_{ij} = p_i
    A_row = np.zeros((n_s, n_vars))
    for i in range(n_s):
        A_row[i, i * n_t:(i + 1) * n_t] = 1.0

    # Col-sum constraints:  sum_i T_{ij} = q_j
    A_col = np.zeros((n_t, n_vars))
    for j in range(n_t):
        A_col[j, j::n_t] = 1.0

    A_eq = np.vstack([A_row, A_col])
    b_eq = np.concatenate([p, q])

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    T = res.x.reshape(n_s, n_t)
    return T


# Probe POT availability once at import time.  A broken POT install may
# segfault during import (C-level abort), which try/except cannot catch.
# To guard against this, the fallback (scipy linprog) is used when POT
# is not importable.  Set _POT_AVAILABLE = True / False accordingly.
try:
    import ot as _pot_mod
    _POT_AVAILABLE = True
except Exception:
    _POT_AVAILABLE = False


def compute_wasserstein(source, target):
    """1-Wasserstein distance (Earth Mover's Distance).

    Uses the POT library when available.  If POT cannot be imported
    (missing or broken install), falls back to ``scipy.optimize.linprog``
    which is slower but fully self-contained.

    Returns
    -------
    float, non-negative.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.shape[1] != target.shape[1]:
        raise ValueError("source and target must have the same number of columns")

    n_s = source.shape[0]
    n_t = target.shape[0]
    p = np.ones(n_s) / n_s
    q = np.ones(n_t) / n_t

    cost = compute_distance_matrix(source, target)
    cost = np.nan_to_num(cost, nan=1e30)

    if _POT_AVAILABLE:
        try:
            plan = _pot_mod.emd(p, q, cost)
            return float(np.sum(plan * cost))
        except Exception:
            pass

    # Fallback: exact LP via scipy
    plan = _wasserstein_scipy(cost, p, q)
    return float(np.sum(plan * cost))


def compute_mahalanobis(source, target, cov_choice="pooled",
                        shrinkage_alpha=None, ridge=1e-6, squared=False):
    """Mahalanobis distance between domain means.

    Parameters
    ----------
    source : ndarray (n_s, p)
    target : ndarray (n_t, p)
    cov_choice : str, one of "pooled", "source", "target"
    shrinkage_alpha : float or None, in [0, 1]
    ridge : float >= 0
    squared : bool

    Returns
    -------
    float, non-negative.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.shape[1] != target.shape[1]:
        raise ValueError("source and target must have the same number of columns")

    mu_x = source.mean(axis=0)
    mu_y = target.mean(axis=0)
    Sx = np.cov(source, rowvar=False, ddof=1)
    Sy = np.cov(target, rowvar=False, ddof=1)

    nx, p = source.shape
    ny = target.shape[0]

    if cov_choice == "pooled":
        if nx + ny - 2 <= 0:
            raise ValueError("Not enough samples for pooled covariance.")
        S = ((nx - 1) * Sx + (ny - 1) * Sy) / (nx + ny - 2)
    elif cov_choice == "source":
        S = Sx
    elif cov_choice == "target":
        S = Sy
    else:
        raise ValueError(f"cov_choice must be 'pooled', 'source', or 'target', got '{cov_choice}'")

    S = (S + S.T) / 2

    if shrinkage_alpha is not None:
        if not (0 <= shrinkage_alpha <= 1):
            raise ValueError("shrinkage_alpha must be in [0, 1]")
        trp = np.trace(S) / p
        S = (1 - shrinkage_alpha) * S + shrinkage_alpha * trp * np.eye(p)

    if ridge is not None and ridge > 0:
        trp = np.trace(S) / p
        S = S + ridge * trp * np.eye(p)

    # Stable inversion via eigen
    vals, vecs = np.linalg.eigh(S)
    eps_val = np.sqrt(np.finfo(float).eps)
    vals = np.maximum(vals, eps_val)
    S_inv = vecs @ np.diag(1.0 / vals) @ vecs.T

    d = mu_x - mu_y
    d2 = float(d @ S_inv @ d)
    return d2 if squared else np.sqrt(d2)
