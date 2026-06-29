"""RBF kernel and bandwidth heuristics."""

import numpy as np


def rbf_kernel(x, y, sigma, standard_scale=True):
    """Compute the RBF (Gaussian) kernel between x and y.

    Parameters
    ----------
    x : ndarray (n_x, p)
    y : ndarray (n_y, p)
    sigma : float > 0
    standard_scale : bool
        If True, gamma = 1/(2*sigma^2); else gamma = 1/sigma^2.

    Returns
    -------
    ndarray (n_x, n_y)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    assert x.shape[1] == y.shape[1], "x and y must have the same number of columns"
    assert sigma > 0, "sigma must be positive"

    xx = np.sum(x ** 2, axis=1)
    yy = np.sum(y ** 2, axis=1)
    D2 = xx[:, None] + yy[None, :] - 2.0 * (x @ y.T)
    D2 = np.maximum(D2, 0.0)

    gamma = 1.0 / (2.0 * sigma ** 2) if standard_scale else 1.0 / (sigma ** 2)
    return np.exp(-gamma * D2)


def sigma_med(X, Y, m=400, seed=0):
    """Median-distance heuristic for RBF bandwidth.

    Parameters
    ----------
    X : ndarray (n1, p)
    Y : ndarray (n2, p)
    m : int
        Max rows to use for pairwise distances.
    seed : int or None
        Subsampling seed. Defaults to 0 (not None) so the bandwidth — and thus
        every RBF metric / KMM weight built on it — is reproducible run-to-run
        and across machines when n1 + n2 > m.

    Returns
    -------
    float, positive median Euclidean distance.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    assert X.shape[1] == Y.shape[1]

    rng = np.random.RandomState(seed)
    XY = np.vstack([X, Y])
    N = XY.shape[0]

    if N <= 2:
        import warnings
        warnings.warn("Not enough samples to compute pairwise distances; returning NaN.")
        return float("nan")

    if N > m:
        idx = rng.choice(N, size=m, replace=False)
        XY = XY[idx]

    from scipy.spatial.distance import pdist
    d_med = float(np.median(pdist(XY, metric="euclidean")))

    if d_med == 0:
        d_med = np.finfo(float).eps

    return d_med
