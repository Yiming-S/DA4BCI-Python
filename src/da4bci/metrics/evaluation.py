"""Evaluation utilities: evaluate_shift, proxy_a_distance, distance_summary."""

import numpy as np
from da4bci.metrics.distance import compute_mmd, compute_wasserstein


def evaluate_shift(source, target, adapted_source, adapted_target):
    """Report MMD and Wasserstein distances before/after adaptation.

    Returns a dict-of-lists matching R's data.frame structure::

        {"Metric": [...], "Before": [...], "After": [...]}

    This matches R's ``evaluate_shift`` return value.
    """
    mmd_before = compute_mmd(source, target, sigma=1)
    mmd_after = compute_mmd(adapted_source, adapted_target, sigma=1)
    wass_before = compute_wasserstein(source, target)
    wass_after = compute_wasserstein(adapted_source, adapted_target)

    return {
        "Metric": ["MMD", "Wasserstein"],
        "Before": [mmd_before, wass_before],
        "After": [mmd_after, wass_after],
    }


def proxy_a_distance(Xs, Xt, folds=5, ridge=1e-3, seed=None):
    """Proxy A-Distance via ridge LDA with K-fold CV.

    Returns
    -------
    dict with 'pad' and 'err'.
    """
    Xs = np.asarray(Xs, dtype=float)
    Xt = np.asarray(Xt, dtype=float)
    if Xs.shape[1] != Xt.shape[1]:
        raise ValueError("Xs and Xt must have the same number of columns.")

    X = np.vstack([Xs, Xt])
    y = np.array([0] * Xs.shape[0] + [1] * Xt.shape[0])
    n = X.shape[0]
    p = X.shape[1]

    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    X = X[idx]
    y = y[idx]

    # Stratified folds
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    K = max(2, min(folds, len(idx0), len(idx1)))

    rng.shuffle(idx0)
    rng.shuffle(idx1)
    folds0 = np.array_split(idx0, K)
    folds1 = np.array_split(idx1, K)
    folds_idx = [np.sort(np.concatenate([folds0[k], folds1[k]])) for k in range(K)]

    pred = np.full(n, -1, dtype=int)
    for k in range(K):
        te = folds_idx[k]
        tr = np.setdiff1d(np.arange(n), te)
        if len(np.unique(y[tr])) < 2:
            continue
        # Standardize
        mu = X[tr].mean(axis=0)
        sd = np.maximum(X[tr].std(axis=0, ddof=1), 1e-8)
        Xtr = (X[tr] - mu) / sd
        Xte = (X[te] - mu) / sd

        # Ridge LDA
        A0 = Xtr[y[tr] == 0]
        A1 = Xtr[y[tr] == 1]
        S0 = np.cov(A0, rowvar=False, ddof=1) if len(A0) > 1 else np.zeros((p, p))
        S1 = np.cov(A1, rowvar=False, ddof=1) if len(A1) > 1 else np.zeros((p, p))
        Sp = ((max(len(A0) - 1, 0) * S0 + max(len(A1) - 1, 0) * S1) /
              max(len(Xtr) - 2, 1)) + ridge * np.eye(p)
        iSp = np.linalg.solve(Sp, np.eye(p))
        mu0 = A0.mean(axis=0)
        mu1 = A1.mean(axis=0)
        w = iSp @ (mu1 - mu0)
        b = -0.5 * (mu1 @ iSp @ mu1 - mu0 @ iSp @ mu0)

        pred[te] = (Xte @ w + b >= 0).astype(int)

    ok = pred >= 0
    err = float(np.mean(pred[ok] != y[ok]))
    err = min(err, 1 - err)
    pad = 2 * (1 - 2 * err)
    return {"pad": pad, "err": err}


def distance_summary(source, target, sigma=None,
                     include=None, format="list",
                     pad_folds=5, pad_ridge=1e-3, pad_seed=None):
    """Compute multiple distribution distance metrics.

    Parameters
    ----------
    source, target : ndarray
    sigma : float or None
    include : list of str or None
        Metrics to include. Default: all.
    format : str, 'list' or 'table'.
        'list' returns a dict; 'table' returns a dict-of-lists with
        columns 'Metric' and 'Value', plus 'sigma_used' attribute.

    Returns
    -------
    dict (format='list') or dict with Metric/Value lists (format='table').
    """
    from da4bci.metrics.kernels import sigma_med
    from da4bci.metrics.distance import (
        compute_energy, compute_mahalanobis,
    )
    from da4bci.geometry.spd import compute_geodesic

    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)

    if sigma is None:
        sigma = sigma_med(source, target)

    all_metrics = [
        "PAD", "MMD2", "Energy", "MMD", "Wasserstein", "Geodesic", "Mahalanobis"
    ]
    if include is None:
        include = all_metrics

    results = {}
    if "PAD" in include:
        results["PAD"] = proxy_a_distance(source, target,
                                          folds=pad_folds, ridge=pad_ridge,
                                          seed=pad_seed)["pad"]
    if "MMD2" in include or "MMD" in include:
        mmd2 = compute_mmd(source, target, sigma)
        if "MMD2" in include:
            results["MMD2"] = mmd2
        if "MMD" in include:
            results["MMD"] = np.sqrt(max(mmd2, 0))
    if "Energy" in include:
        results["Energy"] = compute_energy(source, target)
    if "Wasserstein" in include:
        results["Wasserstein"] = compute_wasserstein(source, target)
    if "Geodesic" in include:
        results["Geodesic"] = compute_geodesic(source, target)
    if "Mahalanobis" in include:
        results["Mahalanobis"] = compute_mahalanobis(source, target)

    if format == "table":
        keep = [m for m in include if m in results]
        out = {
            "Metric": keep,
            "Value": [results[m] for m in keep],
        }
        # Attach sigma_used as attribute (matches R's attr(out, "sigma_used"))
        out["sigma_used"] = sigma
        return out

    return results


# R-compatible alias
distanceSummary = distance_summary
