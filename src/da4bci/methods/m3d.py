"""Manifold-based Multi-step Domain Adaptation (M3D)."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def domain_adaptation_m3d(source_data, source_labels, target_data,
                          stage1=None, stage2=None,
                          l_iter=10, lambda_ridge=1e-2, eta_kernel=0.1,
                          label_offset=0, expl_var=0.90, max_dim=30):
    """M3D: two-stage alignment + iterative kernel refinement.

    Parameters
    ----------
    source_data : ndarray (n_s, p)
    source_labels : array-like (n_s,)
    target_data : ndarray (n_t, p)
    stage1 : dict with 'method' and 'control', or None.
    stage2 : dict with 'method' and 'control', or None.
    l_iter : int
    lambda_ridge : float
    eta_kernel : float
    label_offset : float
    expl_var : float
    max_dim : int

    Returns
    -------
    dict with 'weighted_source_data', 'target_data'.
    """
    from da4bci.methods import domain_adaptation

    source_data = np.asarray(source_data, dtype=float)
    target_data = np.asarray(target_data, dtype=float)
    source_labels = np.asarray(source_labels)

    if stage1 is None:
        stage1 = {"method": "tca", "control": {"k": None, "sigma": 1}}
    if stage2 is None:
        stage2 = {"method": "sa", "control": {"k": 10}}

    def auto_dim(X, keep=expl_var, cap=max_dim):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA()
        pca.fit(X_scaled)
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        idx = np.where(cum_var >= keep)[0]
        if len(idx) > 0:
            return min(int(idx[0]) + 1, cap)
        return cap

    def center_kernel(K):
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    # Auto-dimensioning — matches R: if is.null(stage1$control$k)
    if stage1["control"].get("k") is None:
        stage1["control"]["k"] = auto_dim(np.vstack([source_data, target_data]))

    # R: if (is.null(stage2$control$dim_subspace))
    #   stage2$control$dim_subspace <- min(stage1$control$k, max_dim)
    # In R, accessing a NULL key in a list simply returns NULL, so this
    # always fires when dim_subspace was never set.  Replicate that logic:
    stage2_ctrl = stage2.get("control", {})
    if stage2_ctrl.get("dim_subspace") is None:
        stage2_ctrl["dim_subspace"] = min(stage1["control"]["k"], max_dim)
    if "control" not in stage2:
        stage2["control"] = stage2_ctrl

    # Stage 1
    da1 = domain_adaptation(source_data, target_data,
                            method=stage1["method"], control=stage1["control"])
    Zs1 = da1["weighted_source_data"]
    Zt1 = da1["target_data"]

    # Stage 2
    if stage2 is not None:
        da2 = domain_adaptation(Zs1, Zt1,
                                method=stage2["method"], control=stage2["control"])
        Zs = da2["weighted_source_data"]
        Zt = da2["target_data"]
        K_all = da2.get("K", np.vstack([Zs, Zt]) @ np.vstack([Zs, Zt]).T)
    else:
        Zs = Zs1
        Zt = Zt1
        K_all = np.vstack([Zs, Zt]) @ np.vstack([Zs, Zt]).T

    K_all = center_kernel(K_all)

    n_s = Zs.shape[0]
    n_t = Zt.shape[0]
    n_all = n_s + n_t
    idx_s = np.arange(n_s)
    idx_t = np.arange(n_s, n_all)

    # One-hot encode labels
    unique_labels = np.unique(source_labels)
    K_classes = len(unique_labels)
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    Ysrc = np.zeros((n_s, K_classes))
    for i, l in enumerate(source_labels):
        Ysrc[i, label_to_idx[l]] = 1.0

    Kss = K_all[np.ix_(idx_s, idx_s)]
    Kts = K_all[np.ix_(idx_t, idx_s)]

    # Initial pseudo-labels
    W = np.linalg.solve(Kss + lambda_ridge * np.eye(n_s), Ysrc)
    Y_t = np.argmax(Kts @ W, axis=1)

    # Iterative alignment
    Z_all = np.vstack([Zs, Zt])
    M = np.zeros((n_all, n_all))

    for it in range(l_iter):
        M0 = np.full((n_all, n_all), -1.0 / (n_s * n_t))
        M0[np.ix_(idx_s, idx_s)] = 1.0 / n_s ** 2
        M0[np.ix_(idx_t, idx_t)] = 1.0 / n_t ** 2

        Mc = np.zeros((n_all, n_all))
        for c in range(K_classes):
            S = idx_s[Ysrc[:, c] == 1]
            T = idx_t[Y_t == c]
            if len(S) == 0 or len(T) == 0:
                continue
            ns_c = len(S)
            nt_c = len(T)
            Mc[np.ix_(S, S)] += 1.0 / ns_c ** 2
            Mc[np.ix_(T, T)] += 1.0 / nt_c ** 2
            Mc[np.ix_(S, T)] -= 1.0 / (ns_c * nt_c)
            Mc[np.ix_(T, S)] -= 1.0 / (ns_c * nt_c)

        tr_Mc = np.sum(Z_all * (Mc @ Z_all))
        tr_tot = tr_Mc + np.sum(Z_all * (M0 @ Z_all))
        mu = 0.0 if tr_tot < 1e-12 else tr_Mc / tr_tot
        M = (1 - mu) * M0 + mu * Mc

        Hs = Kss + eta_kernel * M[np.ix_(idx_s, idx_s)]
        W = np.linalg.solve(Hs + lambda_ridge * np.eye(n_s), Ysrc - label_offset)
        Y_t = np.argmax(Kts @ W, axis=1)

    # Final kernel eigendecomposition
    K_aligned = K_all + eta_kernel * M
    K_aligned = (K_aligned + K_aligned.T) / 2

    eigenvalues, eigenvectors = np.linalg.eigh(K_aligned)
    # Sort descending (R's eigen() returns descending)
    idx_sort = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx_sort]
    eigenvectors = eigenvectors[:, idx_sort]

    # R: k <- stage2$control$dim_subspace
    # Use dim_subspace (set above), matching R exactly.
    k_dim = stage2["control"]["dim_subspace"]
    k_dim = min(k_dim, len(eigenvalues))
    eigenvalues = eigenvalues[:k_dim]
    eigenvectors = eigenvectors[:, :k_dim]

    eigenvalues = np.maximum(eigenvalues, 0)
    Z_all_transformed = eigenvectors * np.sqrt(eigenvalues)[np.newaxis, :]

    return {
        "weighted_source_data": Z_all_transformed[:n_s],
        "target_data": Z_all_transformed[n_s:],
    }
