"""Transfer Component Analysis (TCA)."""

import numpy as np
from scipy.sparse.linalg import eigsh


def domain_adaptation_tca(source_data, target_data, k=10, sigma=1, mu=1):
    """TCA: project source/target onto shared latent space to reduce MMD.

    Parameters
    ----------
    source_data : ndarray (n_s, p)
    target_data : ndarray (n_t, p)
    k : int, number of latent components.
    sigma : float, RBF bandwidth.
    mu : float, regularization.

    Returns
    -------
    dict with 'weighted_source_data', 'target_data', 'eigenvalue'.
    """
    source_data = np.asarray(source_data, dtype=float)
    target_data = np.asarray(target_data, dtype=float)

    X = np.vstack([source_data, target_data])
    n_s = source_data.shape[0]

    # RBF kernel
    nrmX = np.sum(X ** 2, axis=1)
    K = np.exp(-0.5 / sigma ** 2 * (nrmX[:, None] + nrmX[None, :] - 2.0 * X @ X.T))

    # KHK where H = I - 1/n * 11^T
    K_centered = K - K.mean(axis=1, keepdims=True)
    KHK = K_centered @ K_centered.T

    # Domain difference vector
    a = K[:, :n_s].mean(axis=1) - K[:, n_s:].mean(axis=1)
    cst = mu * (mu + np.sum(a ** 2))

    # B = KHK
    B = KHK

    # inv(A) * B where A = mu * I + a a^T
    # Sherman-Morrison: inv(A)*B = B/mu - (a / cst) * (a^T B)
    invAB = B / mu - np.outer(a / cst, a @ B)

    # Top-k eigenvectors of invAB
    # Make symmetric for eigsh
    invAB_sym = (invAB + invAB.T) / 2.0
    try:
        eigenvalues, W = eigsh(invAB_sym, k=k, which="LM")
    except Exception:
        eigenvalues, W_full = np.linalg.eigh(invAB_sym)
        idx = np.argsort(np.abs(eigenvalues))[::-1][:k]
        W = W_full[:, idx]

    W = np.real(W)

    # Rescale to have norm 1 in metric B
    nrm = np.sum(W * (B @ W), axis=0)
    valid = nrm > 0
    if not np.all(valid):
        W = W[:, valid]
        nrm = nrm[valid]
    W = W / np.sqrt(nrm)[np.newaxis, :]

    Z = K @ W

    return {
        "weighted_source_data": Z[:n_s],
        "target_data": Z[n_s:],
        "eigenvalue": W,
    }
