"""SPD geometry: matrix_power, riemannian_mean, log/exp map, LW covariance, etc."""

import warnings

import numpy as np
from scipy.linalg import eigh, qr, svd


def orthonormal_complement(U, tol=1e-8):
    """Return an orthonormal basis of the subspace orthogonal to columns of U.

    Parameters
    ----------
    U : ndarray (d, k)
        Matrix with (approximately) orthonormal columns.
    tol : float
        Tolerance for rank-revealing QR.

    Returns
    -------
    ndarray (d, d-r) with orthonormal columns, where r = rank(U).
    """
    U = np.asarray(U, dtype=float)
    d = U.shape[0]
    k = U.shape[1] if U.ndim == 2 else 0

    if d == 0:
        raise ValueError("U must have at least one row.")
    if k == 0:
        return np.eye(d)
    if d < k:
        raise ValueError("U must have at least as many rows as columns (d >= k).")

    Q, R, P = qr(U, pivoting=True)
    r = np.sum(np.abs(np.diag(R)) > tol)

    if r >= d:
        return np.zeros((d, 0))

    Q_perp = Q[:, r:]
    # Re-orthonormalize
    Q_perp, _ = np.linalg.qr(Q_perp)
    return Q_perp


def LW_covariance(x):
    """Ledoit-Wolf shrinkage covariance estimator (toward identity).

    Parameters
    ----------
    x : ndarray (n, p)

    Returns
    -------
    ndarray (p, p) shrunk covariance matrix.
    """
    x = np.asarray(x, dtype=float)
    n, p = x.shape

    # Center
    x = x - x.mean(axis=0)

    # Sample covariance (1/n normalization, matching R)
    S = (x.T @ x) / n

    # Shrinkage target: m * I
    m = np.trace(S) / p

    # d2
    d2 = np.sum((S - m * np.eye(p)) ** 2) / p

    # bbar2
    bbar2 = (np.sum(np.sum(x ** 2, axis=1) ** 2) - n * np.sum(S ** 2)) / (p * n ** 2)
    b2 = min(bbar2, d2)

    rho = b2 / d2 if d2 > 0 else 0.0

    S_shrunk = (1 - rho) * S
    np.fill_diagonal(S_shrunk, np.diag(S_shrunk) + rho * m)
    return S_shrunk


def matrix_power(A, power, eig_eps=1e-6, ridge_eps=1e-6, max_retry=3):
    """Compute A^power for SPD matrix A via eigendecomposition.

    Parameters
    ----------
    A : ndarray (n, n)
        Symmetric positive definite matrix.
    power : float
        Exponent.

    Returns
    -------
    ndarray (n, n)
    """
    A = np.asarray(A, dtype=float)
    A = (A + A.T) / 2
    n = A.shape[0]

    if power == 1:
        return A.copy()
    if power == 0:
        return np.eye(n)

    if not np.all(np.isfinite(A)):
        warnings.warn(
            "matrix_power: non-finite input matrix; returning identity "
            "(the alignment is a silent no-op).", RuntimeWarning)
        return np.eye(n)

    # Fast path for inverse via Cholesky
    if power == -1:
        try:
            L = np.linalg.cholesky(A)
            Linv = np.linalg.solve(L, np.eye(n))
            return Linv.T @ Linv
        except np.linalg.LinAlgError:
            pass

    # Spectral decomposition with iterative ridge
    ridge = ridge_eps
    vals = None
    vecs = None
    for _ in range(max_retry):
        M = A + ridge * np.eye(n)
        try:
            vals, vecs = eigh(M)
            break
        except np.linalg.LinAlgError:
            ridge *= 10

    if vals is None:
        warnings.warn(
            "matrix_power: eigendecomposition failed after retries; returning "
            "identity (the alignment is a silent no-op).", RuntimeWarning)
        return np.eye(n)

    vals = np.real(vals)
    vals[~np.isfinite(vals)] = eig_eps
    vals = np.maximum(vals, eig_eps)

    if power == -1:
        vals_pow = 1.0 / vals
    else:
        vals_pow = vals ** power

    # V * diag(sqrt(vals_pow)) then tcrossprod
    V_scaled = vecs * np.sqrt(vals_pow)[np.newaxis, :]
    return V_scaled @ V_scaled.T


def riemannian_mean(cov_matrices, max_iterations=500, epsilon=1e-5):
    """Affine-invariant Riemannian mean of SPD matrices.

    Parameters
    ----------
    cov_matrices : ndarray (p, p, m) or list of (p, p) arrays.
    max_iterations : int
    epsilon : float
        Convergence tolerance on Frobenius norm.

    Returns
    -------
    ndarray (p, p) SPD matrix.
    """
    if isinstance(cov_matrices, list):
        cov_matrices = np.stack(cov_matrices, axis=-1)

    m = cov_matrices.shape[2]
    n = cov_matrices.shape[0]

    # Initialize with element-wise mean
    P_omega = cov_matrices.mean(axis=2)

    for _ in range(max_iterations):
        vals, vecs = eigh(P_omega)
        vals = np.maximum(vals, 1e-12)
        sqrt_P = vecs * np.sqrt(vals)[np.newaxis, :]
        sqrt_P = sqrt_P @ vecs.T
        inv_sqrt_P = vecs * (1.0 / np.sqrt(vals))[np.newaxis, :]
        inv_sqrt_P = inv_sqrt_P @ vecs.T

        S = np.zeros((n, n))
        for i in range(m):
            W = inv_sqrt_P @ cov_matrices[:, :, i] @ inv_sqrt_P
            ew, ev = eigh(W)
            ew = np.maximum(ew, 1e-15)
            S += ev @ np.diag(np.log(ew)) @ ev.T
        S /= m

        PS = P_omega @ S
        norm_val = np.sqrt(np.sum(PS * PS.T))

        # Exponential map
        ew_s, ev_s = eigh(S)
        P_omega = sqrt_P @ ev_s @ np.diag(np.exp(ew_s)) @ ev_s.T @ sqrt_P

        if norm_val < epsilon:
            break

    return P_omega


def log_map(P_omega, P_i):
    """Riemannian logarithmic map at P_omega applied to P_i.

    Parameters
    ----------
    P_omega : ndarray (p, p) or dict with 'sqrt' and 'inv_sqrt'.
    P_i : ndarray (p, p) SPD matrix.

    Returns
    -------
    ndarray (p, p) tangent vector.
    """
    if isinstance(P_omega, dict):
        sqrt_P = P_omega["sqrt"]
        inv_sqrt_P = P_omega["inv_sqrt"]
    else:
        P_omega = np.asarray(P_omega, dtype=float)
        vals, vecs = eigh(P_omega)
        vals = np.maximum(vals, 1e-15)
        sqrt_P = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
        inv_sqrt_P = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T

    W = inv_sqrt_P @ P_i @ inv_sqrt_P
    ew, ev = eigh(W)
    ew = np.maximum(ew, 1e-15)
    log_W = ev @ np.diag(np.log(ew)) @ ev.T

    return sqrt_P @ log_W @ sqrt_P


def exp_map(P_omega, S_i):
    """Riemannian exponential map at P_omega applied to tangent vector S_i.

    Parameters
    ----------
    P_omega : ndarray (p, p) or dict with 'sqrt' and 'inv_sqrt'.
    S_i : ndarray (p, p) symmetric tangent matrix.

    Returns
    -------
    ndarray (p, p) SPD matrix.
    """
    if isinstance(P_omega, dict):
        sqrt_P = P_omega["sqrt"]
        inv_sqrt_P = P_omega["inv_sqrt"]
    else:
        P_omega = np.asarray(P_omega, dtype=float)
        vals, vecs = eigh(P_omega)
        vals = np.maximum(vals, 1e-15)
        sqrt_P = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
        inv_sqrt_P = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T

    W = inv_sqrt_P @ S_i @ inv_sqrt_P
    ew, ev = eigh(W)
    exp_W = ev @ np.diag(np.exp(ew)) @ ev.T

    return sqrt_P @ exp_W @ sqrt_P


def align_riemannian_transport(cov_S, cov_T):
    """Align source SPD matrices to target geometry via log-exp transport.

    Parameters
    ----------
    cov_S : list of (p, p) SPD matrices (source).
    cov_T : list of (p, p) SPD matrices (target).

    Returns
    -------
    list of aligned source SPD matrices.
    """
    if len(cov_S) == 1:
        cov_S = cov_S + cov_S
    if len(cov_T) == 1:
        cov_T = cov_T + cov_T

    mean_S = riemannian_mean(cov_S)
    mean_T = riemannian_mean(cov_T)

    return [exp_map(mean_T, log_map(mean_S, C)) for C in cov_S]


def compute_geodesic(source, target, d=None):
    """Grassmann geodesic distance between column subspaces.

    Parameters
    ----------
    source : ndarray (n_s, p)
    target : ndarray (n_t, p)
    d : int or None
        Subspace dimension; defaults to minimum rank.

    Returns
    -------
    float, non-negative geodesic distance.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    assert source.shape[1] == target.shape[1]

    p = source.shape[1]

    if d is None:
        d_source = np.linalg.matrix_rank(source)
        d_target = np.linalg.matrix_rank(target)
        d = min(d_source, d_target)
    else:
        d = int(d)
        if d < 1 or d > p:
            raise ValueError("d must be between 1 and number of columns")

    def orthonorm_basis(X, k):
        # Feature-space principal subspace: the top-k right singular vectors of
        # the centered data, shape (p, k). This is independent of the number of
        # samples, so U.T @ V is well-defined even when n_s != n_t (the normal
        # domain-adaptation case). The previous QR-in-sample-space basis was
        # (n, k) and crashed whenever n_s != n_t.
        Xc = X - X.mean(axis=0)
        _, _, Vt = svd(Xc, full_matrices=False)
        return Vt[:min(k, Vt.shape[0])].T

    U = orthonorm_basis(source, d)
    V = orthonorm_basis(target, d)

    s = svd(U.T @ V, compute_uv=False)
    cos_t = np.clip(s, -1, 1)
    theta = np.arccos(cos_t)

    return np.sqrt(np.sum(theta ** 2))
