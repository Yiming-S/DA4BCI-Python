"""Kernel Mean Matching (KMM) weights."""

import numpy as np
from da4bci.metrics.kernels import rbf_kernel, sigma_med


def kmm_weights(Xs, Xt, sigma=None, B=100, eps=None):
    """Compute KMM reweighting coefficients via quadratic programming.

    Parameters
    ----------
    Xs : ndarray (n, d) source samples.
    Xt : ndarray (m, d) target samples.
    sigma : float or None, bandwidth; defaults to sigma_med.
    B : float, upper bound for each weight.
    eps : float or None, tolerance on total weight.

    Returns
    -------
    ndarray (n,) weights in [0, B].
    """
    from scipy.optimize import minimize

    Xs = np.asarray(Xs, dtype=float)
    Xt = np.asarray(Xt, dtype=float)
    n = Xs.shape[0]
    m = Xt.shape[0]

    if sigma is None:
        sigma = sigma_med(Xs, Xt)
    if eps is None:
        eps = B / np.sqrt(max(1, n))

    Kss = rbf_kernel(Xs, Xs, sigma)
    Kst = rbf_kernel(Xs, Xt, sigma)
    Kss = (Kss + Kss.T) / 2.0

    # Linear term
    kappa = (n / m) * Kst.sum(axis=1)

    # Add ridge for numerical stability
    Dmat = Kss + 1e-6 * np.eye(n)

    # Objective: 0.5 * w^T Kss w - kappa^T w
    def objective(w):
        return 0.5 * w @ Dmat @ w - kappa @ w

    def gradient(w):
        return Dmat @ w - kappa

    bounds = [(0, B)] * n
    # Constraint: |sum(w) - n| <= eps  =>  n - eps <= sum(w) <= n + eps
    constraints = [
        {"type": "ineq", "fun": lambda w: np.sum(w) - (n - eps)},
        {"type": "ineq", "fun": lambda w: (n + eps) - np.sum(w)},
    ]

    w0 = np.ones(n)
    result = minimize(objective, w0, jac=gradient, bounds=bounds,
                      constraints=constraints, method="SLSQP",
                      options={"maxiter": 1000})

    if not result.success:
        import warnings
        warnings.warn(
            f"kmm_weights: SLSQP did not converge ({result.message}); the "
            "returned weights may be the unadjusted initial guess.", RuntimeWarning)

    return np.clip(result.x, 0, B)
