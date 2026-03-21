"""Unified domain adaptation interface."""

from da4bci.methods.tca import domain_adaptation_tca
from da4bci.methods.sa import domain_adaptation_sa
from da4bci.methods.mida import domain_adaptation_mida
from da4bci.methods.riemannian import domain_adaptation_riemannian
from da4bci.methods.coral import domain_adaptation_coral
from da4bci.methods.gfk import domain_adaptation_gfk
from da4bci.methods.art import domain_adaptation_art
from da4bci.methods.pt import domain_adaptation_pt
from da4bci.methods.ot import domain_adaptation_ot
from da4bci.methods.m3d import domain_adaptation_m3d


def domain_adaptation(source_data, target_data, method="sa", control=None):
    """Unified entry point for domain adaptation methods.

    Parameters
    ----------
    source_data : ndarray (n_s, p)
    target_data : ndarray (n_t, p)
    method : str
        One of 'tca', 'sa', 'mida', 'rd', 'coral', 'gfk', 'art', 'pt', 'ot', 'm3d'.
    control : dict or None
        Method-specific parameters.

    Returns
    -------
    dict with at least 'weighted_source_data' and 'target_data'.
    """
    if control is None:
        control = {}

    valid = ("tca", "sa", "mida", "rd", "coral", "gfk", "art", "pt", "ot", "m3d")
    if method not in valid:
        raise ValueError(f"Invalid method '{method}'. Must be one of {valid}")

    if method == "tca":
        k = control.get("k", 10)
        sigma = control.get("sigma", 1)
        mu = control.get("mu", 1)
        return domain_adaptation_tca(source_data, target_data, k=k, sigma=sigma, mu=mu)

    elif method == "sa":
        k = control.get("k", 10)
        return domain_adaptation_sa(source_data, target_data, k=k)

    elif method == "mida":
        k = control.get("k", 10)
        sigma = control.get("sigma", 1)
        mu = control.get("mu", 0.1)
        maximize = control.get("max", True)
        return domain_adaptation_mida(source_data, target_data,
                                      k=k, sigma=sigma, mu=mu, maximize=maximize)

    elif method == "rd":
        return domain_adaptation_riemannian(source_data, target_data)

    elif method == "coral":
        lam = control.get("lambda", 1e-5)
        return domain_adaptation_coral(source_data, target_data, lam=lam)

    elif method == "gfk":
        dim_sub = control.get("dim_subspace", 10)
        return domain_adaptation_gfk(source_data, target_data, dim_subspace=dim_sub)

    elif method == "art":
        return domain_adaptation_art(source_data, target_data)

    elif method == "pt":
        return domain_adaptation_pt(source_data, target_data)

    elif method == "ot":
        eps = control.get("eps", 0.05)
        maxit = control.get("maxit", 500)
        tol = control.get("tol", 1e-7)
        cost = control.get("cost", "sqeuclidean")
        return domain_adaptation_ot(source_data, target_data,
                                    eps=eps, maxit=maxit, tol=tol, cost=cost)

    elif method == "m3d":
        if "source_labels" not in control:
            raise ValueError("method='m3d' requires control['source_labels']")
        return domain_adaptation_m3d(
            source_data=source_data,
            source_labels=control["source_labels"],
            target_data=target_data,
            stage1=control.get("stage1", {"method": "tca", "control": {"k": None, "sigma": 1}}),
            stage2=control.get("stage2", {"method": "sa", "control": {"k": 10}}),
            l_iter=control.get("l_iter", 10),
            lambda_ridge=control.get("lambda_ridge", 1e-2),
            eta_kernel=control.get("eta_kernel", 0.1),
            label_offset=control.get("label_offset", 0),
            expl_var=control.get("expl_var", 0.90),
            max_dim=control.get("max_dim", 30),
        )
