"""Page-Hinkley change detection."""


def ph_init(delta=0.005, lambda_=50, alpha=0.999, **kwargs):
    """Initialize Page-Hinkley state.

    Parameters
    ----------
    delta : float
    lambda_ : float
        Detection threshold.  Named ``lambda_`` to avoid clashing with
        Python's ``lambda`` keyword.  The R-compatible name ``lambda``
        can be passed as a keyword argument via ``**kwargs``.
    alpha : float

    Returns
    -------
    dict state.
    """
    # Accept R-compatible 'lambda' kwarg (``ph_init(**{"lambda": 50})``)
    lam = kwargs.pop("lambda", lambda_)
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {list(kwargs)}")
    return {
        "mean": 0.0,
        "cum": 0.0,
        "min_cum": 0.0,
        "delta": delta,
        "lambda": lam,
        "alpha": alpha,
    }


def ph_update(state, x):
    """Update Page-Hinkley detector with a new observation.

    Parameters
    ----------
    state : dict from ph_init or previous ph_update.
    x : float

    Returns
    -------
    dict with 'state' and 'change' (bool).
    """
    state = dict(state)  # shallow copy
    state["mean"] = state["alpha"] * state["mean"] + (1 - state["alpha"]) * x
    state["cum"] = state["cum"] + (x - state["mean"] - state["delta"])
    state["min_cum"] = min(state["min_cum"], state["cum"])
    changed = (state["cum"] - state["min_cum"]) > state["lambda"]
    return {"state": state, "change": bool(changed)}
