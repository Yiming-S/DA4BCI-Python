"""Visualization: PCA/t-SNE scatter before/after DA."""

import numpy as np


def plot_data_comparison(source_data, target_data, Z_s=None, Z_t=None,
                         description="", method="pca"):
    """Plot source/target distributions before and after adaptation.

    Parameters
    ----------
    source_data, target_data : ndarray (n, p)
    Z_s, Z_t : ndarray or None, adapted data.
    description : str
    method : 'pca' or 'tsne'

    Returns
    -------
    dict with 'p1' (before) and optionally 'p2' (after), matching R's
    return names.  Matplotlib figure objects.
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    def to_real(data):
        data = np.asarray(data)
        return data.real if np.iscomplexobj(data) else data

    source_data = to_real(source_data)
    target_data = to_real(target_data)

    def reduce_2d(data, method_name):
        if method_name == "pca":
            coords = PCA(n_components=2).fit_transform(data)
        else:
            from sklearn.manifold import TSNE
            coords = TSNE(n_components=2).fit_transform(data)
        return coords

    # Before
    data_before = np.vstack([source_data, target_data])
    coords_b = reduce_2d(data_before, method)

    fig_b, ax_b = plt.subplots()
    n_s = source_data.shape[0]
    ax_b.scatter(coords_b[:n_s, 0], coords_b[:n_s, 1], alpha=0.6, label="Source")
    ax_b.scatter(coords_b[n_s:, 0], coords_b[n_s:, 1], alpha=0.6, label="Target")
    ax_b.set_title(f"Data Distribution Before {description}")
    ax_b.legend()

    result = {"p1": fig_b}

    if Z_s is not None and Z_t is not None:
        Z_s = to_real(Z_s)
        Z_t = to_real(Z_t)
        data_after = np.vstack([Z_s, Z_t])
        coords_a = reduce_2d(data_after, method)

        fig_a, ax_a = plt.subplots()
        n_s2 = Z_s.shape[0]
        ax_a.scatter(coords_a[:n_s2, 0], coords_a[:n_s2, 1], alpha=0.6, label="Source")
        ax_a.scatter(coords_a[n_s2:, 0], coords_a[n_s2:, 1], alpha=0.6, label="Target")
        ax_a.set_title(f"Data Distribution After {description}")
        ax_a.legend()
        result["p2"] = fig_a

    plt.close("all")
    return result
