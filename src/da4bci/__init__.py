"""DA4BCI: Domain Adaptation for Brain-Computer Interfaces."""

from da4bci.methods import domain_adaptation
from da4bci.methods.tca import domain_adaptation_tca
from da4bci.methods.sa import domain_adaptation_sa
from da4bci.methods.coral import domain_adaptation_coral
from da4bci.methods.gfk import domain_adaptation_gfk
from da4bci.methods.mida import domain_adaptation_mida
from da4bci.methods.riemannian import domain_adaptation_riemannian
from da4bci.methods.art import domain_adaptation_art
from da4bci.methods.pt import domain_adaptation_pt
from da4bci.methods.ot import domain_adaptation_ot
from da4bci.methods.m3d import domain_adaptation_m3d
from da4bci.metrics.kernels import rbf_kernel, sigma_med
from da4bci.metrics.distance import (
    compute_distance_matrix,
    compute_mmd,
    compute_energy,
    compute_wasserstein,
    compute_mahalanobis,
)
from da4bci.geometry.spd import (
    orthonormal_complement,
    LW_covariance,
    matrix_power,
    riemannian_mean,
    log_map,
    exp_map,
    align_riemannian_transport,
    compute_geodesic,
)
from da4bci.preprocessing.alignment import euclidean_alignment
from da4bci.preprocessing.weights import kmm_weights
from da4bci.preprocessing.label_shift import label_shift_em
from da4bci.detection.page_hinkley import ph_init, ph_update
from da4bci.metrics.evaluation import (
    evaluate_shift,
    distance_summary,
    distanceSummary,  # R-compatible alias
    proxy_a_distance,
)
from da4bci.plotting import plot_data_comparison

__version__ = "0.1.0"

__all__ = [
    # Domain-adaptation methods
    "domain_adaptation",
    "domain_adaptation_tca",
    "domain_adaptation_sa",
    "domain_adaptation_coral",
    "domain_adaptation_gfk",
    "domain_adaptation_mida",
    "domain_adaptation_riemannian",
    "domain_adaptation_art",
    "domain_adaptation_pt",
    "domain_adaptation_ot",
    "domain_adaptation_m3d",
    # Metrics
    "rbf_kernel",
    "sigma_med",
    "compute_distance_matrix",
    "compute_mmd",
    "compute_energy",
    "compute_wasserstein",
    "compute_mahalanobis",
    # SPD / Riemannian geometry
    "orthonormal_complement",
    "LW_covariance",
    "matrix_power",
    "riemannian_mean",
    "log_map",
    "exp_map",
    "align_riemannian_transport",
    "compute_geodesic",
    # Preprocessing
    "euclidean_alignment",
    "kmm_weights",
    "label_shift_em",
    # Drift detection
    "ph_init",
    "ph_update",
    # Evaluation
    "evaluate_shift",
    "distance_summary",
    "distanceSummary",
    "proxy_a_distance",
    # Plotting
    "plot_data_comparison",
    "__version__",
]
