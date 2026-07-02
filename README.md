# DA4BCI (Python)

**A Unified Framework for Domain Adaptation in EEG-based Brain-Computer Interfaces.**

<p align="center">
  <img src="docs/cover.svg" alt="DA4BCI-Python cover: source and target EEG feature distributions aligned by domain adaptation methods" width="100%">
</p>

Python implementation of DA4BCI — a comprehensive toolkit of domain adaptation methods, distance metrics, and evaluation tools for EEG-based BCI applications. The package provides a unified interface to align EEG distributions across sessions or subjects, mitigating distributional shift and improving model robustness.

> Looking for the R version? See [DA4BCI](https://github.com/Yiming-S/DA4BCI).

## Installation

Requires Python >= 3.10.

### From GitHub

```bash
pip install git+https://github.com/Yiming-S/DA4BCI-Python.git
```

### From source (for development)

```bash
git clone https://github.com/Yiming-S/DA4BCI-Python.git
cd DA4BCI-Python
pip install -e ".[dev]"
```

The `-e` flag installs in editable mode so changes take effect without reinstalling. The `[dev]` extra includes testing and linting dependencies (pytest, pytest-cov, pyflakes).

### Requirements

- numpy >= 1.21
- scipy >= 1.7
- scikit-learn >= 1.0
- matplotlib >= 3.4
- [POT](https://pythonot.github.io/) >= 0.8 (Python Optimal Transport)

## Quick Start

```python
import numpy as np
from da4bci import domain_adaptation, distance_summary

# Simulate EEG-like source and target features
rng = np.random.default_rng(1)
source = rng.standard_normal((100, 20))
target = rng.standard_normal((100, 20)) + 0.5

# Apply domain adaptation (unified interface)
result = domain_adaptation(source, target, method="coral", control={"lambda": 1e-5})
adapted_source = result["weighted_source_data"]
adapted_target = result["target_data"]

# Quantify distribution alignment
ds = distance_summary(adapted_source, adapted_target,
                      include=["MMD", "Energy", "Wasserstein", "Mahalanobis"])

# Or call methods directly
from da4bci import domain_adaptation_coral
result = domain_adaptation_coral(source, target, lam=1e-5)
```

## Available Methods

All methods are dispatched through the unified `domain_adaptation()` function or can be imported individually.

| Method    | Function                       | Description                                                | `control` keys |
| --------- | ------------------------------ | ---------------------------------------------------------- | -------------- |
| **TCA**   | `domain_adaptation_tca`        | Transfer Component Analysis                                | `k, sigma, mu` |
| **SA**    | `domain_adaptation_sa`         | Subspace Alignment                                         | `k` |
| **CORAL** | `domain_adaptation_coral`      | Correlation Alignment                                      | `lambda` |
| **GFK**   | `domain_adaptation_gfk`        | Geodesic Flow Kernel                                       | `dim_subspace` |
| **MIDA**  | `domain_adaptation_mida`       | Maximum Independence Domain Adaptation                     | `k, sigma, mu, max` |
| **RD**    | `domain_adaptation_riemannian` | Riemannian Distance alignment (`method="rd"`)              | — |
| **ART**   | `domain_adaptation_art`        | Aligned Riemannian Transport                               | — |
| **PT**    | `domain_adaptation_pt`         | Parallel Transport on the SPD manifold                     | — |
| **OT**    | `domain_adaptation_ot`         | Entropy-regularized Optimal Transport (Sinkhorn)           | `eps, maxit, tol, cost` |
| **M3D**   | `domain_adaptation_m3d`        | Manifold-based Multi-step Domain Adaptation                | `source_labels, stage1, stage2` |

Every method returns a dict with at least `weighted_source_data` and `target_data`.

## Evaluation Tools

DA4BCI includes a set of distance metrics for quantitatively assessing the effect of adaptation:

- **Euclidean Distance Matrix** — efficient pairwise distances between datasets.
- **Wasserstein Distance** — minimal transport cost between distributions.
- **Maximum Mean Discrepancy (MMD)** — kernel-based discrepancy, sensitive to subtle shifts.
- **Energy Distance** — empirical-distribution discrepancy from pairwise distances.
- **Mahalanobis Distance** — whitening-aware distance with optional shrinkage covariance.

```python
from da4bci import compute_mmd, compute_energy, compute_wasserstein, compute_mahalanobis
from da4bci import evaluate_shift, distance_summary

# Compare before/after adaptation
result = evaluate_shift(source, target, adapted_source, adapted_target)

# Full distance summary
ds = distance_summary(source, target,
                      include=["MMD", "Energy", "Wasserstein", "Mahalanobis"])
```

## Algorithm Selection

For a practical method-selection guide organized by shift type, data representation, supervision level, and risk, see the [Algorithm Selection Guide](https://github.com/Yiming-S/DA4BCI/blob/main/ALGORITHM_SELECTION_GUIDE.md) in the R repository (the guidance applies identically to this Python implementation).

Quick rules of thumb:

- Start with `SA` and `CORAL` as fast linear baselines.
- If covariance structure dominates, move to `PT` and `ART`; keep `RD` as a lightweight baseline.
- If mismatch looks nonlinear or cluster-wise, try `OT`.
- If you have reliable source labels and need class-aware refinement, try `M3D`.

## Package Structure

```
da4bci/
├── methods/          # 10 DA algorithms + unified interface
│   ├── tca.py, sa.py, coral.py, gfk.py, mida.py
│   ├── riemannian.py, art.py, pt.py, ot.py, m3d.py
│   └── __init__.py   # domain_adaptation() dispatcher
├── metrics/
│   ├── kernels.py    # rbf_kernel, sigma_med
│   ├── distance.py   # MMD, Energy, Wasserstein, Mahalanobis
│   └── evaluation.py # evaluate_shift, proxy_a_distance, distance_summary
├── geometry/
│   └── spd.py        # SPD matrix operations (matrix_power, riemannian_mean, log/exp map)
├── preprocessing/
│   ├── alignment.py  # Euclidean alignment for EEG trials
│   ├── weights.py    # KMM reweighting
│   └── label_shift.py
├── detection/
│   └── page_hinkley.py
└── plotting.py       # PCA / t-SNE before/after visualization
```

## Testing

```bash
pytest tests/ -v    # 154 tests
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

[MIT](LICENSE) © Yiming Shen and David Degras

## Authors

- Yiming Shen — yiming.shen001@umb.edu
- David Degras — david.degras@umb.edu
