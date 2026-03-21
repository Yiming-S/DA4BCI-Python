# DA4BCI (Python)

A Unified Framework for Domain Adaptation in EEG-based Brain-Computer Interfaces.

Python port of the [DA4BCI R package](https://github.com/Yiming-S/DA4BCI), providing a comprehensive suite of domain adaptation methods tailored for EEG-based BCI applications with a unified interface, evaluation metrics, and visualization tools.

## Installation

### From GitHub

```bash
pip install git+https://github.com/Yiming-S/DA4BCI-Python.git
```

### From Source (for development)

```bash
git clone https://github.com/Yiming-S/DA4BCI-Python.git
cd DA4BCI-Python
pip install -e ".[dev]"
```

The `-e` flag installs in editable mode so changes take effect without reinstalling. The `[dev]` extra includes testing dependencies (pytest).

### Requirements

- Python >= 3.9
- numpy >= 1.21
- scipy >= 1.7
- scikit-learn >= 1.0
- matplotlib >= 3.4
- [POT](https://pythonot.github.io/) >= 0.8 (Python Optimal Transport)

## Quick Start

```python
import numpy as np
from da4bci import domain_adaptation

# Simulate EEG data: source and target domains
source = np.random.randn(100, 20)
target = np.random.randn(100, 20) + 0.5

# Apply domain adaptation (unified interface)
result = domain_adaptation(source, target, method="sa", control={"k": 10})
adapted_source = result["weighted_source_data"]
adapted_target = result["target_data"]

# Or call methods directly
from da4bci import domain_adaptation_coral
result = domain_adaptation_coral(source, target, lam=1e-5)
```

## Available Methods

| Method | Function | Description |
|--------|----------|-------------|
| **TCA** | `domain_adaptation_tca` | Transfer Component Analysis |
| **SA** | `domain_adaptation_sa` | Subspace Alignment |
| **CORAL** | `domain_adaptation_coral` | Correlation Alignment |
| **GFK** | `domain_adaptation_gfk` | Geodesic Flow Kernel |
| **MIDA** | `domain_adaptation_mida` | Maximum Independence Domain Adaptation |
| **RD** | `domain_adaptation_riemannian` | Riemannian Distance alignment |
| **ART** | `domain_adaptation_art` | Aligned Riemannian Transport |
| **PT** | `domain_adaptation_pt` | Parallel Transport |
| **OT** | `domain_adaptation_ot` | Entropy-regularized Optimal Transport (Sinkhorn) |
| **M3D** | `domain_adaptation_m3d` | Manifold-based Multi-step Domain Adaptation |

All methods are accessible through the unified `domain_adaptation()` function or can be imported individually.

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
└── plotting.py       # PCA/t-SNE before/after visualization
```

## Evaluation Tools

```python
from da4bci import compute_mmd, compute_energy, compute_wasserstein, compute_mahalanobis
from da4bci import evaluate_shift, distance_summary

# Compare before/after adaptation
result = evaluate_shift(source, target, adapted_source, adapted_target)

# Full distance summary
ds = distance_summary(source, target, include=["MMD", "Energy", "Wasserstein"])
```

---

## Benchmark

### Experimental Design

Replicating the R package's `ParallelTEST` experiment: **10 DA methods x 10 distribution-shift scenarios = 100 experiments**.

Each experiment generates source and target data (1500 samples x 50 features) from different distribution pairs, applies domain adaptation, and measures distribution alignment via MMD, Energy distance, and Mahalanobis distance.

**10 Distribution-Shift Scenarios**:

| ID | Scenario | Source | Target |
|----|----------|--------|--------|
| 1 | Standard Normal | N(0,1) | N(0,1) |
| 2 | Uniform | U(0,1) | U(0,1) |
| 3 | Different Means | N(5,1) | N(-5,1) |
| 4 | Exponential | Exp(1) | Exp(1) |
| 5 | Different SD | N(0,2) | N(0,0.5) |
| 6 | Poisson | Pois(3) | Pois(10) |
| 7 | Student-t | t(5) | t(10) |
| 8 | Binomial | Bin(10,0.3) | Bin(10,0.7) |
| 9 | Normal Variant | N(0,1) | N(0,3) |
| 10 | Normal + Cauchy | N(0,1) | Cauchy(0,1) |

### Runtime

Average runtime across all 10 distributions (1500 x 50 data, single run):

| Method | Avg Time (ms) | Speed Tier |
|--------|------:|-----------|
| CORAL | 0.5 | < 1 ms |
| PT | 0.9 | < 1 ms |
| ART | 3.4 | ~ 3 ms |
| GFK | 5.5 | ~ 6 ms |
| SA | 5.9 | ~ 6 ms |
| RD | 6.9 | ~ 7 ms |
| TCA | 214.8 | ~ 200 ms |
| OT | 346.6 | ~ 350 ms |
| M3D | 6948.5 | ~ 7 s |
| MIDA | 13792.9 | ~ 14 s |

### Python vs R Performance

Benchmarked on the same data (500 x 50), median of 5 runs:

| Method | R (ms) | Python (ms) | Speedup |
|--------|-------:|------------:|--------:|
| MIDA | 3567.9 | 173.1 | **20.6x** |
| M3D | 1750.6 | 158.2 | **11.1x** |
| TCA | 192.1 | 29.7 | **6.5x** |
| Wasserstein | 101.1 | 16.1 | **6.3x** |
| CORAL | 1.7 | 0.3 | **6.0x** |
| Energy | 23.6 | 5.0 | **4.7x** |
| OT | 11.6 | 2.8 | **4.1x** |
| MMD | 19.1 | 4.8 | **4.0x** |
| LW_cov | 0.65 | 0.18 | **3.7x** |
| GFK | 7.2 | 2.4 | **3.0x** |
| PT | 2.5 | 0.8 | **3.0x** |
| Mahalanobis | 1.2 | 0.4 | **3.0x** |
| SA | 5.7 | 2.3 | **2.5x** |
| ART | 7.9 | 6.9 | **1.1x** |
| riem_mean | 99.4 | 73.6 | **1.4x** |

Python is faster across all methods, with speedups from 1.1x to 20.6x. The advantage grows with data size.

### Adaptation Quality

Energy distance reduction % (positive = domain gap reduced, higher is better):

| Method | Normal | Uniform | Diff Mean | Exp | Diff SD | Poisson | t-dist | Binomial | Normal-v2 | Cauchy |
|--------|:------:|:-------:|:---------:|:---:|:-------:|:-------:|:------:|:--------:|:---------:|:------:|
| **TCA** | 94 | 86 | **97** | 92 | 82 | **96** | 73 | **95** | 83 | **93** |
| SA | -- | -- | 93 | -- | 64 | 91 | -- | 88 | 66 | 94 |
| **MIDA** | 47 | -- | **99** | 58 | **97** | **99** | 79 | **100** | **97** | **99** |
| CORAL | 6 | -- | -1 | -- | **96** | 36 | 57 | 0 | 92 | -- |
| GFK | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **ART** | 29 | 25 | **99** | 23 | **97** | **99** | 68 | **98** | **94** | -- |
| **PT** | 29 | 25 | **99** | 23 | **97** | **99** | 68 | **99** | **94** | -- |
| OT | -- | -- | 82 | -- | 30 | 54 | -- | 58 | -- | -- |
| M3D | -- | 12 | 94 | -- | 70 | **99** | -- | 92 | 43 | 88 |

`--` indicates the method increased the gap on that distribution (negative reduction).

Mahalanobis distance reduction %:

| Method | Normal | Uniform | Diff Mean | Exp | Diff SD | Poisson | t-dist | Binomial | Normal-v2 | Cauchy |
|--------|:------:|:-------:|:---------:|:---:|:-------:|:-------:|:------:|:--------:|:---------:|:------:|
| TCA | 97 | 78 | 74 | 1 | -- | 59 | -- | 36 | -- | -- |
| **SA** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** |
| MIDA | **100** | -- | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** |
| CORAL | -2 | -- | -1 | -- | -19 | 48 | -4 | -1 | -1 | -8 |
| GFK | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **ART** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** |
| **PT** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** |
| **OT** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** |
| M3D | **100** | **100** | **100** | **100** | 95 | **100** | **100** | **100** | **100** | **100** |

### Key Findings

1. **TCA has the most robust Energy reduction** across all 10 distributions (73-97%), making it the safest general-purpose choice.

2. **ART and PT achieve perfect Mahalanobis alignment** (100%) and strong Energy reduction on most distributions, but degrade on heavy-tailed Cauchy data.

3. **MIDA excels on structured shifts** (mean shift, count data: 97-100% Energy reduction) but struggles with Uniform distributions.

4. **CORAL and PT are the fastest** (< 1 ms), suitable for real-time BCI applications where latency matters.

5. **Cauchy distributions are the hardest scenario** -- only TCA and MIDA maintain positive Energy reduction. This stress-tests robustness to extreme outliers.

6. **GFK preserves the original distribution** by design (near-zero change in all metrics), as it only performs subspace rotation without explicit alignment.

### Reproducing the Benchmark

```bash
cd benchmark/dist_benchmark
python run_benchmark.py          # full: 10 methods x 10 distributions
python run_benchmark.py --quick  # smoke test: 3 methods x 3 distributions
```

Results are saved to `benchmark/dist_benchmark/results/` including JSON data and before/after PCA scatter plots for all 100 experiments.

## Testing

```bash
pytest tests/ -v    # 148 tests
```

## License

MIT

## Authors

- Yiming Shen (yiming.shen001@umb.edu)
- David Degras (david.degras@umb.edu)
