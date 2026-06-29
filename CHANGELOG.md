# Changelog

All notable changes to this project are documented here.

## [Unreleased]

> **These fixes change numerical outputs of GFK, the Geodesic distance, and any
> RBF-based metric / KMM weight. Re-run any benchmark that uses them before
> relying on previously reported numbers.**

### Fixed (correctness)
- **GFK was not a valid geodesic flow.** The kernel was built from full
  orthonormal bases (`U0.T @ U1`), which is always orthogonal, so the principal
  angles were structurally trivial and the resulting `G` was either the identity
  (a no-op) or numerical noise depending on the dimensions. GFK is now computed
  from the principal angles between the source/target PCA *subspaces* (and their
  complement) with the correct geodesic-flow integral, and features are projected
  by `G**(1/2)`. A regression test asserts `G != I` when the subspaces differ.
- **`compute_geodesic` crashed for unequal sample sizes.** It built the subspace
  basis in *sample* space (`QR`, shape `(n, d)`), so `U.T @ V` required
  `n_s == n_t` and raised otherwise (the normal DA case). It now uses the
  feature-space right singular vectors (shape `(p, d)`), well-defined for any
  sample sizes.
- **`sigma_med` is now reproducible by default.** Its subsampling seed defaulted
  to `None`, so for `n1 + n2 > 400` the RBF bandwidth — and every MMD / Energy /
  KMM value built on it — changed run-to-run and across machines. The default is
  now `seed=0`.

### Fixed (robustness)
- **No more silent no-op alignments.** `matrix_power` now emits a `RuntimeWarning`
  before returning the identity on non-finite or undecomposable input (instead of
  silently making PT/ART a no-op), and ART warns when Riemannian transport fails
  and it falls back to the target covariance.
- **Shape validation** at the `domain_adaptation` dispatcher: 1D inputs or a
  source/target feature-count mismatch now raise a clear `ValueError` instead of a
  cryptic NumPy broadcasting error.
- **M3D** no longer mutates the caller's `stage1`/`stage2` control dicts in place
  (it deep-copies them), so the auto-selected subspace dimension of the first
  dataset no longer freezes across a reuse loop. It also validates that
  `source_labels` matches the number of source rows, raising `ValueError` instead
  of silently leaving one-hot rows all-zero.

### Changed
- M3D: hoisted the loop-invariant `M0` out of the refinement loop and removed an
  unreachable single-stage branch (numerically identical).
- OT (Sinkhorn): the marginal residual is computed from the scaling vectors
  instead of materializing the full `n×m` transport plan every iteration; the plan
  is built once after convergence (same fixed point).

### Added
- `LICENSE` (MIT) — the project declared MIT but shipped no license file.
- GitHub Actions CI (`pyflakes` + `pytest`).

### Repository
- Replaced the accidental one-line `.gitignore` (which ignored `README.md`) with a
  proper Python `.gitignore`; stopped tracking `__pycache__`/`.pyc`/`.DS_Store`.
- Removed dead code (unused `PCA` import in `sa.py`, unused locals in `tca.py`);
  added `__all__` to the package; bumped `requires-python` to `>=3.10`.
