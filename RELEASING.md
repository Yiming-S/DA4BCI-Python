# Releasing da4bci

da4bci publishes to PyPI via GitHub Actions **Trusted Publishing** (see
`.github/workflows/release.yml`). No API tokens are stored in the repo.

## One-time setup

1. Create a PyPI account. The first upload claims the project name `da4bci`.
2. On PyPI → your project (or "pending publishers") → **Publishing** → add a
   Trusted Publisher:
   - Owner: `Yiming-S`, Repository: `DA4BCI-Python`
   - Workflow: `release.yml`
   - Environment: `pypi`
3. In the GitHub repo → **Settings → Environments** → create an environment named
   `pypi`.

## Each release

1. Bump `version` in `pyproject.toml` and `__version__` in `src/da4bci/__init__.py`.
2. Update `CHANGELOG.md`.
3. Commit, then tag and push:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```
4. The `release.yml` workflow builds the sdist + wheel and publishes to PyPI.

## Dry run (optional)

To rehearse on TestPyPI first, add `repository-url: https://test.pypi.org/legacy/`
to the publish step and configure a matching TestPyPI trusted publisher.
