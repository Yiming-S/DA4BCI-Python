"""Shared test fixtures."""

import numpy as np
import pytest


@pytest.fixture
def seed42():
    """Return a RandomState with seed 42."""
    return np.random.RandomState(42)


@pytest.fixture
def seed2024():
    """Return a RandomState with seed 2024 (cross-language reference)."""
    return np.random.RandomState(2024)


@pytest.fixture
def src_tgt_42():
    """Standard 20x5 source/target pair (seed 42)."""
    rng = np.random.RandomState(42)
    src = rng.randn(20, 5)
    tgt = rng.randn(20, 5) + 2
    return src, tgt


@pytest.fixture
def src_tgt_large_42():
    """Larger 40x5 source/target pair (seed 42)."""
    rng = np.random.RandomState(42)
    src = rng.randn(40, 5)
    tgt = rng.randn(40, 5) + 2
    return src, tgt


@pytest.fixture
def spd_pair_42():
    """Two SPD matrices of size 5x5 (seed 42)."""
    rng = np.random.RandomState(42)
    M1 = rng.randn(5, 5)
    A = M1.T @ M1 + np.eye(5)
    M2 = rng.randn(5, 5)
    B = M2.T @ M2 + np.eye(5)
    return A, B
