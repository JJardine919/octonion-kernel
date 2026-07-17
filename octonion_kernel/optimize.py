"""SK-model instance, energy, and simulated-annealing move-proposal strategies.

Pure module: numpy + the Phase-1 kernel (octonion, shadow) only. No I/O.
"""
from __future__ import annotations

import numpy as np

from .octonion import Octonion
from .shadow import shadow_decompose

CHUNK_SIZE = 8


def make_sk_instance(n: int = 64, seed: int = 0) -> np.ndarray:
    """Symmetric SK coupling matrix, zero diagonal, J_ij ~ N(0, 1/n) for i<j."""
    rng = np.random.default_rng(seed)
    upper = rng.normal(loc=0.0, scale=1.0 / np.sqrt(n), size=(n, n))
    J = np.zeros((n, n))
    iu = np.triu_indices(n, k=1)
    J[iu] = upper[iu]
    return J + J.T


def energy(state: np.ndarray, J: np.ndarray) -> float:
    """E(s) = -sum_{i<j} J_ij s_i s_j == -0.5 * s @ J @ s (J symmetric, zero diagonal)."""
    return float(-0.5 * state @ J @ state)


def local_fields(state: np.ndarray, J: np.ndarray) -> np.ndarray:
    """h_i = sum_j J_ij s_j. ΔE for flipping spin i is 2 * state[i] * h[i]."""
    return J @ state
