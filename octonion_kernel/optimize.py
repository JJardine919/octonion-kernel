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


def propose_random(state: np.ndarray, J: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.integers(0, len(state)))


def propose_greedy(state: np.ndarray, J: np.ndarray, rng: np.random.Generator) -> int:
    """The 'trivial' baseline: argmax|h_i| directly, no chunking, no octonion structure."""
    h = local_fields(state, J)
    return int(np.argmax(np.abs(h)))


def propose_generic_nonlinear(state: np.ndarray, J: np.ndarray, rng: np.random.Generator) -> int:
    """Fixed, pre-declared elementwise combination score_i = |state_i * h_i|. Same
    per-chunk 16-numbers-per-chunk input as propose_shadow, no octonion algebra."""
    h = local_fields(state, J)
    return int(np.argmax(np.abs(state * h)))


def _shadow_chunk_scores(a_chunk: np.ndarray, b_chunk: np.ndarray) -> np.ndarray:
    """|associator_i| for one 8-spin chunk given its spins (a) and local fields (b)."""
    result = shadow_decompose(Octonion(a_chunk), Octonion(b_chunk))
    return np.abs(result.associator.coeffs)


def propose_shadow(state: np.ndarray, J: np.ndarray, rng: np.random.Generator) -> int:
    """Per chunk, score_i = |associator_i| from shadow_decompose(chunk spins, chunk
    fields); propose the global argmax. Fixed, pre-declared rule -- no open search."""
    n = len(state)
    h = local_fields(state, J)
    n_chunks = n // CHUNK_SIZE
    best_score = -np.inf
    best_idx = 0
    for c in range(n_chunks):
        start = c * CHUNK_SIZE
        end = start + CHUNK_SIZE
        scores = _shadow_chunk_scores(state[start:end], h[start:end])
        local_best = int(np.argmax(scores))
        if scores[local_best] > best_score:
            best_score = scores[local_best]
            best_idx = start + local_best
    return best_idx


def anneal(J: np.ndarray, initial_state: np.ndarray, propose_fn, steps: int = 5000,
           t0: float = 2.0, t_min: float = 0.05, seed: int = 0) -> dict:
    """Single-spin-flip simulated annealing. `propose_fn(state, J, rng) -> int` is the
    only thing that varies between arms; acceptance rule and cooling schedule are shared
    across every arm so the proposal rule is the sole controlled variable."""
    rng = np.random.default_rng(seed)
    state = initial_state.astype(float).copy()
    current_energy = energy(state, J)
    best_energy = current_energy
    if steps <= 1:
        temps = np.full(max(steps, 1), t0)
    else:
        temps = t0 * (t_min / t0) ** (np.arange(steps) / (steps - 1))
    for t in range(steps):
        T = temps[t]
        i = propose_fn(state, J, rng)
        h_i = float(J[i] @ state)
        dE = 2.0 * state[i] * h_i
        if dE <= 0.0 or rng.random() < np.exp(-dE / T):
            state[i] = -state[i]
            current_energy += dE
            if current_energy < best_energy:
                best_energy = current_energy
    return {"best_energy": float(best_energy), "final_state": state}
