"""SK-model instance, energy, and simulated-annealing move-proposal strategies.

Pure module: numpy + the Phase-1 kernel (octonion, shadow) only. No I/O.
"""
from __future__ import annotations

import numpy as np

from .shadow import shadow_decompose_batch

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


def _sample_by_score(scores: np.ndarray, rng: np.random.Generator) -> int:
    """Sample an index with probability proportional to its (non-negative) score.
    Hard argmax deadlocks single-spin-flip SA: once the top-scoring spin's flip is
    thermally rejected, the state hasn't changed, so the same spin is proposed again
    next step -- deterministically, forever, at whatever temperature remains. Score-
    weighted sampling keeps the heuristic's priority while preserving exploration
    diversity, using the `rng` every proposal function already takes."""
    total = scores.sum()
    if total <= 0.0:
        return int(rng.integers(0, len(scores)))
    return int(rng.choice(len(scores), p=scores / total))


def _flip_improvement_scores(state: np.ndarray, h: np.ndarray) -> np.ndarray:
    """max(0, -dE_i) for each spin i: how much flipping spin i would improve (lower)
    the energy, zero if flipping would make it worse. dE_i = 2*state_i*h_i. Using
    raw |h_i| instead conflates a spin that's already correctly settled with one
    that's genuinely frustrated -- |h_i| doesn't encode which side of alignment the
    spin is currently on, only how strongly it feels about it either way."""
    return np.maximum(0.0, -2.0 * state * h)


def propose_greedy(state: np.ndarray, J: np.ndarray, rng: np.random.Generator) -> int:
    """The 'trivial' baseline: sample by flip-improvement magnitude directly, no
    chunking, no octonion structure."""
    h = local_fields(state, J)
    return _sample_by_score(_flip_improvement_scores(state, h), rng)


def propose_generic_nonlinear(state: np.ndarray, J: np.ndarray, rng: np.random.Generator) -> int:
    """Fixed, pre-declared elementwise combination (the same flip-improvement
    formula as propose_greedy -- it's already local per-spin, so chunking doesn't
    change an elementwise computation). Same per-chunk 16-numbers-per-chunk input
    as propose_shadow, no octonion algebra."""
    h = local_fields(state, J)
    return _sample_by_score(_flip_improvement_scores(state, h), rng)


def propose_shadow(state: np.ndarray, J: np.ndarray, rng: np.random.Generator) -> int:
    """Per chunk, score_i = |associator_i| from shadow_decompose(chunk spins, chunk
    fields); sample proportional to score across all n spins. Fixed, pre-declared
    rule -- no open search. Vectorized: all chunks processed in one batched call via
    shadow_decompose_batch, no per-chunk Octonion construction or Python-level loop
    (that construction dominated Phase 4's original ~10hr full-scale runtime)."""
    n = len(state)
    h = local_fields(state, J)
    n_chunks = n // CHUNK_SIZE
    a_batch = state.reshape(n_chunks, CHUNK_SIZE)
    b_batch = h.reshape(n_chunks, CHUNK_SIZE)
    _, _, associator, _ = shadow_decompose_batch(a_batch, b_batch)
    scores = np.abs(associator).reshape(n)
    return _sample_by_score(scores, rng)


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
