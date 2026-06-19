"""Control harness for the octonion dynamics layer (Phase 2).

Harness code: numpy + scipy + sklearn allowed (never imported by the pure
dynamics engine or the kernel). Pure compute: returns dicts, no I/O.

Builds matched baselines (linear, generic-nonlinear, random walk) and asks
whether iterating the octonion walk makes structured-vs-random initial states
more linearly separable than the raw input and than every baseline.
"""
from __future__ import annotations

import numpy as np

from .octonion import Octonion
from .dynamics import associator_step, renorm, DEFAULT_GENERATOR

# declared structured-class subspace dimension (a "great subsphere")
_STRUCTURED_DIM = 3


def make_random(rng) -> Octonion:
    """Uniform on S^7."""
    v = rng.standard_normal(8)
    return Octonion(v / np.linalg.norm(v))


def make_structured(rng) -> Octonion:
    """Unit octonion supported only on the first _STRUCTURED_DIM basis axes."""
    v = np.zeros(8)
    v[:_STRUCTURED_DIM] = rng.standard_normal(_STRUCTURED_DIM)
    return Octonion(v / np.linalg.norm(v))


def _mean_assoc_norm(g: Octonion, seed: int = 12345, n: int = 2000) -> float:
    """E[||associator(x, g)||] over random unit octonions x (for scale matching)."""
    rng = np.random.default_rng(seed)
    tot = 0.0
    for _ in range(n):
        tot += associator_step(make_random(rng), g).norm()
    return tot / n


def _mean_step_norm(step_fn, seed: int = 999, n: int = 2000) -> float:
    """E[||step_fn(x)||] over random unit octonions x."""
    rng = np.random.default_rng(seed)
    tot = 0.0
    for _ in range(n):
        tot += step_fn(make_random(rng)).norm()
    return tot / n


def make_linear_map(g: Octonion, seed: int = 0):
    """Scale-matched random linear step term x -> A x (A rescaled so
    E[||A x||] == E[||associator(x, g)||])."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((8, 8))

    def raw_step(x: Octonion) -> Octonion:
        return Octonion(A @ x.coeffs)

    scale = _mean_assoc_norm(g) / _mean_step_norm(raw_step)
    A = A * scale

    def step(x: Octonion) -> Octonion:
        return Octonion(A @ x.coeffs)

    return step


def make_generic_nonlinear_map(g: Octonion, seed: int = 0):
    """Scale-matched generic random quadratic step term Q(x)_i = x^T T_i x
    (rescaled so E[||Q(x)||] == E[||associator(x, g)||])."""
    rng = np.random.default_rng(seed)
    T = rng.standard_normal((8, 8, 8))  # T[i] is the matrix for output coord i

    def raw_step(x: Octonion) -> Octonion:
        xc = x.coeffs
        return Octonion(np.einsum("ijk,j,k->i", T, xc, xc))

    scale = _mean_assoc_norm(g) / _mean_step_norm(raw_step)

    def step(x: Octonion) -> Octonion:
        xc = x.coeffs
        return Octonion(scale * np.einsum("ijk,j,k->i", T, xc, xc))

    return step


def make_random_walk_step(g: Octonion, seed: int = 0):
    """Fresh scale-matched noise each call (pure diffusion baseline). The emitted
    step term has norm == E[||associator(x, g)||] and ignores x."""
    target = _mean_assoc_norm(g)
    rng = np.random.default_rng(seed)

    def step(x: Octonion) -> Octonion:
        eta = rng.standard_normal(8)
        return Octonion(eta / np.linalg.norm(eta) * target)

    return step


def octonion_step(g: Octonion):
    """The octonion associator step term as a step_fn (uniform interface)."""
    def step(x: Octonion) -> Octonion:
        return associator_step(x, g)
    return step


def run_map(x0: Octonion, step_fn, lam: float = 0.5, steps: int = 32) -> Octonion:
    """Iterate renorm(lam*x + (1-lam)*step_fn(x)) from x0; return the final state.
    Halts (returns current state) if a step underflows (||y|| < 1e-12)."""
    x = renorm(x0)
    for _ in range(steps):
        y = Octonion(lam * x.coeffs + (1.0 - lam) * step_fn(x).coeffs)
        if y.norm() < 1e-12:
            break
        x = renorm(y)
    return x
