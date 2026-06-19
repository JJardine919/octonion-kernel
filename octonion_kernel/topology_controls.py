"""Control harness for the octonion topology layer (Phase 3).

Harness-tier: imports numpy and ripser (via topology). NEVER imported by the
kernel or the pure engine (dynamics.py). No aoi_collapse. Pure compute: returns
dicts, no I/O.
"""
from __future__ import annotations

import numpy as np

from .octonion import Octonion
from .dynamics import renorm, DEFAULT_GENERATOR
from .dynamics_controls import (
    make_random, make_linear_map, make_generic_nonlinear_map,
    make_random_walk_step, octonion_step,
)
from .topology import persistence_summary


def run_map_trajectory(x0: Octonion, step_fn, lam: float = 0.5, steps: int = 256) -> list[Octonion]:
    """Iterate renorm(lam*x + (1-lam)*step_fn(x)) from x0; return the FULL trajectory
    [x0_normalized, x1, ..., x_T]. Halts (returns trajectory so far) on norm < 1e-12."""
    traj = [renorm(x0)]
    for _ in range(steps):
        x = traj[-1]
        y = Octonion(lam * x.coeffs + (1.0 - lam) * step_fn(x).coeffs)
        if y.norm() < 1e-12:
            break
        traj.append(renorm(y))
    return traj


def iid_cloud(rng, n_points: int) -> list[Octonion]:
    """n_points iid-uniform unit octonions (a structureless point-cloud null)."""
    return [make_random(rng) for _ in range(n_points)]


def _bootstrap_mean_diff_ci(a, b, n_boot: int = 2000, seed: int = 0):
    """95% CI of mean(a) - mean(b) over shared (paired) resamples of the index set."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = a[idx].mean() - b[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)
