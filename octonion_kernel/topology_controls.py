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


def run_topology_control(n: int = 200, steps: int = 256, lam: float = 0.5, seed: int = 0) -> dict:
    """Run the topology richness control; return per-map mean max-H1 and the verdict."""
    g = DEFAULT_GENERATOR
    rng = np.random.default_rng(seed)
    x0s = [make_random(rng) for _ in range(n)]

    maps = {
        "linear": make_linear_map(g, seed=seed + 1),
        "generic_nonlinear": make_generic_nonlinear_map(g, seed=seed + 2),
        "random_walk": make_random_walk_step(g, seed=seed + 3),
        "octonion": octonion_step(g),
    }

    per_map_maxh1 = {k: [] for k in maps}
    context = {k: {"total_h1": 0.0, "n_h1": 0.0, "total_h0": 0.0} for k in maps}
    for k, step in maps.items():
        for x0 in x0s:
            s = persistence_summary(run_map_trajectory(x0, step, lam, steps))
            per_map_maxh1[k].append(s["max_h1"])
            context[k]["total_h1"] += s["total_h1"]
            context[k]["n_h1"] += s["n_h1"]
            context[k]["total_h0"] += s["total_h0"]
    for k in maps:
        for key in context[k]:
            context[k][key] /= n

    mean_maxh1 = {k: float(np.mean(per_map_maxh1[k])) for k in maps}

    # iid_cloud sanity null (reported, not gating): clouds of one trajectory's size
    iid_rng = np.random.default_rng(seed + 50)
    iid_vals = [persistence_summary(iid_cloud(iid_rng, steps + 1))["max_h1"] for _ in range(n)]
    iid_cloud_max_h1 = float(np.mean(iid_vals))

    baseline_keys = ("linear", "generic_nonlinear", "random_walk")
    best_baseline = max(baseline_keys, key=lambda k: mean_maxh1[k])
    diff_lo, diff_hi = _bootstrap_mean_diff_ci(
        per_map_maxh1["octonion"], per_map_maxh1[best_baseline], seed=seed + 100)
    adds = bool(mean_maxh1["octonion"] > mean_maxh1[best_baseline] and diff_lo > 0.0)

    return {
        "max_h1": mean_maxh1,
        "context": context,
        "iid_cloud_max_h1": iid_cloud_max_h1,
        "verdict": {
            "octonion_adds_topology": adds,
            "best_baseline": best_baseline,
            "octonion_max_h1": mean_maxh1["octonion"],
            "best_baseline_max_h1": mean_maxh1[best_baseline],
            "diff_ci": [diff_lo, diff_hi],
        },
    }
