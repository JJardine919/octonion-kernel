"""Control harness F -- does shadow-guided proposal find better optima?

Question: does the Jordan-Shadow associator, given exactly the same local information
(each chunk's 8 spins + 8 local fields) as a fixed non-octonion combination, produce a
move-proposal rule that reaches lower SK-model energy than random, greedy local-field,
and generic-nonlinear baselines? Same instance, same initial state, same cooling
schedule and step budget across every arm -- only propose_fn differs.

Before trusting any NO: greedy and generic_nonlinear must each reliably beat random, or
the run has no demonstrated power to separate any method from noise and the shadow
verdict is reported as inconclusive rather than NO.

Pure compute: returns dicts, no I/O.
"""
from __future__ import annotations

import numpy as np

from .optimize import (
    anneal, make_sk_instance, propose_generic_nonlinear, propose_greedy,
    propose_random, propose_shadow,
)

_ARMS = {
    "random": propose_random,
    "greedy": propose_greedy,
    "generic_nonlinear": propose_generic_nonlinear,
    "shadow": propose_shadow,
}
_BASELINE_ARMS = ("random", "greedy", "generic_nonlinear")


def bootstrap_mean_diff_ci(diffs: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """95% bootstrap CI of the mean of paired per-instance differences."""
    rng = np.random.default_rng(seed)
    n = len(diffs)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        means[b] = diffs[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def run_optimize_control(n_instances: int = 500, n: int = 64, steps: int = 5000,
                          seed: int = 0) -> dict:
    energies = {arm: np.empty(n_instances) for arm in _ARMS}
    for k in range(n_instances):
        instance_seed = seed * 1_000_003 + k
        J = make_sk_instance(n=n, seed=instance_seed)
        rng_init = np.random.default_rng(instance_seed + 500_000)
        initial_state = rng_init.choice([-1.0, 1.0], size=n)
        for arm, propose_fn in _ARMS.items():
            result = anneal(J, initial_state, propose_fn, steps=steps, seed=instance_seed)
            energies[arm][k] = result["best_energy"]

    means = {arm: float(energies[arm].mean()) for arm in _ARMS}

    power = {}
    for arm in ("greedy", "generic_nonlinear"):
        diff = energies["random"] - energies[arm]  # positive => arm beats random
        lo, hi = bootstrap_mean_diff_ci(diff, seed=seed + 1)
        power[arm] = {
            "mean_advantage": float(diff.mean()), "ci_lo": lo, "ci_hi": hi,
            "beats_random": bool(lo > 0.0),
        }
    power_ok = bool(power["greedy"]["beats_random"] and power["generic_nonlinear"]["beats_random"])

    best_baseline = min(_BASELINE_ARMS, key=lambda a: means[a])
    diff = energies[best_baseline] - energies["shadow"]  # positive => shadow wins
    diff_lo, diff_hi = bootstrap_mean_diff_ci(diff, seed=seed + 2)
    shadow_wins = bool(power_ok and diff_lo > 0.0)

    return {
        "mean_energy": means,
        "power_check": power,
        "power_check_passed": power_ok,
        "best_baseline": best_baseline,
        "sep_difference_ci": [diff_lo, diff_hi],
        "verdict": {
            "shadow_finds_better_optima": shadow_wins,
            "inconclusive": not power_ok,
        },
    }
