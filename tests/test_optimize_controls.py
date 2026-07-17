import numpy as np

from octonion_kernel.optimize_controls import run_optimize_control


def test_run_optimize_control_returns_well_formed_verdict():
    result = run_optimize_control(n_instances=8, n=16, steps=100, seed=0)
    assert set(result["mean_energy"].keys()) == {
        "random", "greedy", "generic_nonlinear", "shadow"
    }
    for energy_val in result["mean_energy"].values():
        assert np.isfinite(energy_val)
    assert result["best_baseline"] in ("random", "greedy", "generic_nonlinear")
    assert isinstance(result["power_check_passed"], bool)
    for arm in ("greedy", "generic_nonlinear"):
        p = result["power_check"][arm]
        assert isinstance(p["beats_random"], bool)
        assert p["ci_lo"] <= p["ci_hi"]
    lo, hi = result["sep_difference_ci"]
    assert lo <= hi
    assert isinstance(result["verdict"]["shadow_finds_better_optima"], bool)
    assert isinstance(result["verdict"]["inconclusive"], bool)


def test_run_optimize_control_deterministic():
    r1 = run_optimize_control(n_instances=6, n=16, steps=50, seed=42)
    r2 = run_optimize_control(n_instances=6, n=16, steps=50, seed=42)
    assert r1["mean_energy"] == r2["mean_energy"]
