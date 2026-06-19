import numpy as np
import pytest
from octonion_kernel import Octonion
from octonion_kernel.dynamics import DEFAULT_GENERATOR
from octonion_kernel.dynamics_controls import octonion_step, make_random
from octonion_kernel.topology_controls import (
    run_map_trajectory, iid_cloud, _bootstrap_mean_diff_ci,
)


def test_run_map_trajectory_unit_norm_length_and_deterministic():
    step = octonion_step(DEFAULT_GENERATOR)
    rng = np.random.default_rng(0)
    x0 = make_random(rng)
    t1 = run_map_trajectory(x0, step, steps=32)
    t2 = run_map_trajectory(x0, step, steps=32)
    assert len(t1) == 33  # x0 + 32 steps (this seed does not underflow-halt)
    for x in t1:
        assert abs(x.norm() - 1.0) < 1e-10
    for a, b in zip(t1, t2):
        assert a.approx_eq(b, tol=1e-12)


def test_iid_cloud_unit_norm_and_count():
    rng = np.random.default_rng(1)
    cloud = iid_cloud(rng, 50)
    assert len(cloud) == 50
    for x in cloud:
        assert abs(x.norm() - 1.0) < 1e-12


def test_bootstrap_mean_diff_ci_brackets_positive_difference():
    rng = np.random.default_rng(2)
    a = rng.normal(1.0, 0.1, 300)
    b = rng.normal(0.0, 0.1, 300)
    lo, hi = _bootstrap_mean_diff_ci(a, b, n_boot=500, seed=3)
    assert lo <= hi
    assert lo > 0.0  # clearly positive difference, CI excludes 0


from octonion_kernel.topology_controls import run_topology_control


@pytest.mark.slow
def test_topology_control_runs_and_is_agnostic():
    out = run_topology_control(n=30, steps=64, seed=0)
    v = out["verdict"]
    # a verdict is produced regardless of its value (do NOT assert YES/NO)
    assert isinstance(v["octonion_adds_topology"], bool)
    assert v["best_baseline"] in ("linear", "generic_nonlinear", "random_walk")
    for k in ("linear", "generic_nonlinear", "random_walk", "octonion"):
        assert np.isfinite(out["max_h1"][k]) and out["max_h1"][k] >= 0.0
    assert np.isfinite(out["iid_cloud_max_h1"]) and out["iid_cloud_max_h1"] >= 0.0
    lo, hi = v["diff_ci"]
    assert lo <= hi
