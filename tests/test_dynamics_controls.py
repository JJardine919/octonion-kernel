import numpy as np
import pytest
from octonion_kernel import Octonion
from octonion_kernel.dynamics import DEFAULT_GENERATOR
from octonion_kernel.dynamics_controls import (
    make_random, make_structured, _STRUCTURE_DIR, _mean_assoc_norm,
    make_linear_map, make_generic_nonlinear_map, make_random_walk_step,
    octonion_step, run_map, run_dynamics_control,
)


def test_generators_unit_norm_and_structured_has_directional_mean():
    rng = np.random.default_rng(0)
    mu = _STRUCTURE_DIR
    structured_proj, random_proj = [], []
    for _ in range(300):
        r = make_random(rng)
        s = make_structured(rng)
        assert abs(r.norm() - 1.0) < 1e-12
        assert abs(s.norm() - 1.0) < 1e-12
        structured_proj.append(float(s.coeffs @ mu))
        random_proj.append(float(r.coeffs @ mu))
    # structured states carry a positive mean projection onto the declared
    # direction (a linear signal); random states are ~zero-mean along it.
    assert np.mean(structured_proj) > 0.2
    assert abs(np.mean(random_proj)) < 0.1


def test_linear_map_is_linear_and_scale_matched():
    g = DEFAULT_GENERATOR
    step = make_linear_map(g, seed=0)
    rng = np.random.default_rng(5)
    x = make_random(rng)
    # linearity: step(2x) == 2*step(x)
    assert Octonion(2.0 * step(x).coeffs).approx_eq(step(Octonion(2.0 * x.coeffs)), tol=1e-9)
    # scale matched within 10%
    tgt = _mean_assoc_norm(g)
    s_rng = np.random.default_rng(7)
    tot = 0.0
    m = 1000
    for _ in range(m):
        tot += step(make_random(s_rng)).norm()
    assert abs(tot / m - tgt) / tgt < 0.1


def test_generic_nonlinear_is_quadratic_and_scale_matched():
    g = DEFAULT_GENERATOR
    step = make_generic_nonlinear_map(g, seed=0)
    rng = np.random.default_rng(6)
    x = make_random(rng)
    # degree-2 homogeneity: step(2x) == 4*step(x)
    assert Octonion(4.0 * step(x).coeffs).approx_eq(step(Octonion(2.0 * x.coeffs)), tol=1e-7)
    tgt = _mean_assoc_norm(g)
    s_rng = np.random.default_rng(8)
    tot = 0.0
    m = 1000
    for _ in range(m):
        tot += step(make_random(s_rng)).norm()
    assert abs(tot / m - tgt) / tgt < 0.1


def test_run_map_preserves_unit_norm_and_is_deterministic():
    g = DEFAULT_GENERATOR
    step = octonion_step(g)
    rng = np.random.default_rng(9)
    x0 = make_random(rng)
    xf1 = run_map(x0, step, steps=16)
    xf2 = run_map(x0, step, steps=16)
    assert abs(xf1.norm() - 1.0) < 1e-10
    assert xf1.approx_eq(xf2, tol=1e-12)


def test_random_walk_step_scale_matched():
    g = DEFAULT_GENERATOR
    step = make_random_walk_step(g, seed=0)
    tgt = _mean_assoc_norm(g)
    rng = np.random.default_rng(11)
    x = make_random(rng)
    # each emitted step term has norm ~ tgt (independent of x)
    norms = [step(x).norm() for _ in range(100)]
    assert abs(np.mean(norms) - tgt) / tgt < 1e-6


@pytest.mark.slow
def test_dynamics_control_runs_and_is_agnostic():
    out = run_dynamics_control(n=300, steps=16, seed=0)
    v = out["verdict"]
    # a verdict is produced regardless of its value (do NOT assert YES/NO)
    assert isinstance(v["octonion_adds_structure"], bool)
    assert v["best_baseline"] in ("raw", "linear", "generic_nonlinear", "random_walk")
    for k in ("raw", "linear", "generic_nonlinear", "random_walk", "octonion"):
        assert 0.0 <= out["aucs"][k] <= 1.0
    # the difference CI is well-formed
    lo, hi = v["auc_difference_ci"]
    assert lo <= hi
