import numpy as np
from octonion_kernel import Octonion
from octonion_kernel.dynamics import (
    octonion_walk, associator_step, renorm, DEFAULT_GENERATOR,
)


def _rand_oct(rng):
    return Octonion(rng.standard_normal(8))


def test_default_generator_is_unit():
    assert abs(DEFAULT_GENERATOR.norm() - 1.0) < 1e-12


def test_renorm_gives_unit_norm():
    rng = np.random.default_rng(0)
    x = _rand_oct(rng)
    assert abs(renorm(x).norm() - 1.0) < 1e-12


def test_walk_preserves_unit_norm_every_step():
    rng = np.random.default_rng(1)
    for _ in range(50):
        traj = octonion_walk(_rand_oct(rng), steps=16)
        for x in traj:
            assert abs(x.norm() - 1.0) < 1e-10


def test_walk_is_deterministic():
    rng = np.random.default_rng(2)
    x0 = _rand_oct(rng)
    t1 = octonion_walk(x0, lam=0.5, steps=20)
    t2 = octonion_walk(x0, lam=0.5, steps=20)
    assert len(t1) == len(t2)
    for a, b in zip(t1, t2):
        assert a.approx_eq(b, tol=1e-12)


def test_walk_length_is_steps_plus_one():
    rng = np.random.default_rng(3)
    traj = octonion_walk(_rand_oct(rng), steps=32)
    # x0 plus 32 steps, unless an underflow halt occurred (this seed does not halt)
    assert len(traj) == 33


def test_associator_step_is_degree_2_homogeneous():
    # associator(alpha*x, g) == alpha**2 * associator(x, g) -> the map is nonlinear
    rng = np.random.default_rng(4)
    g = _rand_oct(rng)
    for _ in range(200):
        x = _rand_oct(rng)
        alpha = float(rng.standard_normal())
        lhs = associator_step(Octonion(alpha * x.coeffs), g)
        rhs = Octonion(alpha**2 * associator_step(x, g).coeffs)
        assert lhs.approx_eq(rhs, tol=1e-9)


def test_walk_is_finite():
    rng = np.random.default_rng(5)
    for _ in range(50):
        traj = octonion_walk(_rand_oct(rng), steps=32)
        for x in traj:
            assert np.all(np.isfinite(x.coeffs))
