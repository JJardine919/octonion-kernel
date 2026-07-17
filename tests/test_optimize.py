import numpy as np
import pytest

from octonion_kernel.optimize import make_sk_instance, energy, local_fields


def _naive_energy(state, J):
    n = len(state)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            total -= J[i, j] * state[i] * state[j]
    return total


def test_make_sk_instance_symmetric_zero_diagonal():
    J = make_sk_instance(n=16, seed=1)
    assert J.shape == (16, 16)
    assert np.allclose(J, J.T)
    assert np.allclose(np.diag(J), 0.0)


def test_make_sk_instance_deterministic():
    J1 = make_sk_instance(n=16, seed=1)
    J2 = make_sk_instance(n=16, seed=1)
    assert np.array_equal(J1, J2)


def test_energy_matches_naive_brute_force():
    rng = np.random.default_rng(2)
    J = make_sk_instance(n=8, seed=2)
    for _ in range(20):
        state = rng.choice([-1.0, 1.0], size=8)
        assert energy(state, J) == pytest.approx(_naive_energy(state, J), abs=1e-9)


def test_energy_matches_all_256_configs_n8():
    J = make_sk_instance(n=8, seed=3)
    for bits in range(256):
        state = np.array([1.0 if (bits >> k) & 1 else -1.0 for k in range(8)])
        assert energy(state, J) == pytest.approx(_naive_energy(state, J), abs=1e-9)


def test_local_field_matches_energy_delta():
    rng = np.random.default_rng(4)
    J = make_sk_instance(n=8, seed=4)
    state = rng.choice([-1.0, 1.0], size=8)
    h = local_fields(state, J)
    for i in range(8):
        flipped = state.copy()
        flipped[i] = -flipped[i]
        expected_delta = energy(flipped, J) - energy(state, J)
        actual_delta = 2.0 * state[i] * h[i]
        assert actual_delta == pytest.approx(expected_delta, abs=1e-9)
