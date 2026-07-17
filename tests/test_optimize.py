import numpy as np
import pytest

from octonion_kernel.optimize import (
    make_sk_instance, energy, local_fields,
    propose_random, propose_greedy, propose_generic_nonlinear, propose_shadow,
)

_ALL_PROPOSE_FNS = [propose_random, propose_greedy, propose_generic_nonlinear, propose_shadow]


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


@pytest.mark.parametrize("propose_fn", _ALL_PROPOSE_FNS)
def test_propose_returns_valid_index(propose_fn):
    rng = np.random.default_rng(5)
    J = make_sk_instance(n=64, seed=5)
    state = rng.choice([-1.0, 1.0], size=64)
    for _ in range(10):
        i = propose_fn(state, J, rng)
        assert isinstance(i, int)
        assert 0 <= i < 64


def test_greedy_picks_largest_absolute_field():
    rng = np.random.default_rng(6)
    J = make_sk_instance(n=32, seed=6)
    state = rng.choice([-1.0, 1.0], size=32)
    h = local_fields(state, J)
    i = propose_greedy(state, J, rng)
    assert i == int(np.argmax(np.abs(h)))


def test_generic_nonlinear_picks_largest_state_times_field():
    rng = np.random.default_rng(7)
    J = make_sk_instance(n=32, seed=7)
    state = rng.choice([-1.0, 1.0], size=32)
    h = local_fields(state, J)
    i = propose_generic_nonlinear(state, J, rng)
    assert i == int(np.argmax(np.abs(state * h)))


def test_propose_shadow_selects_within_dominant_chunk():
    # Block-diagonal J: each 8-spin chunk's local field depends only on its own
    # spins, so scaling one chunk's coupling strength way up gives it a much
    # larger-magnitude associator and must make propose_shadow pick inside it.
    n = 16
    rng = np.random.default_rng(20)
    J = np.zeros((n, n))
    J[0:8, 0:8] = make_sk_instance(n=8, seed=100)
    J[8:16, 8:16] = make_sk_instance(n=8, seed=101) * 50.0
    state = rng.choice([-1.0, 1.0], size=n)
    i = propose_shadow(state, J, rng)
    assert 8 <= i < 16
