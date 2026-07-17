import numpy as np
import pytest

from octonion_kernel.compress import (
    DEFAULT_GENERATOR, DEFAULT_ROTATION, octonion_inverse, truncate_top_k,
    octonion_encode, octonion_decode, rotation_encode, rotation_decode,
    raw_truncate_encode,
)


def test_default_generator_is_nonzero_unit():
    assert np.linalg.norm(DEFAULT_GENERATOR) == pytest.approx(1.0, abs=1e-9)
    assert not np.allclose(DEFAULT_GENERATOR, 0.0)


def test_default_rotation_is_orthogonal():
    R = DEFAULT_ROTATION
    assert R.shape == (8, 8)
    assert np.allclose(R @ R.T, np.eye(8), atol=1e-9)
    assert np.allclose(R.T @ R, np.eye(8), atol=1e-9)


def test_octonion_inverse_round_trips_generator():
    g_inv = octonion_inverse(DEFAULT_GENERATOR)
    from octonion_kernel.octonion import _cd_mul
    identity = _cd_mul(DEFAULT_GENERATOR, g_inv)
    expected = np.zeros(8)
    expected[0] = 1.0
    assert np.allclose(identity, expected, atol=1e-9)


def test_truncate_top_k_keeps_largest_magnitude_per_row():
    x = np.array([[1.0, -5.0, 2.0, 0.1, 3.0, -0.2, 4.0, -0.5]])
    result = truncate_top_k(x, k=3)
    expected = np.zeros_like(x)
    expected[0, 1] = -5.0
    expected[0, 6] = 4.0
    expected[0, 4] = 3.0
    assert np.array_equal(result, expected)


def test_truncate_top_k_at_full_width_is_unchanged():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 8))
    assert np.array_equal(truncate_top_k(x, k=8), x)


def test_octonion_roundtrip_at_k8_is_exact():
    rng = np.random.default_rng(1)
    chunks = rng.standard_normal((20, 8))
    encoded = octonion_encode(chunks, DEFAULT_GENERATOR, k=8)
    decoded = octonion_decode(encoded, DEFAULT_GENERATOR)
    assert np.allclose(decoded, chunks, atol=1e-9)


def test_rotation_roundtrip_at_k8_is_exact():
    rng = np.random.default_rng(2)
    chunks = rng.standard_normal((20, 8))
    encoded = rotation_encode(chunks, DEFAULT_ROTATION, k=8)
    decoded = rotation_decode(encoded, DEFAULT_ROTATION)
    assert np.allclose(decoded, chunks, atol=1e-9)


def test_octonion_encode_reduces_nonzero_count_at_k3():
    rng = np.random.default_rng(3)
    chunks = rng.standard_normal((10, 8))
    encoded = octonion_encode(chunks, DEFAULT_GENERATOR, k=3)
    assert np.all(np.count_nonzero(encoded, axis=-1) <= 3)


def test_raw_truncate_keeps_largest_raw_pixels():
    x = np.array([[0.0, 9.0, 1.0, 8.0, 2.0, 7.0, 3.0, 0.5]])
    result = raw_truncate_encode(x, k=3)
    expected = np.zeros_like(x)
    expected[0, 1] = 9.0
    expected[0, 3] = 8.0
    expected[0, 5] = 7.0
    assert np.array_equal(result, expected)
