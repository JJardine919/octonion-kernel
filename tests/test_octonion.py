import numpy as np
import pytest
from octonion_kernel import Octonion


def test_construction_and_accessors():
    o = Octonion([1, 2, 3, 4, 5, 6, 7, 8])
    assert o.coeffs.shape == (8,)
    assert o.coeffs.dtype == np.float64
    assert o.real == 1.0
    assert np.array_equal(o.vec, np.array([2, 3, 4, 5, 6, 7, 8], dtype=float))


def test_construction_rejects_wrong_length():
    with pytest.raises(ValueError):
        Octonion([1, 2, 3])


def test_is_frozen():
    o = Octonion([1, 0, 0, 0, 0, 0, 0, 0])
    with pytest.raises(Exception):
        o.coeffs = np.zeros(8)  # frozen dataclass forbids reassignment


def test_add_sub_neg():
    a = Octonion([1, 1, 1, 1, 0, 0, 0, 0])
    b = Octonion([0, 0, 0, 0, 2, 2, 2, 2])
    assert (a + b).approx_eq(Octonion([1, 1, 1, 1, 2, 2, 2, 2]))
    assert (a - a).approx_eq(Octonion(np.zeros(8)))
    assert (-a).approx_eq(Octonion([-1, -1, -1, -1, 0, 0, 0, 0]))


def test_scalar_mul_both_sides():
    a = Octonion([1, 2, 3, 4, 5, 6, 7, 8])
    assert (a * 2).approx_eq(Octonion([2, 4, 6, 8, 10, 12, 14, 16]))
    assert (3 * a).approx_eq(Octonion([3, 6, 9, 12, 15, 18, 21, 24]))


def test_conjugate_and_norm():
    a = Octonion([1, 2, 3, 4, 5, 6, 7, 8])
    assert a.conjugate().approx_eq(Octonion([1, -2, -3, -4, -5, -6, -7, -8]))
    assert abs(a.norm() - np.sqrt(204.0)) < 1e-12  # 1+4+9+...+64 = 204


def test_approx_eq_tolerance():
    a = Octonion([1, 0, 0, 0, 0, 0, 0, 0])
    b = Octonion([1 + 1e-12, 0, 0, 0, 0, 0, 0, 0])
    assert a.approx_eq(b)
    assert not a.approx_eq(Octonion([1.1, 0, 0, 0, 0, 0, 0, 0]))


def test_coeffs_and_vec_are_immutable():
    o = Octonion([1, 2, 3, 4, 5, 6, 7, 8])
    with pytest.raises(ValueError):
        o.coeffs[0] = 99.0
    with pytest.raises(ValueError):
        o.vec[0] = 99.0


def test_construction_copies_input_array():
    src = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
    o = Octonion(src)
    src[0] = 42.0  # mutating the source must not change the octonion
    assert o.real == 1.0


def test_approx_eq_uses_absolute_tolerance_only():
    # large coefficients must not widen the window via numpy's default rtol
    a = Octonion([1e6, 0, 0, 0, 0, 0, 0, 0])
    b = Octonion([1e6 + 1.0, 0, 0, 0, 0, 0, 0, 0])
    assert not a.approx_eq(b)


def test_octonion_mul_raises_until_task2():
    a = Octonion([1, 0, 0, 0, 0, 0, 0, 0])
    with pytest.raises(NotImplementedError):
        _ = a * a
