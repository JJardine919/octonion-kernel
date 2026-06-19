import numpy as np
from octonion_kernel import Octonion, multiply


def _basis(i: int) -> Octonion:
    c = np.zeros(8)
    c[i] = 1.0
    return Octonion(c)


def _rand_oct(rng) -> Octonion:
    return Octonion(rng.standard_normal(8))


E0 = _basis(0)
NEG_E0 = Octonion(np.array([-1.0, 0, 0, 0, 0, 0, 0, 0]))


def test_identity_is_e0():
    rng = np.random.default_rng(0)
    for _ in range(200):
        a = _rand_oct(rng)
        assert (E0 * a).approx_eq(a)
        assert (a * E0).approx_eq(a)


def test_imaginary_units_square_to_minus_one():
    for i in range(1, 8):
        e = _basis(i)
        assert (e * e).approx_eq(NEG_E0)


def test_imaginary_units_anticommute():
    for i in range(1, 8):
        for j in range(1, 8):
            if i != j:
                assert (_basis(i) * _basis(j)).approx_eq(-(_basis(j) * _basis(i)))


def test_norm_is_multiplicative():
    # composition-algebra property: ||a*b|| == ||a|| * ||b||
    rng = np.random.default_rng(1)
    for _ in range(2000):
        a, b = _rand_oct(rng), _rand_oct(rng)
        assert abs((a * b).norm() - a.norm() * b.norm()) < 1e-9


def test_conjugate_gives_norm_squared():
    # a * conj(a) == ||a||^2 (purely real)
    rng = np.random.default_rng(2)
    for _ in range(500):
        a = _rand_oct(rng)
        p = a * a.conjugate()
        assert abs(p.real - a.norm() ** 2) < 1e-9
        assert float(np.linalg.norm(p.vec)) < 1e-9


def test_product_via_operator_matches_function():
    rng = np.random.default_rng(3)
    a, b = _rand_oct(rng), _rand_oct(rng)
    assert (a * b).approx_eq(multiply(a, b))
