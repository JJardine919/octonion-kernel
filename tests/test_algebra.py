import json
import os
from collections import Counter

import numpy as np
import pytest
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


def test_nonassociative_for_generic_triples():
    # Octonions are NOT associative: (a*b)*c != a*(b*c) generically.
    rng = np.random.default_rng(10)
    found_nonassoc = False
    for _ in range(100):
        a, b, c = _rand_oct(rng), _rand_oct(rng), _rand_oct(rng)
        if not ((a * b) * c).approx_eq(a * (b * c), tol=1e-9):
            found_nonassoc = True
            break
    assert found_nonassoc, "expected a non-associative triple; these may not be octonions"


def test_moufang_identity_holds():
    # Octonions satisfy the Moufang identities, e.g. a*(b*(a*c)) == ((a*b)*a)*c.
    rng = np.random.default_rng(11)
    for _ in range(500):
        a, b, c = _rand_oct(rng), _rand_oct(rng), _rand_oct(rng)
        lhs = a * (b * (a * c))
        rhs = ((a * b) * a) * c
        assert lhs.approx_eq(rhs, tol=1e-8)


def test_basis_products_have_fano_plane_structure():
    E = [_basis(i) for i in range(8)]
    # e0 is the identity
    for i in range(8):
        assert (E[0] * E[i]).approx_eq(E[i])
        assert (E[i] * E[0]).approx_eq(E[i])
    # imaginary units square to -e0
    for i in range(1, 8):
        assert (E[i] * E[i]).approx_eq(NEG_E0)
    # product of two distinct imaginary units is a single signed imaginary unit
    lines = set()
    for i in range(1, 8):
        for j in range(i + 1, 8):
            p = (E[i] * E[j]).coeffs
            nz = np.nonzero(np.abs(p) > 1e-9)[0]
            assert len(nz) == 1, f"e{i}*e{j} is not a single basis element"
            k = int(nz[0])
            assert k >= 1, "product of imaginaries should be imaginary"
            assert (E[i] * E[j]).approx_eq(-(E[j] * E[i])), f"e{i}*e{j} not anticommutative"
            lines.add(frozenset({i, j, k}))
    # exactly 7 Fano lines
    assert len(lines) == 7
    # each of the 7 points lies on exactly 3 lines
    point_count = Counter()
    for line in lines:
        for p in line:
            point_count[p] += 1
    assert all(point_count[p] == 3 for p in range(1, 8))
    # each of the 21 point-pairs lies on exactly one line
    pair_count = Counter()
    for line in lines:
        pts = sorted(line)
        for x in range(3):
            for y in range(x + 1, 3):
                pair_count[(pts[x], pts[y])] += 1
    assert len(pair_count) == 21
    assert all(v == 1 for v in pair_count.values())


def test_multiply_matches_pinned_table():
    path = os.path.join(os.path.dirname(__file__), "fixtures", "basis_table.json")
    with open(path) as f:
        table = json.load(f)
    for i in range(8):
        for j in range(8):
            product = _basis(i) * _basis(j)
            nz = [(round(float(product.coeffs[k])), k)
                  for k in range(8) if abs(product.coeffs[k]) > 1e-9]
            expected = [tuple(entry) for entry in table[i][j]]
            assert nz == expected, f"e{i}*e{j} drifted from pinned table"


def test_legacy_cross_check_if_available():
    # Optional: if the legacy AOI engine is importable, its shadow decomposition's
    # product must agree with ours up to a documented basis isomorphism. Skipped
    # (never a dependency) when the module is absent.
    import importlib
    try:
        legacy = importlib.import_module("aoi_collapse")
    except Exception:
        pytest.skip("aoi_collapse not importable; legacy cross-check skipped")
    if not hasattr(legacy, "octonion_shadow_decompose"):
        pytest.skip("legacy octonion_shadow_decompose not present")
    # If present, assert agreement on the product norm (basis-convention agnostic):
    rng = np.random.default_rng(99)
    a, b = _rand_oct(rng), _rand_oct(rng)
    ours = (a * b).norm()
    legacy_result = legacy.octonion_shadow_decompose(a.coeffs, b.coeffs)
    # legacy returns a dict-like with a 'product' 8-vector; compare norms only.
    legacy_prod = np.asarray(legacy_result["product"], dtype=float).reshape(-1)
    assert abs(float(np.linalg.norm(legacy_prod)) - ours) < 1e-6
