import numpy as np
from octonion_kernel import Octonion, shadow_decompose, identity_residuals, ShadowResult


def _rand_oct(rng):
    return Octonion(rng.standard_normal(8))


def test_shadow_result_fields_and_losslessness():
    rng = np.random.default_rng(0)
    a, b = _rand_oct(rng), _rand_oct(rng)
    r = shadow_decompose(a, b)
    assert isinstance(r, ShadowResult)
    # J + C == a*b exactly (lossless split)
    recombined = r.jordan + r.commutator
    assert recombined.approx_eq(r.product, tol=1e-12)
    # associator == jordan * commutator
    assert r.associator.approx_eq(r.jordan * r.commutator, tol=1e-12)


def test_commutator_is_purely_imaginary():
    # real part of a*b equals real part of b*a, so the commutator has zero real part
    rng = np.random.default_rng(1)
    for _ in range(500):
        a, b = _rand_oct(rng), _rand_oct(rng)
        r = shadow_decompose(a, b)
        assert abs(r.commutator.real) < 1e-10


def test_identity_residuals_within_tolerance_over_10k_pairs():
    # Harness B: the three Jordan-Shadow identities hold over many random pairs.
    rng = np.random.default_rng(2)
    max_loss = max_orth = max_pyth = 0.0
    for _ in range(10_000):
        a, b = _rand_oct(rng), _rand_oct(rng)
        res = identity_residuals(a, b)
        max_loss = max(max_loss, res["losslessness"])
        max_orth = max(max_orth, res["orthogonality"])
        max_pyth = max(max_pyth, res["pythagorean"])
    assert max_loss <= 1e-10, f"losslessness max residual {max_loss}"
    assert max_orth <= 1e-8, f"orthogonality max residual {max_orth}"
    assert max_pyth <= 1e-8, f"pythagorean max residual {max_pyth}"
