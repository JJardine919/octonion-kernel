import pytest
import numpy as np
from octonion_kernel import Octonion, shadow_decompose, identity_residuals, ShadowResult
from octonion_kernel.shadow import shadow_decompose_batch


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


@pytest.mark.slow
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


def test_shadow_decompose_batch_matches_per_row_scalar_reference():
    # shadow_decompose_batch exists to avoid per-chunk Octonion construction in
    # optimize.py's hot path (propose_shadow), but nothing pinned its output
    # against the known-correct scalar shadow_decompose -- a regression here
    # (e.g. _cd_mul reverting to first-axis slicing for a (n_chunks, 8) batch)
    # would be invisible to the rest of the suite.
    rng = np.random.default_rng(11)
    n_chunks = 5
    a_batch = rng.standard_normal((n_chunks, 8))
    b_batch = rng.standard_normal((n_chunks, 8))

    jordan, commutator, associator, product = shadow_decompose_batch(a_batch, b_batch)

    for i in range(n_chunks):
        expected = shadow_decompose(Octonion(a_batch[i]), Octonion(b_batch[i]))
        assert np.allclose(jordan[i], expected.jordan.coeffs, atol=1e-12)
        assert np.allclose(commutator[i], expected.commutator.coeffs, atol=1e-12)
        assert np.allclose(associator[i], expected.associator.coeffs, atol=1e-12)
        assert np.allclose(product[i], expected.product.coeffs, atol=1e-12)


def test_associator_norm_equals_product_of_component_norms():
    # ||associator|| == ||jordan|| * ||commutator|| EXACTLY, because
    # associator = jordan * commutator and the octonion norm is multiplicative.
    # Consequence: the associator's norm carries no information beyond its two
    # component norms — it cannot separate anything they cannot.
    rng = np.random.default_rng(7)
    for _ in range(1000):
        a, b = _rand_oct(rng), _rand_oct(rng)
        r = shadow_decompose(a, b)
        assert abs(r.associator.norm() - r.jordan.norm() * r.commutator.norm()) < 1e-9
