"""Jordan-Shadow decomposition (DOI 18690444). Pure: no I/O, no engine imports.

For octonions a, b with product ab and reverse product ba:
    jordan     = (ab + ba) / 2     (symmetric part)
    commutator = (ab - ba) / 2     (antisymmetric part; purely imaginary)
    associator = jordan * commutator
    product    = ab
The split is lossless (jordan + commutator == ab). The kernel asserts only what
this provably IS; it makes no claim about what the parts MEAN.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .octonion import Octonion, multiply


@dataclass(frozen=True)
class ShadowResult:
    jordan: Octonion
    commutator: Octonion
    associator: Octonion
    product: Octonion


def shadow_decompose(a: Octonion, b: Octonion) -> ShadowResult:
    ab = multiply(a, b)
    ba = multiply(b, a)
    jordan = Octonion((ab.coeffs + ba.coeffs) / 2.0)
    commutator = Octonion((ab.coeffs - ba.coeffs) / 2.0)
    associator = multiply(jordan, commutator)
    return ShadowResult(jordan=jordan, commutator=commutator,
                        associator=associator, product=ab)


def identity_residuals(a: Octonion, b: Octonion) -> dict[str, float]:
    r = shadow_decompose(a, b)
    losslessness = float(np.linalg.norm((r.jordan.coeffs + r.commutator.coeffs) - r.product.coeffs))
    orthogonality = float(abs(r.jordan.vec @ r.commutator.vec))
    prod_vec_sq = float(r.product.vec @ r.product.vec)
    j_vec_sq = float(r.jordan.vec @ r.jordan.vec)
    c_vec_sq = float(r.commutator.vec @ r.commutator.vec)
    pythagorean = float(abs(prod_vec_sq - (j_vec_sq + c_vec_sq)))
    return {"losslessness": losslessness,
            "orthogonality": orthogonality,
            "pythagorean": pythagorean}
