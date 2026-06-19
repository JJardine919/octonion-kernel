"""Octonion value type and the Cayley-Dickson product (Baez convention).

This module is the single source of the algebra's structure. It is pure:
no I/O, no global mutable state, no dependency on any AOI-engine code.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Octonion:
    """An octonion: 8 real coefficients [e0, e1, ..., e7]. Immutable."""

    coeffs: np.ndarray

    def __post_init__(self) -> None:
        arr = np.asarray(self.coeffs, dtype=np.float64).reshape(-1)
        if arr.shape != (8,):
            raise ValueError(f"Octonion needs 8 coefficients, got shape {arr.shape}")
        object.__setattr__(self, "coeffs", arr)

    @property
    def real(self) -> float:
        return float(self.coeffs[0])

    @property
    def vec(self) -> np.ndarray:
        return self.coeffs[1:]

    def __add__(self, other: "Octonion") -> "Octonion":
        return Octonion(self.coeffs + other.coeffs)

    def __sub__(self, other: "Octonion") -> "Octonion":
        return Octonion(self.coeffs - other.coeffs)

    def __neg__(self) -> "Octonion":
        return Octonion(-self.coeffs)

    def __mul__(self, other):
        if isinstance(other, Octonion):
            return multiply(self, other)
        return Octonion(self.coeffs * float(other))  # scalar on the right

    def __rmul__(self, other) -> "Octonion":
        return Octonion(self.coeffs * float(other))  # scalar on the left

    def conjugate(self) -> "Octonion":
        c = self.coeffs.copy()
        c[1:] = -c[1:]
        return Octonion(c)

    def norm(self) -> float:
        return float(np.sqrt(self.coeffs @ self.coeffs))

    def approx_eq(self, other: "Octonion", tol: float = 1e-9) -> bool:
        return bool(np.allclose(self.coeffs, other.coeffs, atol=tol))


def multiply(a: "Octonion", b: "Octonion") -> "Octonion":
    """Placeholder — implemented in Task 2."""
    raise NotImplementedError("multiply is implemented in Task 2")
