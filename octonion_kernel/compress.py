"""Octonion-multiply transform coding: compress via chunk*g, top-k truncation,
decode via *g^-1. Alternativity (any two octonions generate an associative
sub-algebra) guarantees (a*g)*g^-1 == a exactly for nonzero g -- reconstruction
error is purely a function of what truncation threw away, not an approximate
inverse. Also provides the fixed random-rotation and raw-truncation baselines.

Pure module: numpy + octonion_kernel.octonion (_cd_mul) only. No I/O.
"""
from __future__ import annotations

import numpy as np

from .octonion import _cd_mul

_GENERATOR_RAW = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
DEFAULT_GENERATOR = _GENERATOR_RAW / np.linalg.norm(_GENERATOR_RAW)


def _make_default_rotation(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((8, 8))
    q, _ = np.linalg.qr(a)
    return q


DEFAULT_ROTATION = _make_default_rotation(seed=0)


def octonion_inverse(g: np.ndarray) -> np.ndarray:
    """g^-1 = conjugate(g) / ||g||^2 (the standard octonion inverse formula)."""
    conj = g.copy()
    conj[..., 1:] = -conj[..., 1:]
    norm_sq = float(g @ g)
    return conj / norm_sq


def truncate_top_k(transformed: np.ndarray, k: int) -> np.ndarray:
    """Zero all but the k largest-|value| entries per row. Batched: shape (..., n)."""
    n_features = transformed.shape[-1]
    if k >= n_features:
        return transformed.copy()
    order = np.argsort(-np.abs(transformed), axis=-1)
    keep_idx = order[..., :k]
    mask = np.zeros_like(transformed, dtype=bool)
    np.put_along_axis(mask, keep_idx, True, axis=-1)
    return np.where(mask, transformed, 0.0)


def octonion_encode(chunks: np.ndarray, g: np.ndarray, k: int) -> np.ndarray:
    transformed = _cd_mul(chunks, g)
    return truncate_top_k(transformed, k)


def octonion_decode(truncated: np.ndarray, g: np.ndarray) -> np.ndarray:
    return _cd_mul(truncated, octonion_inverse(g))


def rotation_encode(chunks: np.ndarray, r: np.ndarray, k: int) -> np.ndarray:
    return truncate_top_k(chunks @ r, k)


def rotation_decode(truncated: np.ndarray, r: np.ndarray) -> np.ndarray:
    return truncated @ r.T


def raw_truncate_encode(chunks: np.ndarray, k: int) -> np.ndarray:
    """No transform; decode is identity (the returned array IS the decoded value)."""
    return truncate_top_k(chunks, k)
