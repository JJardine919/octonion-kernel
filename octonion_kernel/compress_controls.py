"""Control harness G -- does octonion-multiply transform coding compress real
digit-image data better than a fixed random-rotation baseline and raw
truncation? PCA (fit via SVD) is reported as an adaptive upper-bound
reference, not gated on -- it is the mathematically optimal linear transform
for this exact reconstruction criterion, so it is expected to win; the
question is whether octonion multiply beats the *other fixed* methods.

Pure compute plus loading the bundled (no-network) digits dataset.
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits

from .compress import (
    DEFAULT_GENERATOR, DEFAULT_ROTATION, octonion_encode, octonion_decode,
    rotation_encode, rotation_decode, raw_truncate_encode,
)


def load_and_split_digits(seed: int = 0, test_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    data = load_digits()
    images = data.images.astype(float).reshape(len(data.images), 64)
    rng = np.random.default_rng(seed)
    n = len(images)
    idx = rng.permutation(n)
    n_test = int(n * test_frac)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    train_chunks = images[train_idx].reshape(-1, 8)
    test_chunks = images[test_idx].reshape(-1, 8)
    return train_chunks, test_chunks


def fit_pca(train_chunks: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    mean = train_chunks.mean(axis=0)
    centered = train_chunks - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return mean, vt[:k]


def pca_encode(chunks: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (chunks - mean) @ components.T


def pca_decode(codes: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return codes @ components + mean


def bootstrap_mean_diff_ci(diffs: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """95% bootstrap CI of the mean of paired per-chunk differences."""
    rng = np.random.default_rng(seed)
    n = len(diffs)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        means[b] = diffs[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def run_compress_control(k: int = 3, seed: int = 0) -> dict:
    train_chunks, test_chunks = load_and_split_digits(seed=seed)

    octo_encoded = octonion_encode(test_chunks, DEFAULT_GENERATOR, k)
    octo_decoded = octonion_decode(octo_encoded, DEFAULT_GENERATOR)

    rot_encoded = rotation_encode(test_chunks, DEFAULT_ROTATION, k)
    rot_decoded = rotation_decode(rot_encoded, DEFAULT_ROTATION)

    raw_encoded = raw_truncate_encode(test_chunks, k)  # decode is identity

    mean, components = fit_pca(train_chunks, k)
    pca_codes = pca_encode(test_chunks, mean, components)
    pca_decoded = pca_decode(pca_codes, mean, components)

    per_chunk_sq_error = {
        "octonion": np.mean((octo_decoded - test_chunks) ** 2, axis=-1),
        "random_rotation": np.mean((rot_decoded - test_chunks) ** 2, axis=-1),
        "raw_truncation": np.mean((raw_encoded - test_chunks) ** 2, axis=-1),
        "pca": np.mean((pca_decoded - test_chunks) ** 2, axis=-1),
    }
    mean_mse = {method: float(errs.mean()) for method, errs in per_chunk_sq_error.items()}

    verdict = {}
    for baseline in ("random_rotation", "raw_truncation"):
        diff = per_chunk_sq_error[baseline] - per_chunk_sq_error["octonion"]
        lo, hi = bootstrap_mean_diff_ci(diff, seed=seed + 1)
        verdict[baseline] = {
            "mean_advantage": float(diff.mean()), "ci_lo": lo, "ci_hi": hi,
            "octonion_wins": bool(lo > 0.0),
        }

    octonion_beats_fixed_baselines = bool(
        verdict["random_rotation"]["octonion_wins"] and verdict["raw_truncation"]["octonion_wins"]
    )

    return {
        "mean_mse": mean_mse,
        "verdict_by_baseline": verdict,
        "octonion_beats_fixed_baselines": octonion_beats_fixed_baselines,
        "test_chunks": test_chunks,
        "codes": {
            "octonion": octo_encoded,
            "random_rotation": rot_encoded,
            "raw_truncation": raw_encoded,
            "pca": pca_codes,
        },
    }
