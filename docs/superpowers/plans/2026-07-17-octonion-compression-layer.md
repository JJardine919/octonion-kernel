# Octonion Compression Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Phase 5 of the octonion-kernel project — octonion-multiply transform coding
(encode via `chunk·g`, top-k truncation, decode via `·g⁻¹`, exact by alternativity) on real
8×8 digit images, with a paired control against a fixed random-rotation baseline, raw truncation,
and PCA (adaptive reference), plus a Betti-number-normalized information-density metric.

**Architecture:** `octonion_kernel/compress.py` (pure engine — the three fixed-transform
encode/decode pairs, vectorized over batches of chunks). `octonion_kernel/compress_controls.py`
(harness — data loading/split, PCA via SVD, the four-method MSE comparison, paired-bootstrap
verdict, information density). A small backward-compatible refactor to
`octonion_kernel/topology.py` exposes a raw-array persistent-homology path (Phase 3's version only
accepts 8-dimensional `Octonion`-wrapped trajectories; Phase 5's PCA codes are 3-dimensional).
`harness_report.py` gains a `[G]` section.

**Tech Stack:** Python, numpy, `sklearn.datasets.load_digits` (bundled, no network — sklearn is
already a harness-level dependency since Phase 2), `ripser` (already a harness-level dependency
since Phase 3). PCA via `np.linalg.svd` directly, not `sklearn.decomposition.PCA`. pytest.

## Global Constraints

- `compress.py` is a **pure** module: numpy + `octonion_kernel.octonion` (`_cd_mul`) only. No I/O,
  no scipy, no sklearn, no `aoi_collapse`. (Spec §3)
- `compress_controls.py` is harness code: pure compute (returns dicts) plus loading the bundled
  `load_digits` dataset; may import sklearn (for the dataset only) and reuses `topology.py`.
  (Spec §3)
- Fixed, pre-declared constants, never tuned to a result: `DEFAULT_GENERATOR` (a pinned unit
  octonion), `DEFAULT_ROTATION` (a pinned 8×8 orthogonal matrix via seeded QR), `k = 3`. (Spec §4)
- Data split: seeded 80/20 train/test at the *image* level, so one image's chunks never cross the
  split. PCA fits on training chunks; all four methods evaluate on the same held-out test chunks.
  (Spec §4)
- Verdict gates only on beating **both** fixed baselines (random rotation, raw truncation) with a
  paired-bootstrap CI excluding zero; PCA is reported for context, never gates the verdict.
  (Spec §5.1, §5.3)
- Information density: `reconstruction_fidelity = 1 - (MSE / variance_of_original_test_chunks)`,
  `information_density = reconstruction_fidelity / (1 + max_h1)`, computed on a seeded 200-point
  subsample of each method's test-set codes. (Spec §5.2)
- Control tests must be agnostic to which method wins. (Spec §5.4)

---

## File Structure

- `octonion_kernel/compress.py` — **create**. `DEFAULT_GENERATOR`, `DEFAULT_ROTATION`,
  `octonion_inverse`, `truncate_top_k`, the three fixed-transform encode/decode pairs.
- `tests/test_compress.py` — **create**. Engine correctness tests.
- `octonion_kernel/topology.py` — **modify**. Extract `persistence_summary_from_cloud(cloud)` from
  the existing `persistence_summary`, which becomes a thin wrapper over it (backward-compatible).
- `tests/test_topology.py` — **modify**. Add tests for the new raw-array function.
- `octonion_kernel/compress_controls.py` — **create**. Data split, PCA-via-SVD, the four-method
  comparison, paired-bootstrap verdict, information density.
- `tests/test_compress_controls.py` — **create**. Control-test discipline.
- `harness_report.py` — **modify**. Add `report_g()` and wire it into `__main__`.

---

### Task 1: Engine primitives — generator, rotation, inverse, truncation, encode/decode

**Files:**
- Create: `octonion_kernel/compress.py`
- Test: `tests/test_compress.py`

**Interfaces:**
- Consumes: `octonion_kernel.octonion._cd_mul(x, y)` (batched-capable Cayley-Dickson product,
  operates on the last axis — see `octonion_kernel/octonion.py`).
- Produces: `DEFAULT_GENERATOR: np.ndarray` (shape `(8,)`), `DEFAULT_ROTATION: np.ndarray` (shape
  `(8, 8)`), `octonion_inverse(g: np.ndarray) -> np.ndarray`,
  `truncate_top_k(transformed: np.ndarray, k: int) -> np.ndarray`,
  `octonion_encode(chunks: np.ndarray, g: np.ndarray, k: int) -> np.ndarray`,
  `octonion_decode(truncated: np.ndarray, g: np.ndarray) -> np.ndarray`,
  `rotation_encode(chunks: np.ndarray, R: np.ndarray, k: int) -> np.ndarray`,
  `rotation_decode(truncated: np.ndarray, R: np.ndarray) -> np.ndarray`,
  `raw_truncate_encode(chunks: np.ndarray, k: int) -> np.ndarray` (decode is identity, no separate
  function).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_compress.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_compress.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'octonion_kernel.compress'`

- [ ] **Step 3: Write the implementation**

```python
# octonion_kernel/compress.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_compress.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add octonion_kernel/compress.py tests/test_compress.py
git commit -m "feat: Phase 5 octonion transform-coding engine (encode/decode + baselines)"
```

---

### Task 2: Expose a raw-array persistent-homology path in topology.py

**Files:**
- Modify: `octonion_kernel/topology.py`
- Modify: `tests/test_topology.py`

**Interfaces:**
- Produces: `persistence_summary_from_cloud(cloud: np.ndarray) -> dict` (same return shape as the
  existing `persistence_summary`: `max_h1`, `total_h1`, `n_h1`, `total_h0`, `diameter`,
  `max_h1_norm`). `persistence_summary(traj: list[Octonion]) -> dict` keeps its exact existing
  signature and behavior, now implemented as a thin wrapper.

**Why:** Phase 3's `persistence_summary` only accepts 8-dimensional `Octonion`-wrapped
trajectories. Phase 5 needs the same persistent-homology computation on raw point clouds of
*varying* dimensionality (PCA's codes are 3-dimensional, not 8) — a targeted, backward-compatible
refactor rather than a parallel reimplementation.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_topology.py
from octonion_kernel.topology import persistence_summary_from_cloud


def test_persistence_summary_from_cloud_matches_octonion_wrapper():
    traj = _circle_traj(40)
    cloud = trajectory_cloud(traj)
    assert persistence_summary_from_cloud(cloud) == persistence_summary(traj)


def test_persistence_summary_from_cloud_works_on_lower_dimensional_clouds():
    rng = np.random.default_rng(0)
    cloud_3d = rng.standard_normal((30, 3))
    s = persistence_summary_from_cloud(cloud_3d)
    assert set(s) == {"max_h1", "total_h1", "n_h1", "total_h0", "diameter", "max_h1_norm"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_topology.py -v -k from_cloud`
Expected: FAIL with `ImportError: cannot import name 'persistence_summary_from_cloud'`

- [ ] **Step 3: Refactor the implementation**

In `octonion_kernel/topology.py`, replace the existing `persistence_summary` function with:

```python
def persistence_summary_from_cloud(cloud: np.ndarray) -> dict:
    """Persistent-homology summary of a raw point cloud (Euclidean, maxdim=1). Same
    computation as persistence_summary, on any (N, d) array directly -- used by
    Phase 5's compression codes, which aren't always 8-dimensional (e.g. PCA's
    k-dimensional codes).

    Returns:
      max_h1   - single longest H1 (loop) lifetime, 0.0 if none [VERDICT METRIC]
      total_h1 - sum of all H1 lifetimes (context)
      n_h1     - number of H1 features (context)
      total_h0 - sum of finite H0 lifetimes (context)
    """
    dgms = ripser(cloud, maxdim=1)["dgms"]
    h0, h1 = dgms[0], dgms[1]
    if len(h1):
        life1 = h1[:, 1] - h1[:, 0]
        max_h1 = float(np.max(life1))
        total_h1 = float(np.sum(life1))
        n_h1 = int(len(h1))
    else:
        max_h1 = 0.0
        total_h1 = 0.0
        n_h1 = 0
    life0 = h0[:, 1] - h0[:, 0]
    life0 = life0[np.isfinite(life0)]  # drop the one infinite H0 bar
    total_h0 = float(np.sum(life0))
    diameter = float(pdist(cloud).max()) if len(cloud) > 1 else 0.0
    max_h1_norm = float(max_h1 / diameter) if diameter > 1e-12 else 0.0
    return {"max_h1": max_h1, "total_h1": total_h1, "n_h1": n_h1, "total_h0": total_h0,
            "diameter": diameter, "max_h1_norm": max_h1_norm}


def persistence_summary(traj: list[Octonion]) -> dict:
    """Persistent-homology summary of the trajectory point cloud (Euclidean, maxdim=1).
    Thin wrapper over persistence_summary_from_cloud -- see that function for the
    return-value docs."""
    return persistence_summary_from_cloud(trajectory_cloud(traj))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_topology.py -v`
Expected: PASS (all existing Phase-3 topology tests plus the 2 new ones)

- [ ] **Step 5: Commit**

```bash
git add octonion_kernel/topology.py tests/test_topology.py
git commit -m "refactor: expose persistence_summary_from_cloud for non-8D point clouds"
```

---

### Task 3: Data split, PCA via SVD, and the four-method MSE verdict

**Files:**
- Create: `octonion_kernel/compress_controls.py`
- Test: `tests/test_compress_controls.py`

**Interfaces:**
- Consumes: `DEFAULT_GENERATOR`, `DEFAULT_ROTATION`, `octonion_encode`, `octonion_decode`,
  `rotation_encode`, `rotation_decode`, `raw_truncate_encode` from `octonion_kernel.compress`
  (Task 1).
- Produces: `load_and_split_digits(seed: int = 0, test_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray]`,
  `fit_pca(train_chunks: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]`,
  `pca_encode(chunks, mean, components) -> np.ndarray`,
  `pca_decode(codes, mean, components) -> np.ndarray`,
  `bootstrap_mean_diff_ci(diffs: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]`,
  `run_compress_control(k: int = 3, seed: int = 0) -> dict` returning
  `{"mean_mse": {method: float}, "verdict_by_baseline": {baseline: {...}},
  "octonion_beats_fixed_baselines": bool, "test_chunks": np.ndarray,
  "codes": {method: np.ndarray}}` (this task's version has no `"information_density"` key yet --
  Task 4 adds it).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_compress_controls.py
import numpy as np

from octonion_kernel.compress_controls import (
    load_and_split_digits, fit_pca, pca_encode, pca_decode, run_compress_control,
)


def test_load_and_split_digits_shapes():
    train_chunks, test_chunks = load_and_split_digits(seed=0, test_frac=0.2)
    assert train_chunks.shape[1] == 8
    assert test_chunks.shape[1] == 8
    assert train_chunks.shape[0] + test_chunks.shape[0] == 1797 * 8


def test_load_and_split_digits_deterministic():
    a1, b1 = load_and_split_digits(seed=5)
    a2, b2 = load_and_split_digits(seed=5)
    assert np.array_equal(a1, a2)
    assert np.array_equal(b1, b2)


def test_fit_pca_shapes_and_orthonormal_components():
    rng = np.random.default_rng(9)
    train = rng.standard_normal((200, 8))
    mean, components = fit_pca(train, k=3)
    assert mean.shape == (8,)
    assert components.shape == (3, 8)
    assert np.allclose(components @ components.T, np.eye(3), atol=1e-9)


def test_pca_roundtrip_at_k8_is_near_exact():
    rng = np.random.default_rng(10)
    train = rng.standard_normal((200, 8))
    mean, components = fit_pca(train, k=8)
    codes = pca_encode(train, mean, components)
    decoded = pca_decode(codes, mean, components)
    assert np.allclose(decoded, train, atol=1e-7)


def test_run_compress_control_returns_well_formed_result():
    result = run_compress_control(k=3, seed=0)
    assert set(result["mean_mse"].keys()) == {
        "octonion", "random_rotation", "raw_truncation", "pca"
    }
    for mse in result["mean_mse"].values():
        assert np.isfinite(mse) and mse >= 0.0
    for baseline in ("random_rotation", "raw_truncation"):
        v = result["verdict_by_baseline"][baseline]
        assert isinstance(v["octonion_wins"], bool)
        assert v["ci_lo"] <= v["ci_hi"]
    assert isinstance(result["octonion_beats_fixed_baselines"], bool)


def test_run_compress_control_deterministic():
    r1 = run_compress_control(k=3, seed=7)
    r2 = run_compress_control(k=3, seed=7)
    assert r1["mean_mse"] == r2["mean_mse"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_compress_controls.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'octonion_kernel.compress_controls'`

- [ ] **Step 3: Write the implementation**

```python
# octonion_kernel/compress_controls.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_compress_controls.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add octonion_kernel/compress_controls.py tests/test_compress_controls.py
git commit -m "feat: Phase 5 data split, PCA-via-SVD, and four-method MSE verdict"
```

---

### Task 4: Information density (Betti-number-normalized fidelity)

**Files:**
- Modify: `octonion_kernel/compress_controls.py`
- Modify: `tests/test_compress_controls.py`

**Interfaces:**
- Consumes: `persistence_summary_from_cloud` from `octonion_kernel.topology` (Task 2); the
  `"test_chunks"` and `"codes"` keys `run_compress_control` already returns (Task 3).
- Produces: `run_compress_control`'s return dict gains an `"information_density"` key:
  `{method: {"reconstruction_fidelity": float, "max_h1": float, "information_density": float}}`
  for each of the 4 methods.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_compress_controls.py
def test_run_compress_control_information_density_is_well_formed():
    result = run_compress_control(k=3, seed=0)
    density = result["information_density"]
    assert set(density.keys()) == {"octonion", "random_rotation", "raw_truncation", "pca"}
    for method_stats in density.values():
        assert np.isfinite(method_stats["reconstruction_fidelity"])
        assert np.isfinite(method_stats["max_h1"])
        assert method_stats["max_h1"] >= 0.0
        assert np.isfinite(method_stats["information_density"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_compress_controls.py -v -k information_density`
Expected: FAIL with `KeyError: 'information_density'`

- [ ] **Step 3: Write the implementation**

In `octonion_kernel/compress_controls.py`, add the import and a helper, then extend
`run_compress_control`'s return:

```python
# add to the imports at the top of compress_controls.py
from .topology import persistence_summary_from_cloud


# add this function above run_compress_control
def _information_density(codes: np.ndarray, mse: float, variance: float,
                          sample_size: int = 200, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = codes.shape[0]
    size = min(sample_size, n)
    idx = rng.choice(n, size=size, replace=False)
    topo = persistence_summary_from_cloud(codes[idx])
    fidelity = 1.0 - (mse / variance)
    density = fidelity / (1.0 + topo["max_h1"])
    return {
        "reconstruction_fidelity": float(fidelity),
        "max_h1": topo["max_h1"],
        "information_density": float(density),
    }
```

Then replace the `return {...}` statement at the end of `run_compress_control` with:

```python
    variance = float(np.var(test_chunks))
    information_density = {
        method: _information_density(codes_array, mean_mse[method], variance, seed=seed + 2)
        for method, codes_array in {
            "octonion": octo_encoded,
            "random_rotation": rot_encoded,
            "raw_truncation": raw_encoded,
            "pca": pca_codes,
        }.items()
    }

    return {
        "mean_mse": mean_mse,
        "verdict_by_baseline": verdict,
        "octonion_beats_fixed_baselines": octonion_beats_fixed_baselines,
        "information_density": information_density,
        "test_chunks": test_chunks,
        "codes": {
            "octonion": octo_encoded,
            "random_rotation": rot_encoded,
            "raw_truncation": raw_encoded,
            "pca": pca_codes,
        },
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_compress_controls.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add octonion_kernel/compress_controls.py tests/test_compress_controls.py
git commit -m "feat: Phase 5 information-density metric via persistent homology"
```

---

### Task 5: Wire the `[G]` report section into `harness_report.py`

**Files:**
- Modify: `harness_report.py:1-14` (docstring + imports), `harness_report.py`'s `__main__` block

**Interfaces:**
- Consumes: `run_compress_control` from `octonion_kernel.compress_controls` (Task 4).

- [ ] **Step 1: Update the module docstring and imports**

Replace the top docstring/import block in `harness_report.py`:

```python
"""Runnable A/B/C/D/E/F/G control report. Run from the repo root: python harness_report.py

A: algebra correctness  (delegates to pytest tests/test_algebra.py)
B: Jordan-Shadow identity residuals over random pairs
C: the anti-castle control — does the associator beat the best trivial baseline?
D: dynamics control — does iterating the octonion walk add linear separability?
E: topology control — does the walk's trajectory carry more loop structure?
F: optimizer control — does shadow-guided move-proposal reach lower SK-model energy
   than random/greedy/generic-nonlinear baselines in simulated annealing?
G: compression control — does octonion-multiply transform coding reconstruct real
   digit-image data better than a fixed random-rotation baseline and raw truncation?
"""
import numpy as np

from octonion_kernel import Octonion, identity_residuals
from octonion_kernel.controls import run_control_c
from octonion_kernel.dynamics_controls import run_dynamics_control
from octonion_kernel.topology_controls import run_topology_control
from octonion_kernel.optimize_controls import run_optimize_control
from octonion_kernel.compress_controls import run_compress_control
```

- [ ] **Step 2: Add `report_g()`**

Insert immediately after `report_f()` (before the `if __name__ == "__main__":` block):

```python
def report_g(k=3, seed=0):
    out = run_compress_control(k=k, seed=seed)
    print(f"\n[G] Compression control (k={k}, real digit-image chunks, held-out test set):")
    print(f"    {'method':<16} {'mean MSE':>10}")
    for method in ("octonion", "random_rotation", "raw_truncation", "pca"):
        print(f"    {method:<16} {out['mean_mse'][method]:>10.4f}")
    print("\n    verdict vs. each fixed baseline (octonion must beat both):")
    for baseline, v in out["verdict_by_baseline"].items():
        status = "OCTONION WINS" if v["octonion_wins"] else "NO"
        print(f"    vs {baseline:<16} advantage {v['mean_advantage']:>9.4f}  "
              f"95% CI [{v['ci_lo']:.4f}, {v['ci_hi']:.4f}]  {status}")
    print("\n    information density (reconstruction_fidelity / (1 + max_h1)):")
    for method, d in out["information_density"].items():
        print(f"    {method:<16} fidelity {d['reconstruction_fidelity']:>7.4f}  "
              f"max_h1 {d['max_h1']:>7.4f}  density {d['information_density']:>7.4f}")
    if out["octonion_beats_fixed_baselines"]:
        print("\n    VERDICT: YES - octonion transform coding reconstructs real digit-image data")
        print("             more accurately than both fixed baselines (random rotation and raw")
        print("             truncation) at this compression budget.")
    else:
        print("\n    VERDICT: NO - octonion transform coding does NOT beat both fixed baselines.")
        print("             Octonion multiplication as a change of basis is not concentrating")
        print("             reconstructable signal better than a fixed non-octonion transform")
        print("             here. (PCA's MSE is reported above for context only -- it is the")
        print("             optimal linear transform for this criterion by construction.)")
```

- [ ] **Step 3: Wire it into `__main__`**

Replace the `if __name__ == "__main__":` block:

```python
if __name__ == "__main__":
    print("=" * 64)
    print("Octonion kernel control report")
    print("[A] algebra correctness: run `python -m pytest tests/test_algebra.py`")
    report_b()
    report_c()
    report_d()
    report_e()
    # NOTE: run_optimize_control's spec-declared defaults (n_instances=500, steps=5000)
    # take ~5.5 hours at current performance -- reduced here so the report finishes in
    # a few minutes; call report_f(n_instances=500, steps=5000) directly for the
    # full-power run.
    report_f(n_instances=20, steps=500)
    report_g()
    print("=" * 64)
```

- [ ] **Step 4: Run the full suite and the report**

Run: `python -m pytest -q`
Expected: all tests pass, pristine output

Run: `python harness_report.py`
Expected: sections `[B]` through `[G]` all print, `[G]` ending in a recorded verdict (YES or NO)

- [ ] **Step 5: Commit**

```bash
git add harness_report.py
git commit -m "feat: wire [G] compression control section into harness_report.py"
```

---

## Definition of done

- `python -m pytest -q` green (all Phase 1-5 tests).
- `python harness_report.py` prints `[B]` through `[G]`, `[G]` ending in a recorded verdict.
- No scipy/sklearn/aoi imports in `compress.py`; `compress_controls.py` imports sklearn only for
  the bundled dataset loader.
