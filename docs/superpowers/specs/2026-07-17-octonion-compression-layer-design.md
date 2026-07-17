# Octonion Compression Layer + Control — Design Spec

**Date:** 2026-07-17
**Author:** James Jardine (Lattice24 / 2396179 Alberta Inc.)
**Status:** Approved design — Phase 5 of the "octonion product" build.

---

## 1. Context & motivation

Phases 1-4 tested the Jordan-Shadow decomposition as a passive information carrier (Phases 1-3,
all NO) and as an active search-guidance signal inside a real optimizer (Phase 4, NO — engine
0-for-4). This phase abandons the search-heuristic paradigm entirely and asks a different kind of
question: does raw **octonion multiplication**, used as a fixed transform-coding basis, compress
real data (rebuild-from-fewer-numbers) better than standard fixed or adaptive baselines?

This is a genuinely different mechanism than Phase 4's associator-based move-scoring. It leans on a
specific, real, distinctive property of octonions: **alternativity**. Octonions are not
associative in general, but they are alternative — any two elements generate an associative
sub-algebra — which guarantees every nonzero octonion `g` has a two-sided inverse with
`(a·g)·g⁻¹ = a` exactly. That gives a clean, honest transform-coding scheme: transform a chunk by
multiplying with a fixed `g`, throw away the smallest coefficients, and invert — the reconstruction
error is purely a function of what was thrown away, not an artifact of a lossy or approximate
inverse.

The governing principle from Phase 1 carries forward unchanged:

> **Every layer ships with its own control/baseline before anything is built on top of it.**
> The layer either provably does something beyond a trivial operation, or we learn that it
> doesn't — early, cheaply. A NO is a valid, expected outcome.

### Where this sits in the stack
1. Algebra kernel — *done (Phase 1, NO)*
2. Dynamics (autonomous walk, linear separability) — *done (Phase 2, NO)*
3. Topology (persistent homology over walk trajectory) — *done (Phase 3, NO)*
4. Optimizer (shadow-guided move-proposal in simulated annealing) — *done (Phase 4, NO)*
5. **Compression layer (this spec)** — octonion multiply as a fixed transform-coding basis,
   gated by a control, against PCA/random-rotation/raw-truncation baselines

## 2. Goal & non-goals

**Goal:** a small, pure Python module implementing octonion-multiply transform coding (encode via
`chunk·g`, top-k-magnitude truncation, decode via `·g⁻¹`) on real 8×8 digit images
(`sklearn.datasets.load_digits`), plus a control harness that honestly compares reconstruction
fidelity against a fixed random-orthogonal-rotation baseline, raw truncation, and PCA (as an
adaptive upper-bound reference) — plus an information-density metric that folds in Phase 3's
persistent-homology tooling.

**Non-goals (Phase 5):**
- No claim that reconstruction quality generalizes beyond 8×8 grayscale digit images; this is one
  concrete real dataset, not a general compression benchmark suite.
- No new sklearn dependency — it's already a harness-level dependency since Phase 2
  (`dynamics_controls.py`). PCA is implemented via `np.linalg.svd` directly, not
  `sklearn.decomposition.PCA`, so there is no hidden centering/whitening behavior to account for.
- No attempt to fix the zero-padding-pattern-affects-topology caveat (§5.2) — documented as a
  known limitation, not engineered around.
- No revisiting of the search-heuristic paradigm (Phase 4 is closed).

## 3. Scope boundary & dependency posture

- `octonion_kernel/compress.py` is a **pure** module: numpy + `octonion_kernel.octonion`
  (`_cd_mul`) only. No I/O, no scipy, no sklearn, no `aoi_collapse`.
- `octonion_kernel/compress_controls.py` is **harness** code: may import scipy and sklearn (same
  boundary every prior phase established — `load_digits` is a bundled, no-network dataset, and
  PCA is implemented via `np.linalg.svd`, not sklearn, kept in the harness only because it
  orchestrates data loading and the train/test split). Pure compute (returns dicts), no I/O beyond
  loading the bundled dataset.
- `harness_report.py` gains a `[G]` section.
- Dependency direction unchanged: kernel → nothing; compress engine → kernel; compress control →
  compress engine + `octonion_kernel.topology` (reused unchanged) + compress engine; report → all.

## 4. The engine — transform coding via octonion multiply

**Chunking:** each 8×8 (64-pixel) digit image reshapes into 8 chunks of 8 pixels (matching the
existing `CHUNK_SIZE=8` convention), each chunk treated as an 8-vector.

**Fixed constants (pinned in code, never tuned to a result):**
- `g` — a fixed unit octonion (the transform generator).
- `R` — a fixed 8×8 orthogonal matrix, generated once via QR-decomposition of a seeded random
  Gaussian matrix.
- `k = 3` — the compression budget (keep 3 of 8 coefficients).

**Three fixed-transform methods**, all following the same rule — transform, keep the `k`
largest-magnitude coefficients, zero the rest, invert:

1. **Octonion** — `transformed = chunk·g` (via the now-batched `_cd_mul`, vectorized over all
   chunks in one call — no per-chunk Python loop, applying Phase 4's performance lesson from the
   start). Truncate to top-`k` by `|coefficient|`. Decode: `truncated·g⁻¹`, where
   `g⁻¹ = conjugate(g) / ‖g‖²`. Alternativity guarantees `(chunk·g)·g⁻¹ = chunk` exactly when
   nothing was truncated.
2. **Random rotation** — `transformed = chunk @ R`. Same top-`k` truncation rule. Decode:
   `truncated @ Rᵀ` (exact inverse since `R` is orthogonal). The fair "decisive bar": another
   fixed, non-adaptive transform, same truncation rule, no octonion structure.
3. **Raw truncation** — keep the `k` largest raw pixel values directly, zero the rest. No
   transform. The trivial null.

**One adaptive reference method:**

4. **PCA** — fit on training chunks via `np.linalg.svd` to `k` components. Encode:
   `(chunk - mean) @ components.T` (genuinely `k`-dimensional, not zero-padded). Decode:
   `code @ components + mean`. Reported as an upper-bound reference, not gated on — PCA is the
   mathematically optimal linear transform for exactly this reconstruction criterion on this data,
   so it is expected to win; the question is whether octonion multiply beats the *other fixed*
   methods, not whether it beats PCA.

**Data split:** `sklearn.datasets.load_digits()` (1797 images, no network), seeded 80/20
train/test split at the *image* level (so a single image's chunks never cross the split). PCA
fits on training chunks; all four methods are evaluated on the same held-out test chunks.

## 5. The control — built to say NO

### 5.1 Verdict metric
Reconstruction MSE per method on held-out test chunks, at the fixed `k=3`. Paired bootstrap CI
(same method as `optimize_controls.bootstrap_mean_diff_ci`) of
`(baseline_MSE − octonion_MSE)` for each fixed baseline (random rotation, raw truncation) —
positive and excluding zero means octonion wins that comparison. PCA's MSE is reported alongside,
not part of the gated verdict.

### 5.2 Information density (Betti-number-normalized fidelity)
On a **seeded 200-point subsample** of the test set's compressed codes per method (matching Phase
3's existing point-cloud scale — full persistent homology does not scale to the whole ~2,875-chunk
test set), reuse `octonion_kernel.topology.persistence_summary()` unchanged to get `max_h1`. Then:

```
variance_of_original_test_chunks = mean((test_chunks - mean(test_chunks)) ** 2)   # one scalar,
                                                                                    # over every
                                                                                    # element of
                                                                                    # every test
                                                                                    # chunk
reconstruction_fidelity = 1 - (MSE / variance_of_original_test_chunks)   # R^2-like: 1.0 = perfect,
                                                                          # 0.0 = no better than
                                                                          # predicting the mean
information_density = reconstruction_fidelity / (1 + max_h1)
```

**Documented limitation:** the three fixed-transform methods keep 8 numbers with 5 zeroed
per-chunk, and *which* 5 varies chunk-to-chunk (top-k selection is data-adaptive), while PCA's
codes are genuinely 3-dimensional. Ripser measures raw Euclidean distance in whatever ambient
space it is given, so the zero-padding pattern can leave a topological fingerprint on top of
actual signal structure. This is acknowledged, not engineered around — same spirit as Phase 3's
diameter-contraction caveat, which was documented rather than hidden.

### 5.3 Verdict rule
`octonion_beats_fixed_baselines` is `True` iff the paired-bootstrap CI of `(MSE_baseline -
MSE_octonion)` excludes zero on the favorable (octonion-lower) side for **both** random rotation
and raw truncation. Anything else → NO. PCA's MSE and information density are always reported for
context, never gating the verdict.

### 5.4 The control test (anti-castle discipline)
The pytest test asserts only: the harness runs and returns a verdict dict; the boolean is a bool;
every reported MSE and information-density value is finite; the CI bounds are ordered. **It does
not assert octonion wins or loses.** A NO is a valid outcome.

## 6. Components & interfaces

`octonion_kernel/compress.py` (pure):
- `DEFAULT_GENERATOR: np.ndarray` — the pinned unit octonion `g` (shape `(8,)`).
- `DEFAULT_ROTATION: np.ndarray` — the pinned orthogonal matrix `R` (shape `(8, 8)`).
- `octonion_inverse(g: np.ndarray) -> np.ndarray`
- `truncate_top_k(transformed: np.ndarray, k: int) -> np.ndarray` — zeros all but the `k`
  largest-`|value|` entries per row; works on a batch, shape `(N, 8)`.
- `octonion_encode(chunks: np.ndarray, g: np.ndarray, k: int) -> np.ndarray`,
  `octonion_decode(truncated: np.ndarray, g: np.ndarray) -> np.ndarray`
- `rotation_encode(chunks: np.ndarray, R: np.ndarray, k: int) -> np.ndarray`,
  `rotation_decode(truncated: np.ndarray, R: np.ndarray) -> np.ndarray`
- `raw_truncate_encode(chunks: np.ndarray, k: int) -> np.ndarray` (decode is identity)

`octonion_kernel/compress_controls.py` (harness):
- `load_and_split_digits(seed: int = 0, test_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray]`
  — returns `(train_chunks, test_chunks)`, each `(N, 8)`.
- `fit_pca(train_chunks: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]` — returns
  `(mean, components)` via `np.linalg.svd`.
- `run_compress_control(k: int = 3, seed: int = 0) -> dict` — runs all four methods, returns
  per-method MSE, the paired-bootstrap verdict, and per-method information density.

`harness_report.py`:
- `report_g(...)` — prints the MSE table, verdict, and information-density table, appended after
  `[F]`.

## 7. Testing approach
- `pytest`, seeded numpy RNG.
- `tests/test_compress.py` — `k=8` (no truncation) reconstructs exactly for octonion and rotation
  methods (proving the invertibility claims to ~1e-9); `R` verified orthogonal
  (`R @ R.T ≈ I`); `g` verified nonzero.
- `tests/test_compress_controls.py` — control checks from §5.4, at reduced scale for speed.
- Full suite green, output pristine.

## 8. Definition of done (Phase 5)
- `compress.py` implements all three fixed-transform encode/decode pairs and `octonion_inverse`;
  invertibility checks pass at `k=8`.
- `compress_controls.py` implements the data split, PCA-via-SVD, the four-method comparison, the
  paired-bootstrap verdict, and the information-density metric; the control test passes and is
  agnostic to the verdict value.
- `harness_report.py` prints the `[G]` section with the MSE table, verdict, and information
  density.
- Full `python -m pytest -q` green; `python harness_report.py` produces sections through `[G]`.

## 9. Deferred to later phases
Larger/harder image datasets, alternative values of `k`, a β0-weighted (rather than
β1/max-H1-weighted) information-density variant, hybrid octonion+PCA schemes, and the 24D/96D
embedding. Each gets its own spec, plan, and per-layer control gate if pursued.

## 10. Open questions
None blocking. `g`, `R`, and `k=3` are the fairness-critical declared constants; all pinned in
code and reported. The zero-padding-vs-topology caveat (§5.2) is a documented limitation, not an
open question requiring resolution before implementation.
