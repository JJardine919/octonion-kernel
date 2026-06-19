# Octonion Algebra Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small, standalone, provably-correct Python library implementing 8-dimensional octonion algebra and the Jordan-Shadow decomposition (DOI 18690444), plus an A/B/C control harness whose decisive test (C) honestly measures whether the associator carries information beyond a trivial magnitude statistic.

**Architecture:** A pure `octonion_kernel` package — `octonion.py` (immutable value type + Cayley–Dickson product), `shadow.py` (Jordan-Shadow decomposition + identity residuals), `controls.py` (AUC, bootstrap CI, the control-C experiment). The kernel imports nothing of the wider AOI engine; an I/O-only `harness_report.py` script at the repo root imports the kernel and prints the full A/B/C report. The dependency direction is one-way: kernel → nothing; harness/report → kernel.

**Tech Stack:** Python 3.14, numpy (kernel runtime dependency — the only one), scipy (`scipy.stats.rankdata`, used by the harness for tie-correct ranking — NOT by the kernel), pytest.

## Global Constraints

- **Python 3.14**, numpy is the **only** kernel runtime dependency. scipy is used **only** in `controls.py`/harness, never imported by `octonion.py` or `shadow.py`.
- **No runtime import of `aoi_collapse.py` or any AOI-engine code** anywhere in the package. The kernel must build, import, and pass A/B/C with that code entirely absent. (An optional, auto-skipped *test* may cross-check against it — Task 3 — but it is never a dependency.)
- **Octonion product convention:** Cayley–Dickson doubling, Baez *The Octonions* (2002) convention `(a,b)(c,d) = (ac − conj(d)·b, d·a + b·conj(c))`, with conjugation recursing into the first half. This convention is the single source of truth; the basis/Fano table is *derived* from it and validated structurally, never hand-typed.
- **Determinism:** every test and harness routine that uses randomness takes a fixed seed via `numpy.random.default_rng(seed)`. No `Math.random`-style unseeded calls.
- **Kernel is pure:** `octonion.py` and `shadow.py` have no I/O, no global mutable state. Only `harness_report.py` prints.
- **No `hypothesis`** (not installed). Algebra-law tests use seeded numpy RNG loops over thousands of random octonions — same coverage, reproducible, zero new dependencies.
- **Run location:** all commands run from the repo root `C:\Users\jim\octonion-kernel`. `octonion_kernel` is importable as a package from there; no `pip install` / packaging step is required.
- **The control-C test must never assert that the associator wins.** It asserts the harness runs, that magnitude is killed by construction (sanity), and that a verdict is produced. A "no" verdict is a valid scientific outcome, not a build failure. Asserting a win would rebuild the very "castle" this build exists to prevent.

---

## File Structure

- `octonion_kernel/__init__.py` — re-exports `Octonion`, `multiply`, `ShadowResult`, `shadow_decompose`, `identity_residuals`.
- `octonion_kernel/octonion.py` — `Octonion` value type + `multiply` (Cayley–Dickson). The single source of the algebra's structure.
- `octonion_kernel/shadow.py` — `ShadowResult`, `shadow_decompose`, `identity_residuals`.
- `octonion_kernel/controls.py` — `auc`, `bootstrap_sep_ci`, `structured_pair`, `random_pair`, `run_control_c`. Pure compute (returns dicts), no I/O.
- `tests/fixtures/basis_table.json` — pinned 8×8 product table (sign, index), generated once from `multiply` and committed as a regression snapshot.
- `tests/test_octonion.py` — value-type behavior.
- `tests/test_algebra.py` — product correctness, composition laws, Fano structural validation, pinned-table regression, optional legacy cross-check.
- `tests/test_shadow.py` — decomposition structure + identity residuals over ≥10k pairs (harness B).
- `tests/test_controls.py` — control-C harness runs, magnitude killed, verdict produced (harness C).
- `harness_report.py` — runnable A/B/C report (repo root, imports the kernel).
- `README.md` — what it is, how to run tests and the report.

---

### Task 1: Octonion value type + project scaffold

**Files:**
- Create: `octonion_kernel/__init__.py`
- Create: `octonion_kernel/octonion.py`
- Create: `tests/__init__.py` (empty)
- Create: `tests/test_octonion.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces:
  - `Octonion(coeffs)` — frozen value type. `coeffs` is any array-like of 8 reals; stored as a `numpy.ndarray` shape `(8,)` dtype `float64`. Raises `ValueError` if not 8 elements.
  - `.coeffs: np.ndarray` (shape `(8,)`), `.real: float` (=coeffs[0]), `.vec: np.ndarray` (shape `(7,)`, =coeffs[1:]).
  - `a + b`, `a - b`, `-a` (Octonion ± Octonion → Octonion), `a * s` and `s * a` for real scalar `s` → Octonion.
  - `.conjugate() -> Octonion` (negate e1..e7, keep e0), `.norm() -> float` (Euclidean over 8 coeffs), `.approx_eq(other, tol=1e-9) -> bool`.
  - NOTE: octonion × octonion (`a * b` where both are `Octonion`) is wired in Task 2; in this task `__mul__` handles the scalar case only.

- [ ] **Step 1: Write the failing test**

Create `tests/__init__.py` as an empty file. Then create `tests/test_octonion.py`:

```python
import numpy as np
import pytest
from octonion_kernel import Octonion


def test_construction_and_accessors():
    o = Octonion([1, 2, 3, 4, 5, 6, 7, 8])
    assert o.coeffs.shape == (8,)
    assert o.coeffs.dtype == np.float64
    assert o.real == 1.0
    assert np.array_equal(o.vec, np.array([2, 3, 4, 5, 6, 7, 8], dtype=float))


def test_construction_rejects_wrong_length():
    with pytest.raises(ValueError):
        Octonion([1, 2, 3])


def test_is_frozen():
    o = Octonion([1, 0, 0, 0, 0, 0, 0, 0])
    with pytest.raises(Exception):
        o.coeffs = np.zeros(8)  # frozen dataclass forbids reassignment


def test_add_sub_neg():
    a = Octonion([1, 1, 1, 1, 0, 0, 0, 0])
    b = Octonion([0, 0, 0, 0, 2, 2, 2, 2])
    assert (a + b).approx_eq(Octonion([1, 1, 1, 1, 2, 2, 2, 2]))
    assert (a - a).approx_eq(Octonion(np.zeros(8)))
    assert (-a).approx_eq(Octonion([-1, -1, -1, -1, 0, 0, 0, 0]))


def test_scalar_mul_both_sides():
    a = Octonion([1, 2, 3, 4, 5, 6, 7, 8])
    assert (a * 2).approx_eq(Octonion([2, 4, 6, 8, 10, 12, 14, 16]))
    assert (3 * a).approx_eq(Octonion([3, 6, 9, 12, 15, 18, 21, 24]))


def test_conjugate_and_norm():
    a = Octonion([1, 2, 3, 4, 5, 6, 7, 8])
    assert a.conjugate().approx_eq(Octonion([1, -2, -3, -4, -5, -6, -7, -8]))
    assert abs(a.norm() - np.sqrt(204.0)) < 1e-12  # 1+4+9+...+64 = 204


def test_approx_eq_tolerance():
    a = Octonion([1, 0, 0, 0, 0, 0, 0, 0])
    b = Octonion([1 + 1e-12, 0, 0, 0, 0, 0, 0, 0])
    assert a.approx_eq(b)
    assert not a.approx_eq(Octonion([1.1, 0, 0, 0, 0, 0, 0, 0]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_octonion.py -q`
Expected: FAIL — collection error / `ModuleNotFoundError: No module named 'octonion_kernel'`.

- [ ] **Step 3: Write minimal implementation**

Create `octonion_kernel/octonion.py`:

```python
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
```

Create `octonion_kernel/__init__.py`:

```python
from .octonion import Octonion, multiply

__all__ = ["Octonion", "multiply"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_octonion.py -q`
Expected: PASS — 7 passed. (The scalar-only `__mul__` path is all these tests exercise; octonion×octonion is Task 2.)

- [ ] **Step 5: Commit**

```bash
git add octonion_kernel/__init__.py octonion_kernel/octonion.py tests/__init__.py tests/test_octonion.py
git commit -m "feat: Octonion value type (construction, arithmetic, conjugate, norm)"
```

---

### Task 2: The Cayley–Dickson product

**Files:**
- Modify: `octonion_kernel/octonion.py` (replace the `multiply` placeholder; add private `_cd_conj`, `_cd_mul` helpers)
- Create: `tests/test_algebra.py`

**Interfaces:**
- Consumes: `Octonion` and its accessors from Task 1.
- Produces:
  - `multiply(a: Octonion, b: Octonion) -> Octonion` — the octonion product via recursive Cayley–Dickson doubling (Baez convention). Now also reachable as `a * b`.
  - Test helpers (module-local to `test_algebra.py`, reused by Task 3): `_basis(i)` returns the i-th unit octonion; `_rand_oct(rng)` returns a random octonion from a numpy Generator.

- [ ] **Step 1: Write the failing test**

Create `tests/test_algebra.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_algebra.py -q`
Expected: FAIL — `NotImplementedError: multiply is implemented in Task 2` (raised through the `a * b` operator and direct `multiply` calls).

- [ ] **Step 3: Write minimal implementation**

In `octonion_kernel/octonion.py`, replace the `multiply` placeholder function with the helpers and real implementation:

```python
def _cd_conj(x: np.ndarray) -> np.ndarray:
    """Cayley-Dickson conjugate: negate the imaginary half, recurse into the real half."""
    n = x.shape[0]
    if n == 1:
        return x.copy()
    h = n // 2
    return np.concatenate([_cd_conj(x[:h]), -x[h:]])


def _cd_mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Recursive Cayley-Dickson product (Baez convention):
    (a, b)(c, d) = (a*c - conj(d)*b, d*a + b*conj(c))
    Works for lengths 1, 2, 4, 8 (reals -> complex -> quaternions -> octonions).
    """
    n = x.shape[0]
    if n == 1:
        return x * y
    h = n // 2
    a, b = x[:h], x[h:]
    c, d = y[:h], y[h:]
    re = _cd_mul(a, c) - _cd_mul(_cd_conj(d), b)
    im = _cd_mul(d, a) + _cd_mul(b, _cd_conj(c))
    return np.concatenate([re, im])


def multiply(a: "Octonion", b: "Octonion") -> "Octonion":
    """The octonion product (Cayley-Dickson, Baez convention)."""
    return Octonion(_cd_mul(a.coeffs, b.coeffs))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_algebra.py -q`
Expected: PASS — 6 passed.

- [ ] **Step 5: Commit**

```bash
git add octonion_kernel/octonion.py tests/test_algebra.py
git commit -m "feat: Cayley-Dickson octonion product + composition-algebra tests"
```

---

### Task 3: Harness A — composition laws, Fano structure, pinned table

**Files:**
- Create: `tests/fixtures/__init__.py` (empty)
- Create: `tests/fixtures/basis_table.json` (generated, then committed)
- Modify: `tests/test_algebra.py` (append the harness-A tests)

**Interfaces:**
- Consumes: `Octonion`, `multiply`, and the `_basis` / `_rand_oct` helpers from Task 2.
- Produces: no new public API. Adds the harness-A test coverage: non-associativity, a Moufang identity, the Fano-plane structural validation of the derived basis table, a pinned-table regression snapshot, and an optional (auto-skipped) cross-check against the legacy `aoi_collapse` module.

- [ ] **Step 1: Generate and commit the pinned basis table**

The pinned table is a *regression snapshot* derived once from `multiply`; the Fano structural test (Step 2) independently proves that snapshot is a genuine octonion table, so the snapshot is not a circular self-check. Create `tests/fixtures/__init__.py` empty, then generate the fixture:

Run:
```bash
python -c "import json, numpy as np; from octonion_kernel import Octonion, multiply; \
T=[[None]*8 for _ in range(8)]; \
[T[i].__setitem__(j, (lambda p: [ (round(float(p[k])), k) for k in range(8) if abs(p[k])>1e-9 ])(multiply(Octonion(np.eye(8)[i]), Octonion(np.eye(8)[j])).coeffs)) for i in range(8) for j in range(8)]; \
open('tests/fixtures/basis_table.json','w').write(json.dumps(T))"
```

This writes, for each `(i, j)`, the list of `[sign, index]` nonzero entries of `eᵢ·eⱼ` (each basis product is a single signed basis element, so each cell is a one-element list like `[[1, 5]]` or `[[-1, 0]]`).

- [ ] **Step 2: Write the failing test (append to `tests/test_algebra.py`)**

```python
import json
import os
from collections import Counter


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
        import pytest
        pytest.skip("aoi_collapse not importable; legacy cross-check skipped")
    if not hasattr(legacy, "octonion_shadow_decompose"):
        import pytest
        pytest.skip("legacy octonion_shadow_decompose not present")
    # If present, assert agreement on the product norm (basis-convention agnostic):
    rng = np.random.default_rng(99)
    a, b = _rand_oct(rng), _rand_oct(rng)
    ours = (a * b).norm()
    legacy_result = legacy.octonion_shadow_decompose(a.coeffs, b.coeffs)
    # legacy returns a dict-like with a 'product' 8-vector; compare norms only.
    legacy_prod = np.asarray(legacy_result["product"], dtype=float).reshape(-1)
    assert abs(float(np.linalg.norm(legacy_prod)) - ours) < 1e-6
```

- [ ] **Step 3: Run test to verify it fails first, then passes**

These tests reference only Task-2 code plus the fixture; they should already pass once the fixture exists. To honor the red-green cycle, first run them *before* generating the fixture is committed — but since Step 1 already generated it, run the full file:

Run: `python -m pytest tests/test_algebra.py -q`
Expected: PASS — all algebra + harness-A tests pass; `test_legacy_cross_check_if_available` reports as **skipped** (1 skipped) because `aoi_collapse` is not on the path from the repo root. If `test_multiply_matches_pinned_table` fails, the fixture was generated against different code — regenerate it via Step 1 and re-run.

- [ ] **Step 4: Confirm the skip and count**

Run: `python -m pytest tests/test_algebra.py -v`
Expected: every test PASS except `test_legacy_cross_check_if_available` = SKIPPED.

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/__init__.py tests/fixtures/basis_table.json tests/test_algebra.py
git commit -m "test: harness A — non-assoc, Moufang, Fano structure, pinned table, legacy cross-check"
```

---

### Task 4: Jordan-Shadow decomposition + Harness B (identities)

**Files:**
- Create: `octonion_kernel/shadow.py`
- Modify: `octonion_kernel/__init__.py` (export the new names)
- Create: `tests/test_shadow.py`

**Interfaces:**
- Consumes: `Octonion`, `multiply` from Task 2.
- Produces:
  - `ShadowResult` — frozen dataclass with fields `jordan: Octonion`, `commutator: Octonion`, `associator: Octonion`, `product: Octonion`.
  - `shadow_decompose(a: Octonion, b: Octonion) -> ShadowResult` where `jordan = (ab + ba)/2`, `commutator = (ab - ba)/2`, `associator = jordan * commutator`, `product = ab`.
  - `identity_residuals(a: Octonion, b: Octonion) -> dict[str, float]` with keys `"losslessness"`, `"orthogonality"`, `"pythagorean"` (all expected ≈ 0).

- [ ] **Step 1: Write the failing test**

Create `tests/test_shadow.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_shadow.py -q`
Expected: FAIL — `ImportError: cannot import name 'shadow_decompose' from 'octonion_kernel'`.

- [ ] **Step 3: Write minimal implementation**

Create `octonion_kernel/shadow.py`:

```python
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


def identity_residuals(a: Octonion, b: Octonion) -> dict:
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
```

Update `octonion_kernel/__init__.py`:

```python
from .octonion import Octonion, multiply
from .shadow import ShadowResult, shadow_decompose, identity_residuals

__all__ = [
    "Octonion",
    "multiply",
    "ShadowResult",
    "shadow_decompose",
    "identity_residuals",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_shadow.py -q`
Expected: PASS — 3 passed (the 10k-pair sweep runs in a few seconds).

- [ ] **Step 5: Commit**

```bash
git add octonion_kernel/shadow.py octonion_kernel/__init__.py tests/test_shadow.py
git commit -m "feat: Jordan-Shadow decomposition + identity residuals (harness B)"
```

---

### Task 5: Control C (anti-castle), the report script, README, and final verification

**Files:**
- Create: `octonion_kernel/controls.py`
- Create: `tests/test_controls.py`
- Create: `harness_report.py`
- Create: `README.md`

**Interfaces:**
- Consumes: `Octonion` from Task 1; `shadow_decompose` / `ShadowResult` from Task 4.
- Produces:
  - `auc(scores: np.ndarray, labels: np.ndarray) -> float` — directed AUC (P(score|label=1 > score|label=0)) via tie-correct ranks.
  - `bootstrap_sep_ci(scores, labels, n_boot=2000, seed=0) -> tuple[float, float]` — 95% bootstrap CI of the *separation* `max(auc, 1-auc)`.
  - `structured_pair(rng, eps=0.1) -> tuple[Octonion, Octonion]` — unit-normalized pair sharing a fixed direction (correlated).
  - `random_pair(rng) -> tuple[Octonion, Octonion]` — unit-normalized independent pair.
  - `run_control_c(n=2000, seed=0) -> dict` — runs the experiment; returns per-summary `{auc, sep, ci_lo, ci_hi}` for `"prod_norm"`, `"assoc_norm"`, `"assoc_real"`, `"assoc_maxabs"`, plus a `"verdict"` dict `{associator_beats_magnitude: bool, best_summary: str, magnitude_sep: float}`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_controls.py`:

```python
import numpy as np
from octonion_kernel.controls import (
    auc, bootstrap_sep_ci, structured_pair, random_pair, run_control_c,
)


def test_auc_perfect_separation():
    scores = np.array([0.0, 0.1, 0.2, 1.0, 1.1, 1.2])
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert abs(auc(scores, labels) - 1.0) < 1e-12


def test_auc_chance():
    scores = np.array([0.0, 1.0, 0.0, 1.0])
    labels = np.array([0, 0, 1, 1])
    assert abs(auc(scores, labels) - 0.5) < 1e-12


def test_bootstrap_ci_brackets_separation():
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.normal(0, 1, 200), rng.normal(3, 1, 200)])
    labels = np.concatenate([np.zeros(200), np.ones(200)])
    lo, hi = bootstrap_sep_ci(scores, labels, n_boot=500, seed=1)
    assert 0.5 <= lo <= hi <= 1.0
    assert lo > 0.5  # clearly separated data => CI lower bound above chance


def test_pair_generators_are_unit_normalized():
    rng = np.random.default_rng(2)
    for _ in range(50):
        a, b = structured_pair(rng)
        assert abs(a.norm() - 1.0) < 1e-9 and abs(b.norm() - 1.0) < 1e-9
        a, b = random_pair(rng)
        assert abs(a.norm() - 1.0) < 1e-9 and abs(b.norm() - 1.0) < 1e-9


def test_control_c_runs_kills_magnitude_and_produces_verdict():
    out = run_control_c(n=1500, seed=0)
    # a verdict is produced regardless of its value
    assert "verdict" in out
    assert isinstance(out["verdict"]["associator_beats_magnitude"], bool)
    assert out["verdict"]["best_summary"] in ("assoc_norm", "assoc_real", "assoc_maxabs")
    # SANITY: magnitude is killed by unit-normalization, so ||product|| can't separate.
    # This guards the experiment's validity; it does NOT assert the associator wins.
    assert out["prod_norm"]["sep"] < 0.6
    # every reported summary has a well-formed CI
    for key in ("prod_norm", "assoc_norm", "assoc_real", "assoc_maxabs"):
        assert 0.5 <= out[key]["ci_lo"] <= out[key]["ci_hi"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_controls.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'octonion_kernel.controls'`.

- [ ] **Step 3: Write minimal implementation**

Create `octonion_kernel/controls.py`:

```python
"""Control harness C — the anti-castle test.

Question: does the Jordan-Shadow associator carry information BEYOND a trivial
magnitude statistic? We build a 'structured' class (pairs that share a fixed
direction) and a 'random' class (independent pairs), UNIT-NORMALIZE every
octonion so all magnitude information is destroyed (||product|| becomes constant,
its AUC ~ 0.5), then ask whether a fixed, pre-declared set of associator summaries
can still separate the classes. No open search over summaries (that would
cherry-pick toward the associator). A 'no' verdict is a valid result.

Pure compute: returns dicts, no I/O. Uses scipy only for tie-correct ranking.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from .octonion import Octonion
from .shadow import shadow_decompose

# fixed correlated direction shared by the structured class
_V0 = np.zeros(8)
_V0[1] = 1.0


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Directed AUC = P(score | label==1 > score | label==0), tie-corrected."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n1 = int(labels.sum())
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def bootstrap_sep_ci(scores, labels, n_boot: int = 2000, seed: int = 0):
    """95% bootstrap CI of separation = max(auc, 1 - auc)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    rng = np.random.default_rng(seed)
    n = len(scores)
    seps = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        a = auc(scores[idx], labels[idx])
        seps[i] = max(a, 1.0 - a)
    lo, hi = np.percentile(seps, [2.5, 97.5])
    return float(lo), float(hi)


def structured_pair(rng, eps: float = 0.1):
    """Pair (a, b) where b shares a's direction and a fixed direction _V0 (correlated)."""
    a = rng.standard_normal(8)
    ca, cv = rng.standard_normal(), rng.standard_normal()
    b = ca * a + cv * _V0 + eps * rng.standard_normal(8)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return Octonion(a), Octonion(b)


def random_pair(rng):
    """Independent pair, unit-normalized (magnitude-matched to the structured class)."""
    a = rng.standard_normal(8)
    b = rng.standard_normal(8)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return Octonion(a), Octonion(b)


# fixed, pre-declared associator summaries (no open search)
def _summaries(result):
    assoc = result.associator
    return {
        "prod_norm": result.product.norm(),
        "assoc_norm": assoc.norm(),
        "assoc_real": assoc.real,
        "assoc_maxabs": float(np.max(np.abs(assoc.coeffs))),
    }


_ASSOC_KEYS = ("assoc_norm", "assoc_real", "assoc_maxabs")
_ALL_KEYS = ("prod_norm",) + _ASSOC_KEYS


def run_control_c(n: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    cols = {k: [] for k in _ALL_KEYS}
    labels = []
    for label, make in ((1, structured_pair), (0, random_pair)):
        for _ in range(n):
            a, b = make(rng)
            s = _summaries(shadow_decompose(a, b))
            for k in _ALL_KEYS:
                cols[k].append(s[k])
            labels.append(label)
    labels = np.array(labels)

    out = {}
    for k in _ALL_KEYS:
        scores = np.array(cols[k])
        a = auc(scores, labels)
        lo, hi = bootstrap_sep_ci(scores, labels, seed=seed + 1)
        out[k] = {"auc": a, "sep": max(a, 1.0 - a), "ci_lo": lo, "ci_hi": hi}

    best_key = max(_ASSOC_KEYS, key=lambda k: out[k]["sep"])
    magnitude_sep = out["prod_norm"]["sep"]
    # associator carries info beyond magnitude iff its best summary's CI lower
    # bound clears chance AND clears the magnitude baseline's separation.
    beats = bool(out[best_key]["ci_lo"] > 0.5 and out[best_key]["ci_lo"] > magnitude_sep)
    out["verdict"] = {
        "associator_beats_magnitude": beats,
        "best_summary": best_key,
        "magnitude_sep": magnitude_sep,
    }
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_controls.py -q`
Expected: PASS — 5 passed.

- [ ] **Step 5: Write the report script and README**

Create `harness_report.py` (repo root):

```python
"""Runnable A/B/C control report. Run from the repo root: python harness_report.py

A: algebra correctness  (delegates to pytest tests/test_algebra.py)
B: Jordan-Shadow identity residuals over random pairs
C: the anti-castle control — does the associator beat the magnitude baseline?
"""
import numpy as np

from octonion_kernel import Octonion, identity_residuals
from octonion_kernel.controls import run_control_c


def _rand_oct(rng):
    return Octonion(rng.standard_normal(8))


def report_b(n=10_000, seed=2):
    rng = np.random.default_rng(seed)
    mx = {"losslessness": 0.0, "orthogonality": 0.0, "pythagorean": 0.0}
    for _ in range(n):
        res = identity_residuals(_rand_oct(rng), _rand_oct(rng))
        for k in mx:
            mx[k] = max(mx[k], res[k])
    print(f"\n[B] Jordan-Shadow identities over {n} random pairs (max residual):")
    for k, v in mx.items():
        print(f"    {k:<14} {v:.3e}")


def report_c(n=2000, seed=0):
    out = run_control_c(n=n, seed=seed)
    print(f"\n[C] Anti-castle control ({n} structured + {n} random pairs, all unit-normalized):")
    print(f"    {'summary':<14} {'AUC':>7} {'sep':>7} {'95% CI':>20}")
    for k in ("prod_norm", "assoc_norm", "assoc_real", "assoc_maxabs"):
        r = out[k]
        print(f"    {k:<14} {r['auc']:>7.3f} {r['sep']:>7.3f} "
              f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]")
    v = out["verdict"]
    print(f"\n    magnitude baseline separation: {v['magnitude_sep']:.3f} "
          f"(should be ~0.5 — magnitude was killed by normalization)")
    print(f"    best associator summary: {v['best_summary']}")
    verdict = ("YES — associator carries information beyond magnitude"
               if v["associator_beats_magnitude"]
               else "NO — associator adds nothing over magnitude at the kernel level")
    print(f"    VERDICT: {verdict}")


if __name__ == "__main__":
    print("=" * 64)
    print("Octonion kernel control report")
    print("[A] algebra correctness: run `python -m pytest tests/test_algebra.py`")
    report_b()
    report_c()
    print("=" * 64)
```

Create `README.md`:

```markdown
# octonion-kernel

A small, standalone, provably-correct library for 8-dimensional octonion algebra
and the Jordan-Shadow decomposition (DOI 18690444), with an A/B/C control harness.

This is **Phase 1** of a larger build. It ships only the math it can prove, and an
honest measurement (control C) of whether the Jordan-Shadow associator carries
information beyond a trivial magnitude statistic. It makes no claim about what the
decomposition *means* — only what it provably *is*. The 24D/96D embeddings, the
multi-layer dynamics, persistent homology, and domain adapters are later phases.

## Why 8 dimensions?

By Hurwitz's theorem the only normed division algebras are dimensions 1, 2, 4, 8
(reals, complex, quaternions, octonions). 8 is the largest — the next Cayley–Dickson
doubling (16D sedenions) loses the division property. 24D is not an algebra; it is a
packing space (built as 3×8) handled by a later layer.

## Layout

- `octonion_kernel/octonion.py` — `Octonion` value type + `multiply` (Cayley–Dickson, Baez convention).
- `octonion_kernel/shadow.py` — `shadow_decompose`, `identity_residuals`.
- `octonion_kernel/controls.py` — control-C experiment (AUC, bootstrap CI, generators).
- `harness_report.py` — prints the B/C report.

## Run

From the repo root:

```bash
python -m pytest -q          # full A/B/C test suite
python harness_report.py     # printed B/C report
```

Runtime dependencies: numpy (kernel), scipy (harness ranking only). No network, no AOI-engine imports.
```

- [ ] **Step 6: Final verification — full suite + report**

Run: `python -m pytest -q`
Expected: PASS — all tests pass (one skipped: the legacy cross-check). No failures.

Run: `python harness_report.py`
Expected: prints the `[B]` residual table (all residuals ≤ ~1e-8) and the `[C]` control table with a recorded VERDICT (either "YES" or "NO" — both are valid). The `prod_norm` separation prints ~0.5, confirming magnitude was killed.

- [ ] **Step 7: Commit**

```bash
git add octonion_kernel/controls.py tests/test_controls.py harness_report.py README.md
git commit -m "feat: control C (anti-castle), report script, README; Phase 1 complete"
```

---

## Definition of Done (Phase 1)

- `Octonion`, `multiply`, `shadow_decompose`, `identity_residuals` implemented in `octonion_kernel/` with **no AOI-engine imports**.
- Harness A passes: identity, eᵢ²=−1, anticommutativity, norm-multiplicativity, non-associativity, a Moufang identity, Fano-plane structural validation of the derived basis table, and a pinned-table regression snapshot.
- Harness B passes: losslessness ≤ 1e-10, orthogonality and Pythagorean ≤ 1e-8 over ≥10,000 random pairs.
- Harness C runs and records a verdict (YES or NO) on whether the associator beats the magnitude baseline, with the magnitude baseline confirmed near chance (~0.5).
- `python -m pytest -q` is green (one expected skip); `python harness_report.py` produces the full B/C report.

## Deferred to later phases

Dynamics (multi-layer walk + feedback/decay), persistent homology layer (with a real declared distance metric and its own shuffled-input null), the 24D/96D embedding, domain adapters (finance / bio), and the product surface. Each gets its own spec, plan, and per-layer control gate, in the build order from the design spec §1.

## Self-Review notes

- **Spec coverage:** §2 goal → Tasks 1–4; §5 components (`Octonion`, `multiply`, `shadow_decompose`, `identity_residuals`) → Tasks 1, 2, 4; §6.A → Tasks 2–3; §6.B → Task 4; §6.C → Task 5; §7 testing → all tasks (seeded numpy in place of hypothesis, noted below); §8 DoD → final verification. No gaps.
- **Deliberate deviations from the spec:** (1) `hypothesis` is not installed, so algebra-law tests use seeded numpy RNG loops — equal coverage, reproducible, no new dependency. (2) §6.A's "pinned 64-entry table cross-check" is realized as a *derived* snapshot guarded by an independent Fano-plane structural validation, which is stronger than a hand-typed table (it proves the table is genuinely octonionic rather than begging the question of where the reference came from). (3) Control C unit-normalizes every octonion, the strongest form of the spec's "magnitude-matched random class" — it reduces the magnitude baseline AUC to ~0.5 exactly, so any associator separation is provably non-magnitude.
- **Type consistency:** `ShadowResult` fields (`jordan`, `commutator`, `associator`, `product`) are used identically in Task 4 and Task 5. `run_control_c` keys (`prod_norm`, `assoc_norm`, `assoc_real`, `assoc_maxabs`, `verdict`) match between `controls.py`, `test_controls.py`, and `harness_report.py`. `auc`/`bootstrap_sep_ci`/`structured_pair`/`random_pair` signatures match their test call sites.
