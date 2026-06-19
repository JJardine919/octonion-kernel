# Octonion Dynamics Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an autonomous, genuinely-nonlinear octonion *walk* (the Phase-1 kernel iterated with feedback + decay on the unit sphere) plus a control harness that honestly tests whether iterating the walk makes two declared classes of initial states more linearly separable than the raw input and than matched linear, generic-nonlinear, and random-walk baselines.

**Architecture:** A pure engine `octonion_kernel/dynamics.py` (numpy + Phase-1 kernel only) implementing `octonion_walk`. A harness `octonion_kernel/dynamics_controls.py` (scipy/sklearn allowed) with the four baseline step-maps, a shared iteration driver `run_map`, the structured/random initial-state generators, and the experiment `run_dynamics_control` with a paired-bootstrap AUC verdict. `harness_report.py` gains a `[D]` section. All maps share one iteration driver so the only thing that differs between octonion and baselines is the per-step update term.

**Tech Stack:** Python 3.14, numpy (engine), scipy + scikit-learn (`sklearn.linear_model.LogisticRegression`, `sklearn.metrics.roc_auc_score`) in the harness only, pytest.

## Global Constraints

- **Python 3.14.** `octonion_kernel/dynamics.py` imports only numpy and the Phase-1 kernel (`.octonion`, `.shadow`) — **no scipy, no sklearn, no `aoi_collapse`**. scipy/sklearn are allowed **only** in `dynamics_controls.py` and `harness_report.py` (same boundary Phase 1 set for scipy).
- **State stays on the unit sphere:** every walk step renormalizes, so magnitude can never masquerade as signal.
- **The map:** `x_{t+1} = renorm( lam*x_t + (1-lam)*associator(x_t, g) )`, `associator(x,g) = shadow_decompose(x,g).associator` (quadratic in x → genuinely nonlinear). Defaults `lam=0.5`, `steps=32`.
- **Pinned, never tuned:** the generator `g = DEFAULT_GENERATOR` (a fixed unit octonion) and the structured-class subspace dimension (`_STRUCTURED_DIM = 3`) are declared constants, not searched over.
- **Determinism:** all randomness via seeded `numpy.random.default_rng`; no `hypothesis`.
- **Anti-castle rule (critical):** the dynamics control test asserts only that the harness runs, returns a verdict, keeps states unit-norm, the chosen best baseline is one of the four declared baselines, and every AUC is in `[0,1]`. It must **NOT** assert the octonion walk wins or loses. A NO is a valid outcome.
- **Run location:** all commands from repo root `C:\Users\jim\octonion-kernel`; `python -m pytest ...`. No packaging step.
- The `slow` pytest marker is already registered in `pytest.ini` (from Phase 1) — reuse it for slow control tests.

## File Structure

- `octonion_kernel/dynamics.py` — pure engine: `octonion_walk`, `associator_step`, `renorm`, `DEFAULT_GENERATOR`.
- `octonion_kernel/dynamics_controls.py` — harness: initial-state generators, four baseline step-maps + scale-matching, `run_map`, `octonion_step`, the experiment `run_dynamics_control`, paired-bootstrap helper. May import scipy/sklearn.
- `tests/test_dynamics.py` — engine checks.
- `tests/test_dynamics_controls.py` — control building-block checks + the agnostic experiment check.
- `harness_report.py` — add `report_d` and call it in `__main__`.
- `README.md` — add a short "Phase 2 — dynamics" note (no hardcoded verdict).

---

### Task 1: The octonion walk engine

**Files:**
- Create: `octonion_kernel/dynamics.py`
- Create: `tests/test_dynamics.py`

**Interfaces:**
- Consumes: `Octonion` (8 float coeffs, `.coeffs`, `.norm()`, `.approx_eq`), `shadow_decompose(a,b).associator` from the Phase-1 kernel.
- Produces:
  - `DEFAULT_GENERATOR: Octonion` — a fixed unit octonion.
  - `renorm(x: Octonion) -> Octonion` — `x / ‖x‖`.
  - `associator_step(x: Octonion, g: Octonion) -> Octonion` — `shadow_decompose(x,g).associator`.
  - `octonion_walk(x0: Octonion, g: Octonion = DEFAULT_GENERATOR, lam: float = 0.5, steps: int = 32) -> list[Octonion]` — trajectory `[x0_normalized, x1, …, x_T]`; halts early (returns trajectory so far) if a step's norm `< 1e-12`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dynamics.py`:

```python
import numpy as np
from octonion_kernel import Octonion
from octonion_kernel.dynamics import (
    octonion_walk, associator_step, renorm, DEFAULT_GENERATOR,
)


def _rand_oct(rng):
    return Octonion(rng.standard_normal(8))


def test_default_generator_is_unit():
    assert abs(DEFAULT_GENERATOR.norm() - 1.0) < 1e-12


def test_renorm_gives_unit_norm():
    rng = np.random.default_rng(0)
    x = _rand_oct(rng)
    assert abs(renorm(x).norm() - 1.0) < 1e-12


def test_walk_preserves_unit_norm_every_step():
    rng = np.random.default_rng(1)
    for _ in range(50):
        traj = octonion_walk(_rand_oct(rng), steps=16)
        for x in traj:
            assert abs(x.norm() - 1.0) < 1e-10


def test_walk_is_deterministic():
    rng = np.random.default_rng(2)
    x0 = _rand_oct(rng)
    t1 = octonion_walk(x0, lam=0.5, steps=20)
    t2 = octonion_walk(x0, lam=0.5, steps=20)
    assert len(t1) == len(t2)
    for a, b in zip(t1, t2):
        assert a.approx_eq(b, tol=1e-12)


def test_walk_length_is_steps_plus_one():
    rng = np.random.default_rng(3)
    traj = octonion_walk(_rand_oct(rng), steps=32)
    # x0 plus 32 steps, unless an underflow halt occurred (this seed does not halt)
    assert len(traj) == 33


def test_associator_step_is_degree_2_homogeneous():
    # associator(alpha*x, g) == alpha**2 * associator(x, g) -> the map is nonlinear
    rng = np.random.default_rng(4)
    g = _rand_oct(rng)
    for _ in range(200):
        x = _rand_oct(rng)
        alpha = float(rng.standard_normal())
        lhs = associator_step(Octonion(alpha * x.coeffs), g)
        rhs = Octonion(alpha**2 * associator_step(x, g).coeffs)
        assert lhs.approx_eq(rhs, tol=1e-9)


def test_walk_is_finite():
    rng = np.random.default_rng(5)
    for _ in range(50):
        traj = octonion_walk(_rand_oct(rng), steps=32)
        for x in traj:
            assert np.all(np.isfinite(x.coeffs))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dynamics.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'octonion_kernel.dynamics'`.

- [ ] **Step 3: Write minimal implementation**

Create `octonion_kernel/dynamics.py`:

```python
"""Autonomous octonion dynamics — an iterated, genuinely-nonlinear walk.

Pure: numpy + the Phase-1 kernel only. No I/O, no scipy/sklearn, no aoi_collapse.

The map (unit-sphere, feedback + decay):
    x_{t+1} = renorm( lam * x_t + (1 - lam) * associator(x_t, g) )
where associator(x, g) = jordan(x, g) * commutator(x, g) from shadow_decompose.
The associator is quadratic in x, so the map is genuinely nonlinear: a fixed-octonion
multiply alone is linear on R^8 and a linear baseline would reproduce it.
"""
from __future__ import annotations

import numpy as np

from .octonion import Octonion
from .shadow import shadow_decompose


def associator_step(x: Octonion, g: Octonion) -> Octonion:
    """The nonlinear term jordan(x,g) * commutator(x,g) (quadratic in x)."""
    return shadow_decompose(x, g).associator


def renorm(x: Octonion) -> Octonion:
    """Project onto the unit sphere. Caller ensures ||x|| is not ~0."""
    return Octonion(x.coeffs / x.norm())


# A fixed, declared generator octonion (unit norm). Pinned, never tuned to a result.
DEFAULT_GENERATOR: Octonion = Octonion(
    np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) / np.sqrt(7.0)
)


def octonion_walk(x0: Octonion, g: Octonion = DEFAULT_GENERATOR,
                  lam: float = 0.5, steps: int = 32) -> list[Octonion]:
    """Iterate the autonomous octonion map from x0, returning [x0, x1, ..., x_T].

    State is renormalized to the unit sphere each step. If a step underflows
    (||y|| < 1e-12), the walk halts and returns the trajectory so far.
    """
    traj = [renorm(x0)]
    for _ in range(steps):
        x = traj[-1]
        y = Octonion(lam * x.coeffs + (1.0 - lam) * associator_step(x, g).coeffs)
        if y.norm() < 1e-12:
            break
        traj.append(renorm(y))
    return traj
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dynamics.py -q`
Expected: PASS — 7 passed.

- [ ] **Step 5: Commit**

```bash
git add octonion_kernel/dynamics.py tests/test_dynamics.py
git commit -m "feat: autonomous octonion walk engine (associator-driven, unit-sphere)"
```

---

### Task 2: Control building blocks — generators, baseline maps, iteration driver

**Files:**
- Create: `octonion_kernel/dynamics_controls.py`
- Create: `tests/test_dynamics_controls.py`

**Interfaces:**
- Consumes: `Octonion`; from `dynamics.py`: `associator_step`, `renorm`, `DEFAULT_GENERATOR`.
- Produces:
  - `make_random(rng) -> Octonion` (uniform on S⁷); `make_structured(rng) -> Octonion` (unit, nonzero only on first `_STRUCTURED_DIM=3` axes).
  - `_mean_assoc_norm(g, seed=12345, n=2000) -> float` — `E[‖associator(x,g)‖]` over random unit x.
  - `make_linear_map(g, seed=0) -> step_fn` where `step_fn(x: Octonion) -> Octonion` applies a scale-matched random linear map (`E[‖A·x‖] = E[‖associator(x,g)‖]`).
  - `make_generic_nonlinear_map(g, seed=0) -> step_fn` — scale-matched random quadratic `Q(x)_i = xᵀT_ix`.
  - `make_random_walk_step(g, seed=0) -> step_fn` — fresh scale-matched noise each call.
  - `octonion_step(g) -> step_fn` — the associator step term as a `step_fn`.
  - `run_map(x0: Octonion, step_fn, lam=0.5, steps=32) -> Octonion` — iterates `renorm(lam*x + (1-lam)*step_fn(x))`, returns the final state; halts on norm `< 1e-12`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dynamics_controls.py`:

```python
import numpy as np
import pytest
from octonion_kernel import Octonion
from octonion_kernel.dynamics import DEFAULT_GENERATOR
from octonion_kernel.dynamics_controls import (
    make_random, make_structured, _STRUCTURED_DIM, _mean_assoc_norm,
    make_linear_map, make_generic_nonlinear_map, make_random_walk_step,
    octonion_step, run_map,
)


def test_generators_unit_norm_and_subspace():
    rng = np.random.default_rng(0)
    for _ in range(50):
        r = make_random(rng)
        s = make_structured(rng)
        assert abs(r.norm() - 1.0) < 1e-12
        assert abs(s.norm() - 1.0) < 1e-12
        assert np.allclose(s.coeffs[_STRUCTURED_DIM:], 0.0)


def test_linear_map_is_linear_and_scale_matched():
    g = DEFAULT_GENERATOR
    step = make_linear_map(g, seed=0)
    rng = np.random.default_rng(5)
    x = make_random(rng)
    # linearity: step(2x) == 2*step(x)
    assert Octonion(2.0 * step(x).coeffs).approx_eq(step(Octonion(2.0 * x.coeffs)), tol=1e-9)
    # scale matched within 10%
    tgt = _mean_assoc_norm(g)
    s_rng = np.random.default_rng(7)
    tot = 0.0
    m = 1000
    for _ in range(m):
        tot += step(make_random(s_rng)).norm()
    assert abs(tot / m - tgt) / tgt < 0.1


def test_generic_nonlinear_is_quadratic_and_scale_matched():
    g = DEFAULT_GENERATOR
    step = make_generic_nonlinear_map(g, seed=0)
    rng = np.random.default_rng(6)
    x = make_random(rng)
    # degree-2 homogeneity: step(2x) == 4*step(x)
    assert Octonion(4.0 * step(x).coeffs).approx_eq(step(Octonion(2.0 * x.coeffs)), tol=1e-7)
    tgt = _mean_assoc_norm(g)
    s_rng = np.random.default_rng(8)
    tot = 0.0
    m = 1000
    for _ in range(m):
        tot += step(make_random(s_rng)).norm()
    assert abs(tot / m - tgt) / tgt < 0.1


def test_run_map_preserves_unit_norm_and_is_deterministic():
    g = DEFAULT_GENERATOR
    step = octonion_step(g)
    rng = np.random.default_rng(9)
    x0 = make_random(rng)
    xf1 = run_map(x0, step, steps=16)
    xf2 = run_map(x0, step, steps=16)
    assert abs(xf1.norm() - 1.0) < 1e-10
    assert xf1.approx_eq(xf2, tol=1e-12)


def test_random_walk_step_scale_matched():
    g = DEFAULT_GENERATOR
    step = make_random_walk_step(g, seed=0)
    tgt = _mean_assoc_norm(g)
    rng = np.random.default_rng(11)
    x = make_random(rng)
    # each emitted step term has norm ~ tgt (independent of x)
    norms = [step(x).norm() for _ in range(100)]
    assert abs(np.mean(norms) - tgt) / tgt < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dynamics_controls.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'octonion_kernel.dynamics_controls'`.

- [ ] **Step 3: Write minimal implementation**

Create `octonion_kernel/dynamics_controls.py`:

```python
"""Control harness for the octonion dynamics layer (Phase 2).

Harness code: numpy + scipy + sklearn allowed (never imported by the pure
dynamics engine or the kernel). Pure compute: returns dicts, no I/O.

Builds matched baselines (linear, generic-nonlinear, random walk) and asks
whether iterating the octonion walk makes structured-vs-random initial states
more linearly separable than the raw input and than every baseline.
"""
from __future__ import annotations

import numpy as np

from .octonion import Octonion
from .dynamics import associator_step, renorm, DEFAULT_GENERATOR

# declared structured-class subspace dimension (a "great subsphere")
_STRUCTURED_DIM = 3


def make_random(rng) -> Octonion:
    """Uniform on S^7."""
    v = rng.standard_normal(8)
    return Octonion(v / np.linalg.norm(v))


def make_structured(rng) -> Octonion:
    """Unit octonion supported only on the first _STRUCTURED_DIM basis axes."""
    v = np.zeros(8)
    v[:_STRUCTURED_DIM] = rng.standard_normal(_STRUCTURED_DIM)
    return Octonion(v / np.linalg.norm(v))


def _mean_assoc_norm(g: Octonion, seed: int = 12345, n: int = 2000) -> float:
    """E[||associator(x, g)||] over random unit octonions x (for scale matching)."""
    rng = np.random.default_rng(seed)
    tot = 0.0
    for _ in range(n):
        tot += associator_step(make_random(rng), g).norm()
    return tot / n


def _mean_step_norm(step_fn, seed: int = 999, n: int = 2000) -> float:
    """E[||step_fn(x)||] over random unit octonions x."""
    rng = np.random.default_rng(seed)
    tot = 0.0
    for _ in range(n):
        tot += step_fn(make_random(rng)).norm()
    return tot / n


def make_linear_map(g: Octonion, seed: int = 0):
    """Scale-matched random linear step term x -> A x (A rescaled so
    E[||A x||] == E[||associator(x, g)||])."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((8, 8))

    def raw_step(x: Octonion) -> Octonion:
        return Octonion(A @ x.coeffs)

    scale = _mean_assoc_norm(g) / _mean_step_norm(raw_step)
    A = A * scale

    def step(x: Octonion) -> Octonion:
        return Octonion(A @ x.coeffs)

    return step


def make_generic_nonlinear_map(g: Octonion, seed: int = 0):
    """Scale-matched generic random quadratic step term Q(x)_i = x^T T_i x
    (rescaled so E[||Q(x)||] == E[||associator(x, g)||])."""
    rng = np.random.default_rng(seed)
    T = rng.standard_normal((8, 8, 8))  # T[i] is the matrix for output coord i

    def raw_step(x: Octonion) -> Octonion:
        xc = x.coeffs
        return Octonion(np.einsum("ijk,j,k->i", T, xc, xc))

    scale = _mean_assoc_norm(g) / _mean_step_norm(raw_step)

    def step(x: Octonion) -> Octonion:
        xc = x.coeffs
        return Octonion(scale * np.einsum("ijk,j,k->i", T, xc, xc))

    return step


def make_random_walk_step(g: Octonion, seed: int = 0):
    """Fresh scale-matched noise each call (pure diffusion baseline). The emitted
    step term has norm == E[||associator(x, g)||] and ignores x."""
    target = _mean_assoc_norm(g)
    rng = np.random.default_rng(seed)

    def step(x: Octonion) -> Octonion:
        eta = rng.standard_normal(8)
        return Octonion(eta / np.linalg.norm(eta) * target)

    return step


def octonion_step(g: Octonion):
    """The octonion associator step term as a step_fn (uniform interface)."""
    def step(x: Octonion) -> Octonion:
        return associator_step(x, g)
    return step


def run_map(x0: Octonion, step_fn, lam: float = 0.5, steps: int = 32) -> Octonion:
    """Iterate renorm(lam*x + (1-lam)*step_fn(x)) from x0; return the final state.
    Halts (returns current state) if a step underflows (||y|| < 1e-12)."""
    x = renorm(x0)
    for _ in range(steps):
        y = Octonion(lam * x.coeffs + (1.0 - lam) * step_fn(x).coeffs)
        if y.norm() < 1e-12:
            break
        x = renorm(y)
    return x
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dynamics_controls.py -q`
Expected: PASS — 5 passed.

- [ ] **Step 5: Commit**

```bash
git add octonion_kernel/dynamics_controls.py tests/test_dynamics_controls.py
git commit -m "feat: dynamics control building blocks (generators, scale-matched baselines, run_map)"
```

---

### Task 3: The dynamics experiment + paired-bootstrap verdict

**Files:**
- Modify: `octonion_kernel/dynamics_controls.py` (add the experiment + bootstrap helper)
- Modify: `tests/test_dynamics_controls.py` (append the agnostic experiment test)

**Interfaces:**
- Consumes: everything from Task 2; `sklearn.linear_model.LogisticRegression`, `sklearn.metrics.roc_auc_score`.
- Produces:
  - `_bootstrap_auc_diff_ci(scores_a, scores_b, y, n_boot=2000, seed=0) -> tuple[float,float]` — 95% CI of `AUC(scores_a) − AUC(scores_b)` on shared (paired) test-set resamples.
  - `run_dynamics_control(n=400, lam=0.5, steps=32, seed=0) -> dict` — returns `{"aucs": {map: float}, "verdict": {...}}` where `aucs` has keys `raw, linear, generic_nonlinear, random_walk, octonion` and `verdict` has `octonion_adds_structure: bool`, `best_baseline: str`, `octonion_auc`, `best_baseline_auc`, `auc_difference_ci: [lo,hi]`.

- [ ] **Step 1: Write the failing test (append to `tests/test_dynamics_controls.py`)**

```python
from octonion_kernel.dynamics_controls import run_dynamics_control


@pytest.mark.slow
def test_dynamics_control_runs_and_is_agnostic():
    out = run_dynamics_control(n=300, steps=16, seed=0)
    v = out["verdict"]
    # a verdict is produced regardless of its value (do NOT assert YES/NO)
    assert isinstance(v["octonion_adds_structure"], bool)
    assert v["best_baseline"] in ("raw", "linear", "generic_nonlinear", "random_walk")
    for k in ("raw", "linear", "generic_nonlinear", "random_walk", "octonion"):
        assert 0.0 <= out["aucs"][k] <= 1.0
    # the difference CI is well-formed
    lo, hi = v["auc_difference_ci"]
    assert lo <= hi
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dynamics_controls.py::test_dynamics_control_runs_and_is_agnostic -q`
Expected: FAIL — `ImportError: cannot import name 'run_dynamics_control'`.

- [ ] **Step 3: Write minimal implementation (append to `octonion_kernel/dynamics_controls.py`)**

Add the sklearn import at the TOP of the file with the other imports:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
```

Append at the end of the file:

```python
def _final_features(step_or_none, x0_list, lam, steps):
    """8-vector final state for each x0 under a step map (raw = no walk)."""
    feats = []
    for x0 in x0_list:
        xf = renorm(x0) if step_or_none is None else run_map(x0, step_or_none, lam, steps)
        feats.append(xf.coeffs)
    return np.asarray(feats)


def _auc_on_split(X, y, train_idx, test_idx):
    """Fit logistic regression on the train split, return (test AUC, test scores)."""
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X[train_idx], y[train_idx])
    scores = clf.decision_function(X[test_idx])
    return float(roc_auc_score(y[test_idx], scores)), scores


def _bootstrap_auc_diff_ci(scores_a, scores_b, y, n_boot: int = 2000, seed: int = 0):
    """95% CI of AUC(scores_a) - AUC(scores_b) on shared (paired) test-set resamples."""
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    y = np.asarray(y, dtype=int)
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ys = y[idx]
        if ys.min() == ys.max():
            continue  # degenerate resample (single class) — skip
        diffs.append(roc_auc_score(ys, scores_a[idx]) - roc_auc_score(ys, scores_b[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def run_dynamics_control(n: int = 400, lam: float = 0.5, steps: int = 32, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    g = DEFAULT_GENERATOR

    x0s, labels = [], []
    for _ in range(n):
        x0s.append(make_structured(rng))
        labels.append(1)
    for _ in range(n):
        x0s.append(make_random(rng))
        labels.append(0)
    labels = np.asarray(labels)

    maps = {
        "raw": None,
        "linear": make_linear_map(g, seed=seed + 1),
        "generic_nonlinear": make_generic_nonlinear_map(g, seed=seed + 2),
        "random_walk": make_random_walk_step(g, seed=seed + 3),
        "octonion": octonion_step(g),
    }
    feats = {k: _final_features(m, x0s, lam, steps) for k, m in maps.items()}

    # fixed seeded 60/40 train/test split
    idx = np.arange(2 * n)
    np.random.default_rng(seed + 100).shuffle(idx)
    cut = int(0.6 * len(idx))
    train_idx, test_idx = idx[:cut], idx[cut:]
    y_test = labels[test_idx]

    aucs, scores = {}, {}
    for k in maps:
        aucs[k], scores[k] = _auc_on_split(feats[k], labels, train_idx, test_idx)

    baseline_keys = ("raw", "linear", "generic_nonlinear", "random_walk")
    best_baseline = max(baseline_keys, key=lambda k: aucs[k])
    diff_lo, diff_hi = _bootstrap_auc_diff_ci(
        scores["octonion"], scores[best_baseline], y_test, seed=seed + 200)
    adds = bool(aucs["octonion"] > aucs["raw"] and diff_lo > 0.0)

    return {
        "aucs": aucs,
        "verdict": {
            "octonion_adds_structure": adds,
            "best_baseline": best_baseline,
            "octonion_auc": aucs["octonion"],
            "best_baseline_auc": aucs[best_baseline],
            "auc_difference_ci": [diff_lo, diff_hi],
        },
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dynamics_controls.py -q`
Expected: PASS — 6 passed (the experiment test runs in a few seconds; it is marked `slow`).

- [ ] **Step 5: Commit**

```bash
git add octonion_kernel/dynamics_controls.py tests/test_dynamics_controls.py
git commit -m "feat: dynamics experiment with paired-bootstrap AUC verdict (anti-castle, agnostic test)"
```

---

### Task 4: Report section, README note, and final verification

**Files:**
- Modify: `harness_report.py` (add `report_d`, call it in `__main__`)
- Modify: `README.md` (add a short Phase-2 note)

**Interfaces:**
- Consumes: `run_dynamics_control` from `dynamics_controls.py`.
- Produces: a `[D]` console section; no new programmatic API.

- [ ] **Step 1: Add the report section**

In `harness_report.py`, add this import near the other `octonion_kernel` imports at the top:

```python
from octonion_kernel.dynamics_controls import run_dynamics_control
```

Add this function (after `report_c`):

```python
def report_d(n=400, steps=32, seed=0):
    out = run_dynamics_control(n=n, steps=steps, seed=seed)
    print(f"\n[D] Dynamics control ({n} structured + {n} random initial states, "
          f"{steps}-step unit-sphere walks):")
    print(f"    {'map':<18} {'test AUC':>9}")
    for k in ("raw", "linear", "generic_nonlinear", "random_walk", "octonion"):
        print(f"    {k:<18} {out['aucs'][k]:>9.3f}")
    v = out["verdict"]
    print(f"\n    best baseline:  {v['best_baseline']} (AUC {v['best_baseline_auc']:.3f})")
    print(f"    octonion AUC:   {v['octonion_auc']:.3f}")
    print(f"    AUC(octonion) - AUC(best baseline), 95% CI: "
          f"[{v['auc_difference_ci'][0]:.3f}, {v['auc_difference_ci'][1]:.3f}]")
    if v["octonion_adds_structure"]:
        print("    VERDICT: YES - iterating the octonion walk separates the classes better")
        print("             than the raw input AND every declared baseline (incl. a matched")
        print("             generic quadratic map).")
    else:
        print("    VERDICT: NO - the octonion walk does NOT beat the best baseline. Iteration")
        print("             adds no separability beyond a matched linear / generic-nonlinear /")
        print("             random map; the octonion structure is not doing the work here.")
```

In the `if __name__ == "__main__":` block, add `report_d()` after the `report_c()` call (before the final `print("=" * 64)`).

- [ ] **Step 2: Verify the report runs and produces a [D] verdict**

Run: `python harness_report.py`
Expected: prints `[A]`, `[B]`, `[C]`, and a new `[D]` section with the 5-row AUC table and a recorded VERDICT (YES or NO — whichever it is). Note the verdict value for Step 4.

- [ ] **Step 3: Run the full suite**

Run: `python -m pytest -q`
Expected: all pass plus the pre-existing legacy skip (so `<N> passed, 1 skipped`). No failures, output pristine.

Also confirm the engine stays dependency-clean:

Run: `python -c "import ast; t=ast.parse(open('octonion_kernel/dynamics.py').read()); imps=[a.name for n in ast.walk(t) if isinstance(n,ast.Import) for a in n.names]+[n.module or '' for n in ast.walk(t) if isinstance(n,ast.ImportFrom)]; bad=[m for m in imps if any(w in m for w in ('scipy','sklearn','aoi'))]; print('BAD',bad) if bad else print('clean')"`
Expected: `clean`

- [ ] **Step 4: Add the README note**

In `README.md`, add this subsection at the end (fill the verdict line with the actual verdict observed in Step 2 — YES or NO — and the octonion vs best-baseline AUCs printed by the report):

```markdown
## Phase 2 — dynamics

`octonion_kernel/dynamics.py` iterates the kernel into an autonomous, unit-sphere walk
`x_{t+1} = renorm(λ·x_t + (1−λ)·associator(x_t, g))` (genuinely nonlinear: the associator is
quadratic in `x`). Its control (`dynamics_controls.py`, run via the `[D]` section of
`harness_report.py`) asks whether iterating the walk makes structured-vs-random initial states
more linearly separable than the raw input and than matched **linear, generic-nonlinear, and
random-walk** baselines — the generic-nonlinear map being the decisive bar (the dynamics analog
of `|a·b|`). The octonion walk counts only if it beats the best baseline with a paired-bootstrap
CI excluding zero.

**Phase-2 result:** <VERDICT — copy the [D] verdict from `python harness_report.py`, e.g. "NO —
the octonion walk does not beat the best baseline (octonion AUC X.XXX vs <baseline> X.XXX)">. A NO
is a valid, expected outcome by design.
```

- [ ] **Step 5: Commit**

```bash
git add harness_report.py README.md
git commit -m "feat: [D] dynamics control report section + README Phase-2 note"
```

---

## Definition of Done (Phase 2)

- `dynamics.py` implements `octonion_walk` + helpers with **no scipy/sklearn/aoi imports** (verified by the Step-3 import check); engine tests pass (unit-norm preserved, determinism, degree-2 homogeneity / nonlinearity, finiteness).
- `dynamics_controls.py` implements the generators, four scale-matched baselines, `run_map`, the experiment, and the paired-bootstrap verdict; the control test passes and is **agnostic** to the verdict value.
- `harness_report.py` prints the `[D]` section with the AUC table and a recorded verdict (YES or NO).
- Full `python -m pytest -q` green (plus the pre-existing legacy skip); `python harness_report.py` produces A/B/C/D.

## Self-Review notes

- **Spec coverage:** §4 engine → Task 1; §5.1 generators → Task 2; §5.2 separability metric (`LogisticRegression` test-AUC) → Task 3; §5.3 baselines (raw/linear/generic-nonlinear/random) → Task 2 (maps) + Task 3 (wired into the experiment); §5.4 verdict rule (beat best baseline, paired-bootstrap CI > 0, and beat raw) → Task 3; §5.5 anti-castle agnostic test → Task 3; §6 components → Tasks 1–3; §7 testing → all tasks; §8 DoD → Task 4. No gaps.
- **Scale-matching realized exactly as the spec's tightened wording:** `_mean_step_norm` / `_mean_assoc_norm` estimate `E[‖·‖]` on seeded samples and rescale `A` and `Q` so `E[‖A·x‖] = E[‖Q(x)‖] = E[‖associator(x,g)‖]` (Task 2).
- **Type consistency:** `step_fn` everywhere is `(Octonion) -> Octonion`; `run_map(x0, step_fn, lam, steps)` signature matches all call sites; `run_dynamics_control` keys (`aucs`, `verdict.octonion_adds_structure`, `best_baseline`, `octonion_auc`, `best_baseline_auc`, `auc_difference_ci`) match between `dynamics_controls.py`, the Task-3 test, and `report_d` in Task 4. `DEFAULT_GENERATOR`/`associator_step`/`renorm` names match between `dynamics.py` and `dynamics_controls.py`.
- **Anti-castle:** the only verdict-bearing test (`test_dynamics_control_runs_and_is_agnostic`) asserts structure/ranges/CI well-formedness, never the YES/NO value — matching the Global Constraint.
