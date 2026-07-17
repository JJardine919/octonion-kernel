# Octonion Optimizer Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Phase 4 of the octonion-kernel project — a simulated-annealing solver over a
random SK-model (QUBO-equivalent) instance with four interchangeable move-proposal strategies
(random, greedy local-field, generic-nonlinear, shadow-guided), plus a paired-bootstrap control
harness that honestly tests whether the Jordan-Shadow-guided strategy reaches lower energy than
the best of the other three.

**Architecture:** Two new pure-Python modules mirroring the existing Phase 1-3 layout exactly:
`octonion_kernel/optimize.py` (the engine — instance generation, energy, the four proposal
strategies, the annealing loop) and `octonion_kernel/optimize_controls.py` (the harness — paired
runs across many instances, a power check, and the verdict). `harness_report.py` gains a `[F]`
printed section. Everything reuses `octonion_kernel.octonion.Octonion` and
`octonion_kernel.shadow.shadow_decompose` unchanged.

**Tech Stack:** Python, numpy only (no scipy/sklearn needed for this phase — energy differences
use a simple mean-based paired bootstrap, not AUC). pytest for tests.

## Global Constraints

- `optimize.py` is a **pure** module: numpy + `octonion_kernel.octonion` + `octonion_kernel.shadow`
  only. No I/O, no scipy, no `aoi_collapse`. (Spec §3)
- `optimize_controls.py` is harness code: pure compute (returns dicts), no I/O. (Spec §3)
- Default scale: `n = 64` spins (8 chunks of `CHUNK_SIZE = 8`), `steps = 5000`,
  `n_instances = 500`. (Spec §4, §5.1)
- SK coupling: `J_ij ~ N(0, 1/n)` for `i < j`, symmetric, zero diagonal. (Spec §4)
- All four proposal strategies share the identical solver loop (same instance, same initial
  state, same cooling schedule, same Metropolis acceptance) — only `propose_fn` differs. (Spec §4)
- The `generic_nonlinear` and `shadow` arms must receive exactly the same information (each
  chunk's 8 spins + 8 local fields) — the only difference is octonion algebra vs. a fixed
  elementwise combination. (Spec §4, arm 3 vs. 4)
- Verdict requires a **power check** (greedy and generic_nonlinear must each reliably beat random)
  before a shadow NO is trusted; otherwise the result is `inconclusive`, not NO. (Spec §5.3)
- Control tests must be agnostic to the verdict value (never assert shadow wins or loses). (Spec §5.5)
- Tests use reduced `n_instances`/`steps` for speed; full-resolution run lives in
  `harness_report.py`. (Spec §7)

---

## File Structure

- `octonion_kernel/optimize.py` — **create**. SK instance, energy, local fields, the four
  proposal functions, `anneal()`.
- `tests/test_optimize.py` — **create**. Engine correctness tests.
- `octonion_kernel/optimize_controls.py` — **create**. Paired harness, power check, verdict.
- `tests/test_optimize_controls.py` — **create**. Control-test (agnostic to verdict).
- `harness_report.py` — **modify**. Add `report_f()` and wire it into `__main__` and the module
  docstring.

---

### Task 1: SK instance, energy, and local fields

**Files:**
- Create: `octonion_kernel/optimize.py`
- Test: `tests/test_optimize.py`

**Interfaces:**
- Produces: `make_sk_instance(n: int = 64, seed: int = 0) -> np.ndarray` (symmetric, zero-diagonal
  `n x n` matrix), `energy(state: np.ndarray, J: np.ndarray) -> float`,
  `local_fields(state: np.ndarray, J: np.ndarray) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_optimize.py
import numpy as np
import pytest

from octonion_kernel.optimize import make_sk_instance, energy, local_fields


def _naive_energy(state, J):
    n = len(state)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            total -= J[i, j] * state[i] * state[j]
    return total


def test_make_sk_instance_symmetric_zero_diagonal():
    J = make_sk_instance(n=16, seed=1)
    assert J.shape == (16, 16)
    assert np.allclose(J, J.T)
    assert np.allclose(np.diag(J), 0.0)


def test_make_sk_instance_deterministic():
    J1 = make_sk_instance(n=16, seed=1)
    J2 = make_sk_instance(n=16, seed=1)
    assert np.array_equal(J1, J2)


def test_energy_matches_naive_brute_force():
    rng = np.random.default_rng(2)
    J = make_sk_instance(n=8, seed=2)
    for _ in range(20):
        state = rng.choice([-1.0, 1.0], size=8)
        assert energy(state, J) == pytest.approx(_naive_energy(state, J), abs=1e-9)


def test_energy_matches_all_256_configs_n8():
    J = make_sk_instance(n=8, seed=3)
    for bits in range(256):
        state = np.array([1.0 if (bits >> k) & 1 else -1.0 for k in range(8)])
        assert energy(state, J) == pytest.approx(_naive_energy(state, J), abs=1e-9)


def test_local_field_matches_energy_delta():
    rng = np.random.default_rng(4)
    J = make_sk_instance(n=8, seed=4)
    state = rng.choice([-1.0, 1.0], size=8)
    h = local_fields(state, J)
    for i in range(8):
        flipped = state.copy()
        flipped[i] = -flipped[i]
        expected_delta = energy(flipped, J) - energy(state, J)
        actual_delta = 2.0 * state[i] * h[i]
        assert actual_delta == pytest.approx(expected_delta, abs=1e-9)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_optimize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'octonion_kernel.optimize'`

- [ ] **Step 3: Write the implementation**

```python
# octonion_kernel/optimize.py
"""SK-model instance, energy, and simulated-annealing move-proposal strategies.

Pure module: numpy + the Phase-1 kernel (octonion, shadow) only. No I/O.
"""
from __future__ import annotations

import numpy as np

from .octonion import Octonion
from .shadow import shadow_decompose

CHUNK_SIZE = 8


def make_sk_instance(n: int = 64, seed: int = 0) -> np.ndarray:
    """Symmetric SK coupling matrix, zero diagonal, J_ij ~ N(0, 1/n) for i<j."""
    rng = np.random.default_rng(seed)
    upper = rng.normal(loc=0.0, scale=1.0 / np.sqrt(n), size=(n, n))
    J = np.zeros((n, n))
    iu = np.triu_indices(n, k=1)
    J[iu] = upper[iu]
    return J + J.T


def energy(state: np.ndarray, J: np.ndarray) -> float:
    """E(s) = -sum_{i<j} J_ij s_i s_j == -0.5 * s @ J @ s (J symmetric, zero diagonal)."""
    return float(-0.5 * state @ J @ state)


def local_fields(state: np.ndarray, J: np.ndarray) -> np.ndarray:
    """h_i = sum_j J_ij s_j. ΔE for flipping spin i is 2 * state[i] * h[i]."""
    return J @ state
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_optimize.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add octonion_kernel/optimize.py tests/test_optimize.py
git commit -m "feat: Phase 4 SK-model instance, energy, and local fields"
```

---

### Task 2: The four proposal strategies

**Files:**
- Modify: `octonion_kernel/optimize.py` (append to the file from Task 1)
- Test: `tests/test_optimize.py` (append)

**Interfaces:**
- Consumes: `local_fields`, `energy`, `make_sk_instance` from Task 1;
  `Octonion` (`octonion_kernel/octonion.py` — constructor takes any length-8 array-like,
  `.coeffs` gives the 8-array), `shadow_decompose(a, b) -> ShadowResult` with `.associator`
  (an `Octonion`) from `octonion_kernel/shadow.py`.
- Produces: `propose_random(state, J, rng) -> int`, `propose_greedy(state, J, rng) -> int`,
  `propose_generic_nonlinear(state, J, rng) -> int`, `propose_shadow(state, J, rng) -> int`.
  All take `state: np.ndarray`, `J: np.ndarray`, `rng: np.random.Generator` and return a spin
  index in `[0, len(state))`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_optimize.py
import numpy as np
import pytest

from octonion_kernel.optimize import (
    make_sk_instance, propose_random, propose_greedy,
    propose_generic_nonlinear, propose_shadow,
)

_ALL_PROPOSE_FNS = [propose_random, propose_greedy, propose_generic_nonlinear, propose_shadow]


@pytest.mark.parametrize("propose_fn", _ALL_PROPOSE_FNS)
def test_propose_returns_valid_index(propose_fn):
    rng = np.random.default_rng(5)
    J = make_sk_instance(n=64, seed=5)
    state = rng.choice([-1.0, 1.0], size=64)
    for _ in range(10):
        i = propose_fn(state, J, rng)
        assert isinstance(i, int)
        assert 0 <= i < 64


def test_greedy_picks_largest_absolute_field():
    rng = np.random.default_rng(6)
    J = make_sk_instance(n=32, seed=6)
    state = rng.choice([-1.0, 1.0], size=32)
    from octonion_kernel.optimize import local_fields
    h = local_fields(state, J)
    i = propose_greedy(state, J, rng)
    assert i == int(np.argmax(np.abs(h)))


def test_generic_nonlinear_picks_largest_state_times_field():
    rng = np.random.default_rng(7)
    J = make_sk_instance(n=32, seed=7)
    state = rng.choice([-1.0, 1.0], size=32)
    from octonion_kernel.optimize import local_fields
    h = local_fields(state, J)
    i = propose_generic_nonlinear(state, J, rng)
    assert i == int(np.argmax(np.abs(state * h)))


def test_propose_shadow_selects_within_dominant_chunk():
    # Block-diagonal J: each 8-spin chunk's local field depends only on its own
    # spins, so scaling one chunk's coupling strength way up gives it a much
    # larger-magnitude associator and must make propose_shadow pick inside it.
    n = 16
    rng = np.random.default_rng(20)
    J = np.zeros((n, n))
    J[0:8, 0:8] = make_sk_instance(n=8, seed=100)
    J[8:16, 8:16] = make_sk_instance(n=8, seed=101) * 50.0
    state = rng.choice([-1.0, 1.0], size=n)
    i = propose_shadow(state, J, rng)
    assert 8 <= i < 16
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_optimize.py -v -k "propose"`
Expected: FAIL with `ImportError` (the four `propose_*` names don't exist yet)

- [ ] **Step 3: Write the implementation**

```python
# append to octonion_kernel/optimize.py

def propose_random(state: np.ndarray, J: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.integers(0, len(state)))


def propose_greedy(state: np.ndarray, J: np.ndarray, rng: np.random.Generator) -> int:
    """The 'trivial' baseline: argmax|h_i| directly, no chunking, no octonion structure."""
    h = local_fields(state, J)
    return int(np.argmax(np.abs(h)))


def propose_generic_nonlinear(state: np.ndarray, J: np.ndarray, rng: np.random.Generator) -> int:
    """Fixed, pre-declared elementwise combination score_i = |state_i * h_i|. Same
    per-chunk 16-numbers-per-chunk input as propose_shadow, no octonion algebra."""
    h = local_fields(state, J)
    return int(np.argmax(np.abs(state * h)))


def _shadow_chunk_scores(a_chunk: np.ndarray, b_chunk: np.ndarray) -> np.ndarray:
    """|associator_i| for one 8-spin chunk given its spins (a) and local fields (b)."""
    result = shadow_decompose(Octonion(a_chunk), Octonion(b_chunk))
    return np.abs(result.associator.coeffs)


def propose_shadow(state: np.ndarray, J: np.ndarray, rng: np.random.Generator) -> int:
    """Per chunk, score_i = |associator_i| from shadow_decompose(chunk spins, chunk
    fields); propose the global argmax. Fixed, pre-declared rule -- no open search."""
    n = len(state)
    h = local_fields(state, J)
    n_chunks = n // CHUNK_SIZE
    best_score = -np.inf
    best_idx = 0
    for c in range(n_chunks):
        start = c * CHUNK_SIZE
        end = start + CHUNK_SIZE
        scores = _shadow_chunk_scores(state[start:end], h[start:end])
        local_best = int(np.argmax(scores))
        if scores[local_best] > best_score:
            best_score = scores[local_best]
            best_idx = start + local_best
    return best_idx
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_optimize.py -v`
Expected: PASS (all tests from Task 1 and Task 2)

- [ ] **Step 5: Commit**

```bash
git add octonion_kernel/optimize.py tests/test_optimize.py
git commit -m "feat: Phase 4 proposal strategies (random, greedy, generic-nonlinear, shadow)"
```

---

### Task 3: The shared annealing loop

**Files:**
- Modify: `octonion_kernel/optimize.py` (append)
- Test: `tests/test_optimize.py` (append)

**Interfaces:**
- Consumes: `energy` from Task 1; any `propose_fn(state, J, rng) -> int` from Task 2.
- Produces: `anneal(J: np.ndarray, initial_state: np.ndarray, propose_fn, steps: int = 5000, t0: float = 2.0, t_min: float = 0.05, seed: int = 0) -> dict` returning
  `{"best_energy": float, "final_state": np.ndarray}`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_optimize.py
import numpy as np

from octonion_kernel.optimize import (
    make_sk_instance, energy, anneal, propose_random, propose_greedy,
)


def test_anneal_deterministic():
    J = make_sk_instance(n=64, seed=7)
    initial = np.random.default_rng(8).choice([-1.0, 1.0], size=64)
    r1 = anneal(J, initial, propose_greedy, steps=200, seed=9)
    r2 = anneal(J, initial, propose_greedy, steps=200, seed=9)
    assert r1["best_energy"] == r2["best_energy"]
    assert np.array_equal(r1["final_state"], r2["final_state"])


def test_anneal_best_energy_never_worse_than_initial():
    J = make_sk_instance(n=64, seed=10)
    initial = np.random.default_rng(11).choice([-1.0, 1.0], size=64)
    result = anneal(J, initial, propose_random, steps=300, seed=12)
    assert result["best_energy"] <= energy(initial, J) + 1e-9


def test_anneal_final_state_is_valid_spin_config():
    J = make_sk_instance(n=32, seed=13)
    initial = np.random.default_rng(14).choice([-1.0, 1.0], size=32)
    result = anneal(J, initial, propose_greedy, steps=150, seed=15)
    assert np.all(np.isin(result["final_state"], [-1.0, 1.0]))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_optimize.py -v -k anneal`
Expected: FAIL with `ImportError: cannot import name 'anneal'`

- [ ] **Step 3: Write the implementation**

```python
# append to octonion_kernel/optimize.py

def anneal(J: np.ndarray, initial_state: np.ndarray, propose_fn, steps: int = 5000,
           t0: float = 2.0, t_min: float = 0.05, seed: int = 0) -> dict:
    """Single-spin-flip simulated annealing. `propose_fn(state, J, rng) -> int` is the
    only thing that varies between arms; acceptance rule and cooling schedule are shared
    across every arm so the proposal rule is the sole controlled variable."""
    rng = np.random.default_rng(seed)
    state = initial_state.astype(float).copy()
    current_energy = energy(state, J)
    best_energy = current_energy
    if steps <= 1:
        temps = np.full(max(steps, 1), t0)
    else:
        temps = t0 * (t_min / t0) ** (np.arange(steps) / (steps - 1))
    for t in range(steps):
        T = temps[t]
        i = propose_fn(state, J, rng)
        h_i = float(J[i] @ state)
        dE = 2.0 * state[i] * h_i
        if dE <= 0.0 or rng.random() < np.exp(-dE / T):
            state[i] = -state[i]
            current_energy += dE
            if current_energy < best_energy:
                best_energy = current_energy
    return {"best_energy": float(best_energy), "final_state": state}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_optimize.py -v`
Expected: PASS (all tests so far)

- [ ] **Step 5: Commit**

```bash
git add octonion_kernel/optimize.py tests/test_optimize.py
git commit -m "feat: Phase 4 shared simulated-annealing loop"
```

---

### Task 4: The control harness — paired runs, power check, verdict

**Files:**
- Create: `octonion_kernel/optimize_controls.py`
- Test: `tests/test_optimize_controls.py`

**Interfaces:**
- Consumes: `make_sk_instance`, `anneal`, `propose_random`, `propose_greedy`,
  `propose_generic_nonlinear`, `propose_shadow` from `octonion_kernel/optimize.py` (Tasks 1-3).
- Produces: `run_optimize_control(n_instances: int = 500, n: int = 64, steps: int = 5000, seed: int = 0) -> dict` with keys `mean_energy` (dict of the 4 arm names to float),
  `power_check` (dict with `"greedy"`/`"generic_nonlinear"` keys, each
  `{"mean_advantage": float, "ci_lo": float, "ci_hi": float, "beats_random": bool}`),
  `power_check_passed: bool`, `best_baseline: str`, `sep_difference_ci: [float, float]`,
  `verdict: {"shadow_finds_better_optima": bool, "inconclusive": bool}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_optimize_controls.py
import numpy as np

from octonion_kernel.optimize_controls import run_optimize_control


def test_run_optimize_control_returns_well_formed_verdict():
    result = run_optimize_control(n_instances=8, n=16, steps=100, seed=0)
    assert set(result["mean_energy"].keys()) == {
        "random", "greedy", "generic_nonlinear", "shadow"
    }
    for energy_val in result["mean_energy"].values():
        assert np.isfinite(energy_val)
    assert result["best_baseline"] in ("random", "greedy", "generic_nonlinear")
    assert isinstance(result["power_check_passed"], bool)
    for arm in ("greedy", "generic_nonlinear"):
        p = result["power_check"][arm]
        assert isinstance(p["beats_random"], bool)
        assert p["ci_lo"] <= p["ci_hi"]
    lo, hi = result["sep_difference_ci"]
    assert lo <= hi
    assert isinstance(result["verdict"]["shadow_finds_better_optima"], bool)
    assert isinstance(result["verdict"]["inconclusive"], bool)


def test_run_optimize_control_deterministic():
    r1 = run_optimize_control(n_instances=6, n=16, steps=50, seed=42)
    r2 = run_optimize_control(n_instances=6, n=16, steps=50, seed=42)
    assert r1["mean_energy"] == r2["mean_energy"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_optimize_controls.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'octonion_kernel.optimize_controls'`

- [ ] **Step 3: Write the implementation**

```python
# octonion_kernel/optimize_controls.py
"""Control harness F -- does shadow-guided proposal find better optima?

Question: does the Jordan-Shadow associator, given exactly the same local information
(each chunk's 8 spins + 8 local fields) as a fixed non-octonion combination, produce a
move-proposal rule that reaches lower SK-model energy than random, greedy local-field,
and generic-nonlinear baselines? Same instance, same initial state, same cooling
schedule and step budget across every arm -- only propose_fn differs.

Before trusting any NO: greedy and generic_nonlinear must each reliably beat random, or
the run has no demonstrated power to separate any method from noise and the shadow
verdict is reported as inconclusive rather than NO.

Pure compute: returns dicts, no I/O.
"""
from __future__ import annotations

import numpy as np

from .optimize import (
    anneal, make_sk_instance, propose_generic_nonlinear, propose_greedy,
    propose_random, propose_shadow,
)

_ARMS = {
    "random": propose_random,
    "greedy": propose_greedy,
    "generic_nonlinear": propose_generic_nonlinear,
    "shadow": propose_shadow,
}
_BASELINE_ARMS = ("random", "greedy", "generic_nonlinear")


def bootstrap_mean_diff_ci(diffs: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """95% bootstrap CI of the mean of paired per-instance differences."""
    rng = np.random.default_rng(seed)
    n = len(diffs)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        means[b] = diffs[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def run_optimize_control(n_instances: int = 500, n: int = 64, steps: int = 5000,
                          seed: int = 0) -> dict:
    energies = {arm: np.empty(n_instances) for arm in _ARMS}
    for k in range(n_instances):
        instance_seed = seed * 1_000_003 + k
        J = make_sk_instance(n=n, seed=instance_seed)
        rng_init = np.random.default_rng(instance_seed + 500_000)
        initial_state = rng_init.choice([-1.0, 1.0], size=n)
        for arm, propose_fn in _ARMS.items():
            result = anneal(J, initial_state, propose_fn, steps=steps, seed=instance_seed)
            energies[arm][k] = result["best_energy"]

    means = {arm: float(energies[arm].mean()) for arm in _ARMS}

    power = {}
    for arm in ("greedy", "generic_nonlinear"):
        diff = energies["random"] - energies[arm]  # positive => arm beats random
        lo, hi = bootstrap_mean_diff_ci(diff, seed=seed + 1)
        power[arm] = {
            "mean_advantage": float(diff.mean()), "ci_lo": lo, "ci_hi": hi,
            "beats_random": bool(lo > 0.0),
        }
    power_ok = bool(power["greedy"]["beats_random"] and power["generic_nonlinear"]["beats_random"])

    best_baseline = min(_BASELINE_ARMS, key=lambda a: means[a])
    diff = energies[best_baseline] - energies["shadow"]  # positive => shadow wins
    diff_lo, diff_hi = bootstrap_mean_diff_ci(diff, seed=seed + 2)
    shadow_wins = bool(power_ok and diff_lo > 0.0)

    return {
        "mean_energy": means,
        "power_check": power,
        "power_check_passed": power_ok,
        "best_baseline": best_baseline,
        "sep_difference_ci": [diff_lo, diff_hi],
        "verdict": {
            "shadow_finds_better_optima": shadow_wins,
            "inconclusive": not power_ok,
        },
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_optimize_controls.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add octonion_kernel/optimize_controls.py tests/test_optimize_controls.py
git commit -m "feat: Phase 4 paired control harness with power check and verdict"
```

---

### Task 5: Wire the `[F]` report section into `harness_report.py`

**Files:**
- Modify: `harness_report.py:1-13` (docstring + imports), `harness_report.py:111-119` (`__main__`)

**Interfaces:**
- Consumes: `run_optimize_control` from Task 4.

- [ ] **Step 1: Update the module docstring and imports**

In `harness_report.py`, replace the top docstring/import block:

```python
"""Runnable A/B/C/D/E/F control report. Run from the repo root: python harness_report.py

A: algebra correctness  (delegates to pytest tests/test_algebra.py)
B: Jordan-Shadow identity residuals over random pairs
C: the anti-castle control — does the associator beat the best trivial baseline?
D: dynamics control — does iterating the octonion walk add linear separability?
E: topology control — does the walk's trajectory carry more loop structure?
F: optimizer control — does shadow-guided move-proposal reach lower SK-model energy
   than random/greedy/generic-nonlinear baselines in simulated annealing?
"""
import numpy as np

from octonion_kernel import Octonion, identity_residuals
from octonion_kernel.controls import run_control_c
from octonion_kernel.dynamics_controls import run_dynamics_control
from octonion_kernel.topology_controls import run_topology_control
from octonion_kernel.optimize_controls import run_optimize_control
```

- [ ] **Step 2: Add `report_f()`**

Insert immediately after `report_e()` (before the `if __name__ == "__main__":` block):

```python
def report_f(n_instances=500, n=64, steps=5000, seed=0):
    out = run_optimize_control(n_instances=n_instances, n=n, steps=steps, seed=seed)
    print(f"\n[F] Optimizer control ({n_instances} paired SK-model instances, n={n} spins, "
          f"{steps}-step simulated annealing, shared cooling schedule):")
    print(f"    {'arm':<18} {'mean best-energy':>18}")
    for k in ("random", "greedy", "generic_nonlinear", "shadow"):
        print(f"    {k:<18} {out['mean_energy'][k]:>18.4f}")
    print("\n    power check (arm must reliably beat random):")
    for arm, p in out["power_check"].items():
        status = "OK" if p["beats_random"] else "FAILED"
        print(f"    {arm:<18} advantage {p['mean_advantage']:>8.4f}  "
              f"95% CI [{p['ci_lo']:.4f}, {p['ci_hi']:.4f}]  {status}")
    print(f"\n    best baseline: {out['best_baseline']} "
          f"(mean best-energy {out['mean_energy'][out['best_baseline']]:.4f})")
    print(f"    shadow mean best-energy: {out['mean_energy']['shadow']:.4f}")
    print(f"    mean[best_baseline_energy - shadow_energy], 95% CI: "
          f"[{out['sep_difference_ci'][0]:.4f}, {out['sep_difference_ci'][1]:.4f}]")
    v = out["verdict"]
    if v["inconclusive"]:
        print("    VERDICT: INCONCLUSIVE - the power check failed (greedy and/or")
        print("             generic-nonlinear did not reliably beat random), so this run has")
        print("             no demonstrated ability to separate any method from noise.")
    elif v["shadow_finds_better_optima"]:
        print("    VERDICT: YES - the shadow-guided proposal reaches reliably lower energy")
        print("             than every declared baseline (random, greedy local-field, and a")
        print("             matched generic-nonlinear combination of the same information).")
    else:
        print("    VERDICT: NO - the shadow-guided proposal does NOT beat the best baseline.")
        print("             Routing the same local-field information through the")
        print("             Jordan-Shadow associator is not doing useful search-guidance")
        print("             work here.")
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
    report_f()
    print("=" * 64)
```

- [ ] **Step 4: Run the full suite and the report**

Run: `python -m pytest -q`
Expected: all tests pass, pristine output (no warnings)

Run: `python harness_report.py`
Expected: sections `[B]` through `[F]` all print, including a recorded `[F]` verdict
(YES, NO, or INCONCLUSIVE)

- [ ] **Step 5: Commit**

```bash
git add harness_report.py
git commit -m "feat: wire [F] optimizer control section into harness_report.py"
```

---

## Definition of done

- `python -m pytest -q` green (all Phase 1-4 tests).
- `python harness_report.py` prints `[B]` through `[F]`, `[F]` ending in a recorded verdict.
- No scipy/sklearn/aoi imports anywhere in `optimize.py`; `optimize_controls.py` uses only numpy.
