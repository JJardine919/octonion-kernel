# Octonion Topology Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute a persistent-homology summary of an octonion walk's trajectory (via ripser) and a control that honestly tests whether the octonion walk's trajectory carries more loop structure (max-H1 persistence) than matched linear, generic-nonlinear, and random-walk (diffusion) trajectories.

**Architecture:** A harness-tier `octonion_kernel/topology.py` (trajectory → point cloud → ripser persistence summary) and `octonion_kernel/topology_controls.py` (trajectory runner, iid null, paired-bootstrap, the experiment `run_topology_control`). Both reuse the Phase-2 walk and baseline maps unchanged. `harness_report.py` gains an `[E]` section. The pure engine (`dynamics.py`) and kernel stay numpy-only; ripser lives only in the topology modules.

**Tech Stack:** Python 3.14, numpy, ripser (0.6.15, installed) for persistent homology, pytest. scipy/sklearn already present but not needed here.

## Global Constraints

- **Python 3.14.** `ripser` is allowed **only** in `octonion_kernel/topology.py` and `octonion_kernel/topology_controls.py`. It must **never** be imported by the kernel (`octonion.py`, `shadow.py`) or the pure engine (`dynamics.py`). No `aoi_collapse` anywhere.
- **Verdict metric is `max_h1`** (the single longest H1 loop lifetime) — NOT `total_h1` (which rewards noise loops). `total_h1`, `n_h1`, `total_h0` are reported for context only.
- **Distance metric:** Euclidean on the 8 coefficients (ripser default).
- **Walk length for topology:** `steps = 256` (→ 257-point clouds); `lam = 0.5`. Pinned defaults.
- **Reuse Phase-2 maps unchanged:** `make_linear_map`, `make_generic_nonlinear_map`, `make_random_walk_step`, `octonion_step` from `dynamics_controls.py`; `renorm`, `DEFAULT_GENERATOR` from `dynamics.py`. Do not modify `dynamics.py` or `dynamics_controls.py`.
- **Gating baselines:** the three matched dynamical maps `linear`, `generic_nonlinear`, `random_walk` (random_walk = the dynamical noise null). `iid_cloud` is **reported only, never gating**.
- **Verdict rule:** octonion "adds topology" iff `mean_maxH1(octonion) > mean_maxH1(best_baseline)` (argmax over the three gating baselines) AND the paired-bootstrap 95% CI of the mean difference excludes 0 on the positive side.
- **Anti-castle rule (critical):** the control test asserts only that the harness runs, returns a verdict bool, the best baseline is one of the three gating baselines, max-H1 values are finite and ≥ 0, and the CI is well-formed. It must **NOT** assert the octonion walk wins or loses. A NO is valid.
- **Determinism:** all randomness via seeded `numpy.random.default_rng`; no `hypothesis`.
- **Run location:** repo root `C:\Users\jim\octonion-kernel`; `python -m pytest ...`. The `slow` marker is registered in `pytest.ini`.

## File Structure

- `octonion_kernel/topology.py` — `trajectory_cloud`, `persistence_summary` (ripser).
- `octonion_kernel/topology_controls.py` — `run_map_trajectory`, `iid_cloud`, `_bootstrap_mean_diff_ci`, `run_topology_control`.
- `tests/test_topology.py` — PH summary correctness (circle vs cluster), determinism.
- `tests/test_topology_controls.py` — building-block checks + the agnostic experiment check.
- `harness_report.py` — add `report_e` and call it in `__main__`.
- `README.md` — add a short "Phase 3 — topology" note (verdict filled from the actual run).

---

### Task 1: Persistent-homology summary

**Files:**
- Create: `octonion_kernel/topology.py`
- Create: `tests/test_topology.py`

**Interfaces:**
- Consumes: `Octonion` (`.coeffs`, shape `(8,)`); `ripser.ripser`.
- Produces:
  - `trajectory_cloud(traj: list[Octonion]) -> np.ndarray` — `(len(traj), 8)` float array.
  - `persistence_summary(traj: list[Octonion]) -> dict` — `{"max_h1": float, "total_h1": float, "n_h1": int, "total_h0": float}`. `max_h1` is the verdict metric.

- [ ] **Step 1: Write the failing test**

Create `tests/test_topology.py`:

```python
import numpy as np
from octonion_kernel import Octonion
from octonion_kernel.topology import trajectory_cloud, persistence_summary


def _circle_traj(n=60):
    # n points on a circle in the first 2 octonion coords (the rest zero)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [Octonion(np.array([np.cos(a), np.sin(a), 0, 0, 0, 0, 0, 0])) for a in t]


def _cluster_traj(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return [Octonion(np.concatenate([[1.0], 0.001 * rng.standard_normal(7)])) for _ in range(n)]


def test_trajectory_cloud_shape_and_values():
    cloud = trajectory_cloud(_circle_traj(10))
    assert cloud.shape == (10, 8)
    assert np.allclose(cloud[:, 2:], 0.0)


def test_circle_has_a_persistent_loop():
    s = persistence_summary(_circle_traj(60))
    assert s["n_h1"] >= 1
    assert s["max_h1"] > 0.5  # the circle's loop is long-lived at radius scale


def test_tight_cluster_has_no_real_loop():
    s = persistence_summary(_cluster_traj(60))
    assert s["max_h1"] < 0.05  # a tight blob carries no real loop


def test_persistence_summary_is_deterministic():
    traj = _circle_traj(40)
    assert persistence_summary(traj) == persistence_summary(traj)


def test_summary_keys_and_types():
    s = persistence_summary(_circle_traj(30))
    assert set(s) == {"max_h1", "total_h1", "n_h1", "total_h0"}
    assert all(isinstance(s[k], float) for k in ("max_h1", "total_h1", "total_h0"))
    assert isinstance(s["n_h1"], int)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_topology.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'octonion_kernel.topology'`.

- [ ] **Step 3: Write minimal implementation**

Create `octonion_kernel/topology.py`:

```python
"""Persistent-homology summary of an octonion walk's trajectory.

Harness-tier: imports ripser (and numpy). NEVER imported by the kernel
(octonion, shadow) or the pure engine (dynamics.py). No aoi_collapse.
"""
from __future__ import annotations

import numpy as np
from ripser import ripser

from .octonion import Octonion


def trajectory_cloud(traj: list[Octonion]) -> np.ndarray:
    """Stack a trajectory's coefficients into an (len(traj), 8) array."""
    return np.array([x.coeffs for x in traj], dtype=float)


def persistence_summary(traj: list[Octonion]) -> dict:
    """Persistent-homology summary of the trajectory point cloud (Euclidean, maxdim=1).

    Returns:
      max_h1   - single longest H1 (loop) lifetime, 0.0 if none [VERDICT METRIC]
      total_h1 - sum of all H1 lifetimes (context)
      n_h1     - number of H1 features (context)
      total_h0 - sum of finite H0 lifetimes (context)
    """
    cloud = trajectory_cloud(traj)
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
    return {"max_h1": max_h1, "total_h1": total_h1, "n_h1": n_h1, "total_h0": total_h0}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_topology.py -q`
Expected: PASS — 5 passed.

- [ ] **Step 5: Commit**

```bash
git add octonion_kernel/topology.py tests/test_topology.py
git commit -m "feat: persistent-homology trajectory summary (ripser, max-H1 verdict metric)"
```

---

### Task 2: Topology control building blocks

**Files:**
- Create: `octonion_kernel/topology_controls.py`
- Create: `tests/test_topology_controls.py`

**Interfaces:**
- Consumes: `Octonion`; `renorm`, `DEFAULT_GENERATOR` from `dynamics.py`; `make_random`, `make_linear_map`, `make_generic_nonlinear_map`, `make_random_walk_step`, `octonion_step` from `dynamics_controls.py`; `persistence_summary` from `topology.py`.
- Produces:
  - `run_map_trajectory(x0: Octonion, step_fn, lam=0.5, steps=256) -> list[Octonion]` — full trajectory `[x0_normalized, …, x_T]`, halts on norm < 1e-12.
  - `iid_cloud(rng, n_points: int) -> list[Octonion]` — iid-uniform unit octonions.
  - `_bootstrap_mean_diff_ci(a, b, n_boot=2000, seed=0) -> tuple[float,float]` — 95% CI of `mean(a) − mean(b)` over shared (paired) resamples.

- [ ] **Step 1: Write the failing test**

Create `tests/test_topology_controls.py`:

```python
import numpy as np
import pytest
from octonion_kernel import Octonion
from octonion_kernel.dynamics import DEFAULT_GENERATOR
from octonion_kernel.dynamics_controls import octonion_step, make_random
from octonion_kernel.topology_controls import (
    run_map_trajectory, iid_cloud, _bootstrap_mean_diff_ci,
)


def test_run_map_trajectory_unit_norm_length_and_deterministic():
    step = octonion_step(DEFAULT_GENERATOR)
    rng = np.random.default_rng(0)
    x0 = make_random(rng)
    t1 = run_map_trajectory(x0, step, steps=32)
    t2 = run_map_trajectory(x0, step, steps=32)
    assert len(t1) == 33  # x0 + 32 steps (this seed does not underflow-halt)
    for x in t1:
        assert abs(x.norm() - 1.0) < 1e-10
    for a, b in zip(t1, t2):
        assert a.approx_eq(b, tol=1e-12)


def test_iid_cloud_unit_norm_and_count():
    rng = np.random.default_rng(1)
    cloud = iid_cloud(rng, 50)
    assert len(cloud) == 50
    for x in cloud:
        assert abs(x.norm() - 1.0) < 1e-12


def test_bootstrap_mean_diff_ci_brackets_positive_difference():
    rng = np.random.default_rng(2)
    a = rng.normal(1.0, 0.1, 300)
    b = rng.normal(0.0, 0.1, 300)
    lo, hi = _bootstrap_mean_diff_ci(a, b, n_boot=500, seed=3)
    assert lo <= hi
    assert lo > 0.0  # clearly positive difference, CI excludes 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_topology_controls.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'octonion_kernel.topology_controls'`.

- [ ] **Step 3: Write minimal implementation**

Create `octonion_kernel/topology_controls.py`:

```python
"""Control harness for the octonion topology layer (Phase 3).

Harness-tier: imports numpy and ripser (via topology). NEVER imported by the
kernel or the pure engine (dynamics.py). No aoi_collapse. Pure compute: returns
dicts, no I/O.
"""
from __future__ import annotations

import numpy as np

from .octonion import Octonion
from .dynamics import renorm, DEFAULT_GENERATOR
from .dynamics_controls import (
    make_random, make_linear_map, make_generic_nonlinear_map,
    make_random_walk_step, octonion_step,
)
from .topology import persistence_summary


def run_map_trajectory(x0: Octonion, step_fn, lam: float = 0.5, steps: int = 256) -> list[Octonion]:
    """Iterate renorm(lam*x + (1-lam)*step_fn(x)) from x0; return the FULL trajectory
    [x0_normalized, x1, ..., x_T]. Halts (returns trajectory so far) on norm < 1e-12."""
    traj = [renorm(x0)]
    for _ in range(steps):
        x = traj[-1]
        y = Octonion(lam * x.coeffs + (1.0 - lam) * step_fn(x).coeffs)
        if y.norm() < 1e-12:
            break
        traj.append(renorm(y))
    return traj


def iid_cloud(rng, n_points: int) -> list[Octonion]:
    """n_points iid-uniform unit octonions (a structureless point-cloud null)."""
    return [make_random(rng) for _ in range(n_points)]


def _bootstrap_mean_diff_ci(a, b, n_boot: int = 2000, seed: int = 0):
    """95% CI of mean(a) - mean(b) over shared (paired) resamples of the index set."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = a[idx].mean() - b[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_topology_controls.py -q`
Expected: PASS — 3 passed.

- [ ] **Step 5: Commit**

```bash
git add octonion_kernel/topology_controls.py tests/test_topology_controls.py
git commit -m "feat: topology control building blocks (trajectory runner, iid null, paired bootstrap)"
```

---

### Task 3: The topology experiment + verdict

**Files:**
- Modify: `octonion_kernel/topology_controls.py` (add `run_topology_control`)
- Modify: `tests/test_topology_controls.py` (append the agnostic experiment test)

**Interfaces:**
- Consumes: everything from Task 2 + the Phase-2 map factories.
- Produces:
  - `run_topology_control(n=200, steps=256, lam=0.5, seed=0) -> dict` returning:
    `{"max_h1": {map: mean_float}, "context": {map: {total_h1,n_h1,total_h0}}, "iid_cloud_max_h1": float, "verdict": {octonion_adds_topology: bool, best_baseline: str, octonion_max_h1: float, best_baseline_max_h1: float, diff_ci: [lo,hi]}}` where `map ∈ {linear, generic_nonlinear, random_walk, octonion}`.

- [ ] **Step 1: Write the failing test (append to `tests/test_topology_controls.py`)**

```python
from octonion_kernel.topology_controls import run_topology_control


@pytest.mark.slow
def test_topology_control_runs_and_is_agnostic():
    out = run_topology_control(n=30, steps=64, seed=0)
    v = out["verdict"]
    # a verdict is produced regardless of its value (do NOT assert YES/NO)
    assert isinstance(v["octonion_adds_topology"], bool)
    assert v["best_baseline"] in ("linear", "generic_nonlinear", "random_walk")
    for k in ("linear", "generic_nonlinear", "random_walk", "octonion"):
        assert np.isfinite(out["max_h1"][k]) and out["max_h1"][k] >= 0.0
    assert np.isfinite(out["iid_cloud_max_h1"]) and out["iid_cloud_max_h1"] >= 0.0
    lo, hi = v["diff_ci"]
    assert lo <= hi
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_topology_controls.py::test_topology_control_runs_and_is_agnostic -q`
Expected: FAIL — `ImportError: cannot import name 'run_topology_control'`.

- [ ] **Step 3: Write minimal implementation (append to `octonion_kernel/topology_controls.py`)**

```python
def run_topology_control(n: int = 200, steps: int = 256, lam: float = 0.5, seed: int = 0) -> dict:
    """Run the topology richness control; return per-map mean max-H1 and the verdict."""
    g = DEFAULT_GENERATOR
    rng = np.random.default_rng(seed)
    x0s = [make_random(rng) for _ in range(n)]

    maps = {
        "linear": make_linear_map(g, seed=seed + 1),
        "generic_nonlinear": make_generic_nonlinear_map(g, seed=seed + 2),
        "random_walk": make_random_walk_step(g, seed=seed + 3),
        "octonion": octonion_step(g),
    }

    per_map_maxh1 = {k: [] for k in maps}
    context = {k: {"total_h1": 0.0, "n_h1": 0.0, "total_h0": 0.0} for k in maps}
    for k, step in maps.items():
        for x0 in x0s:
            s = persistence_summary(run_map_trajectory(x0, step, lam, steps))
            per_map_maxh1[k].append(s["max_h1"])
            context[k]["total_h1"] += s["total_h1"]
            context[k]["n_h1"] += s["n_h1"]
            context[k]["total_h0"] += s["total_h0"]
    for k in maps:
        for key in context[k]:
            context[k][key] /= n

    mean_maxh1 = {k: float(np.mean(per_map_maxh1[k])) for k in maps}

    # iid_cloud sanity null (reported, not gating): clouds of one trajectory's size
    iid_rng = np.random.default_rng(seed + 50)
    iid_vals = [persistence_summary(iid_cloud(iid_rng, steps + 1))["max_h1"] for _ in range(n)]
    iid_cloud_max_h1 = float(np.mean(iid_vals))

    baseline_keys = ("linear", "generic_nonlinear", "random_walk")
    best_baseline = max(baseline_keys, key=lambda k: mean_maxh1[k])
    diff_lo, diff_hi = _bootstrap_mean_diff_ci(
        per_map_maxh1["octonion"], per_map_maxh1[best_baseline], seed=seed + 100)
    adds = bool(mean_maxh1["octonion"] > mean_maxh1[best_baseline] and diff_lo > 0.0)

    return {
        "max_h1": mean_maxh1,
        "context": context,
        "iid_cloud_max_h1": iid_cloud_max_h1,
        "verdict": {
            "octonion_adds_topology": adds,
            "best_baseline": best_baseline,
            "octonion_max_h1": mean_maxh1["octonion"],
            "best_baseline_max_h1": mean_maxh1[best_baseline],
            "diff_ci": [diff_lo, diff_hi],
        },
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_topology_controls.py -q`
Expected: PASS — 4 passed (the experiment test is marked slow but still runs; a few seconds at n=30, steps=64).

- [ ] **Step 5: Commit**

```bash
git add octonion_kernel/topology_controls.py tests/test_topology_controls.py
git commit -m "feat: topology experiment with paired-bootstrap max-H1 verdict (anti-castle, agnostic test)"
```

---

### Task 4: Report section, README note, and final verification

**Files:**
- Modify: `harness_report.py` (add `report_e`, call it in `__main__`)
- Modify: `README.md` (add a Phase-3 note)

**Interfaces:**
- Consumes: `run_topology_control` from `topology_controls.py`.
- Produces: an `[E]` console section; no new programmatic API.

- [ ] **Step 1: Add the report section**

In `harness_report.py`, add near the other `octonion_kernel` imports:

```python
from octonion_kernel.topology_controls import run_topology_control
```

Add this function after `report_d`:

```python
def report_e(n=200, steps=256, seed=0):
    out = run_topology_control(n=n, steps=steps, seed=seed)
    print(f"\n[E] Topology control ({n} random initial states, {steps}-step walks, "
          f"max-H1 persistence over the trajectory):")
    print(f"    {'map':<18} {'mean max-H1':>12}")
    for k in ("linear", "generic_nonlinear", "random_walk", "octonion"):
        print(f"    {k:<18} {out['max_h1'][k]:>12.4f}")
    print(f"    {'iid_cloud (null)':<18} {out['iid_cloud_max_h1']:>12.4f}")
    v = out["verdict"]
    print(f"\n    best baseline:  {v['best_baseline']} (mean max-H1 {v['best_baseline_max_h1']:.4f})")
    print(f"    octonion mean max-H1: {v['octonion_max_h1']:.4f}")
    print(f"    mean[maxH1(octonion) - maxH1(best baseline)], 95% CI: "
          f"[{v['diff_ci'][0]:.4f}, {v['diff_ci'][1]:.4f}]")
    if v["octonion_adds_topology"]:
        print("    VERDICT: YES - the octonion walk's trajectory carries more persistent loop")
        print("             structure than every matched baseline (linear / generic-nonlinear /")
        print("             random-walk diffusion).")
    else:
        print("    VERDICT: NO - the octonion walk does NOT produce more loop structure than the")
        print("             best matched baseline. Its trajectory topology is not distinctive")
        print("             beyond a linear / generic-nonlinear / diffusion process.")
```

In the `if __name__ == "__main__":` block, add `report_e()` after the `report_d()` call (before the final `print("=" * 64)`).

- [ ] **Step 2: Verify the report runs and produces an [E] verdict (with a power check)**

Run: `python harness_report.py`
Expected: prints `[A]`–`[E]`; the `[E]` section shows the 5-row mean max-H1 table and a VERDICT (YES or NO). **Record the exact `[E]` output** for Step 4.

POWER CHECK (Phase-2 lesson — outcome-neutral): confirm the max-H1 table is **not all near-zero**. If *every* gating baseline AND the octonion map show `mean max-H1 < 0.05` (the 257-point cloud is too sparse for any loops), the comparison is underpowered — raise `steps` to `512` in `report_e`'s default and the experiment, re-run, and report the new table. Adjust **only** `steps` to give the test power, **never** to change the YES/NO verdict.

- [ ] **Step 3: Run the full suite and the purity check**

Run: `python -m pytest -q`
Expected: all pass plus the pre-existing legacy skip (`<N> passed, 1 skipped`). Output pristine.

Confirm the engine and kernel never import ripser/scipy/sklearn (only the topology modules may):

Run: `python -c "import ast
for f in ('octonion_kernel/dynamics.py','octonion_kernel/octonion.py','octonion_kernel/shadow.py'):
    t=ast.parse(open(f).read())
    imps=[a.name for n in ast.walk(t) if isinstance(n,ast.Import) for a in n.names]+[n.module or '' for n in ast.walk(t) if isinstance(n,ast.ImportFrom)]
    bad=[m for m in imps if any(w in m for w in ('ripser','scipy','sklearn','aoi'))]
    assert not bad, (f,bad)
print('engine+kernel clean')"`
Expected: `engine+kernel clean`

- [ ] **Step 4: Add the README note**

In `README.md`, add this subsection at the end, filling `<VERDICT …>` with the REAL `[E]` verdict + numbers from Step 2:

```markdown
## Phase 3 — topology

`octonion_kernel/topology.py` computes a persistent-homology summary (via `ripser`, Euclidean
distance) of a walk's trajectory; the verdict metric is **max-H1 persistence** (the longest loop;
`total_h1` is reported but not gated on, because it rewards noise loops). The control
(`topology_controls.py`, `[E]` section of `harness_report.py`) asks whether the octonion walk's
256-step trajectory carries more loop structure than matched **linear, generic-nonlinear, and
random-walk (diffusion)** trajectories — the random-walk being the dynamical noise null. The
octonion walk counts only if its mean max-H1 beats the best baseline with a paired-bootstrap CI
excluding zero. An `iid_cloud` scatter is reported as a sanity null but does not gate.

**Phase-3 result:** <VERDICT — copy the [E] verdict from `python harness_report.py`, e.g. "NO —
octonion mean max-H1 X.XXX vs <baseline> X.XXX">. A NO is a valid, expected outcome by design.
```

- [ ] **Step 5: Commit**

```bash
git add harness_report.py README.md
git commit -m "feat: [E] topology control report section + README Phase-3 note"
```

---

## Definition of Done (Phase 3)

- `topology.py` computes `persistence_summary` via ripser; circle-vs-cluster correctness tests pass.
- `topology_controls.py` implements the trajectory runner, iid null, paired-bootstrap, and `run_topology_control`; the control test passes and is **agnostic** to the verdict value.
- `harness_report.py` prints the `[E]` section with the max-H1 table and a recorded verdict (YES or NO).
- Full `python -m pytest -q` green (plus the legacy skip); `python harness_report.py` produces A–E.
- Engine + kernel purity intact (Step-3 check prints `engine+kernel clean`); ripser only in the topology modules.

## Self-Review notes

- **Spec coverage:** §4 summary (`max_h1` metric, Euclidean, 256-step cloud) → Task 1 + Global Constraints; §5.1 paired trajectory generators → Task 2 (`run_map_trajectory`) + Task 3 (wiring); §5.2 gating baselines + random-walk noise null + iid sanity null → Task 3; §5.3 verdict rule (beat best of three, paired-bootstrap CI > 0) → Task 3; §5.4 anti-castle agnostic test → Task 3; §6 components → Tasks 1–3; §7 testing → all tasks; §8 DoD → Task 4; §10 power-calibration → Task 4 Step 2. No gaps.
- **Type consistency:** `persistence_summary` returns `{max_h1, total_h1, n_h1, total_h0}` used identically in Task 3 and the report. `run_topology_control` keys (`max_h1`, `context`, `iid_cloud_max_h1`, `verdict.{octonion_adds_topology, best_baseline, octonion_max_h1, best_baseline_max_h1, diff_ci}`) match between `topology_controls.py`, the Task-3 test, and `report_e`. `run_map_trajectory(x0, step_fn, lam, steps)` and `_bootstrap_mean_diff_ci(a, b, n_boot, seed)` signatures match call sites. Reused Phase-2 names (`make_random`, `make_linear_map`, `make_generic_nonlinear_map`, `make_random_walk_step`, `octonion_step`, `renorm`, `DEFAULT_GENERATOR`) match their Phase-2 definitions.
- **Anti-castle:** the only verdict-bearing test asserts structure/ranges/CI well-formedness, never the YES/NO value.
- **Purity:** ripser imported only in `topology.py`/`topology_controls.py`; Task 4 Step 3 verifies the engine+kernel stay clean.
