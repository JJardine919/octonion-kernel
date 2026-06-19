# Octonion Topology Layer + Control — Design Spec

**Date:** 2026-06-19
**Author:** James Jardine (Lattice24 / 2396179 Alberta Inc.)
**Status:** Approved design — Phase 3 of the "octonion product" build.

---

## 1. Context & motivation

Phase 1 (algebra kernel) and Phase 2 (dynamics) are complete. Both returned honest **NO** verdicts:
the Jordan-Shadow associator carries no information beyond trivial statistics (Phase 1), and
iterating the octonion walk *erodes* linear separability rather than creating it (Phase 2). Each
NO was made trustworthy only after strengthening its control to have real power.

This phase, **topology**, asks a question neither earlier layer touched: does the *shape* of the
octonion walk's trajectory carry non-trivial **loop structure** (persistent H1) that matched
generic processes — including pure noise — do not? Persistent homology sees structure that
separability and pointwise statistics miss, so a NO from Phases 1–2 does not settle it.

The governing principle is unchanged, and this layer is where it bites hardest:

> **Every layer ships with its own control/baseline before anything is built on top of it.**
> A NO is a valid, expected outcome.

This program has been burned by topology-on-noise before (a grade detector that scored pure noise
as full structure). Random points throw off persistent-homology features for free, so the control
here is built specifically to subtract the topology that *any* process — especially a matched
diffusion — produces.

### Where this sits in the stack
1. Algebra kernel — *done (Phase 1, NO)*
2. Dynamics — *done (Phase 2, NO)*
3. **Topology (this spec)** — persistent homology over the walk's trajectory, gated by a control
4. Application adapter (finance / bio) — later
5. Product surface — later

## 2. Goal & non-goals

**Goal:** a harness-tier module that computes a persistent-homology summary of a walk's trajectory,
plus a control that honestly tests whether the octonion walk's trajectory carries more **loop
structure** (max H1 persistence) than matched linear, generic-nonlinear, and random-walk
(diffusion) trajectories.

**Non-goals (Phase 3):**
- No 24D/96D embedding; no domain data. Trajectories come from synthetic octonion states
  (Control-C style), seeded into the Phase-2 walk and baseline maps.
- No application logic; no product surface.
- No claim about what any topological feature *means* — only whether the octonion walk provably
  produces loop structure the baselines do not.
- No new dynamics: this layer consumes the Phase-2 walk/baseline maps unchanged.

## 3. Scope boundary & dependency posture

- Persistent homology requires **`ripser`** (installed, verified: a 60-point circle yields exactly
  one H1 feature). Therefore the topology code is **harness-tier**, never kernel-pure.
- `octonion_kernel/topology.py` and `octonion_kernel/topology_controls.py` may import numpy, scipy,
  sklearn, and **ripser**. They must **never** be imported by the kernel (`octonion`, `shadow`) or
  the pure engine (`dynamics.py`), and must never import `aoi_collapse`.
- Dependency direction stays one-way: kernel → nothing; dynamics engine → kernel; topology →
  dynamics engine + dynamics_controls (for the baseline maps) + ripser; report → all.
- `harness_report.py` (the only file that prints) gains an `[E]` section.

## 4. The topological summary

**Point cloud.** A walk's trajectory `[x₀, …, x_T]` is a set of `T+1` points on S⁷. The default
32-step walk yields only 33 points — far too sparse for meaningful loops. The topology experiment
therefore uses **longer walks**: `steps = 256` (→ 257-point clouds), pinned/declared. Persistent
homology on a few hundred 8-D points via ripser is fast (< ~0.1 s/cloud).

**Distance metric.** Euclidean (chord) distance on the 8 coefficients — declared (ripser default).

**The summary.** `persistence_summary(trajectory) -> dict` computes, via `ripser(cloud, maxdim=1)`:
- `max_h1` — the **single longest H1 (loop) lifetime** (`max(death − birth)` over H1; `0.0` if no
  H1 features). **This is the verdict-bearing metric.**
- `total_h1` — sum of all H1 lifetimes (reported for context only).
- `n_h1` — count of H1 features (reported for context only).
- `total_h0` — sum of finite H0 lifetimes (reported for context only).

**Why `max_h1`, not `total_h1` (decisive choice).** A cloud of random points throws off *many
short-lived* H1 loops, so `total_h1` can score **noise higher** than a structured trajectory that
carries a *few long* loops — the exact topology-on-noise trap this program has hit. `max_h1`
(longest-lived loop) is the standard "is there a *real* loop here" signal and is robust to
noise-loop inflation. The verdict gates on this single pre-declared scalar (a discipline tightening
over Phase 1's three summaries — no multiple-comparison gaming).

## 5. The control — built to say NO

### 5.1 Trajectory generators (paired, declared)
For a fixed seeded set of `N` initial unit octonions (uniform on S⁷ — no class labels are needed;
this layer asks about intrinsic trajectory shape, not class separability), each initial state is
run through **every** map with identical `λ` and `steps`, so the maps are compared on the **same**
initial states (paired). Only the per-step update term differs:
- **octonion** — the Phase-2 walk `renorm(λ·x + (1−λ)·associator(x, g))`.
- **linear** — `renorm(λ·x + (1−λ)·A·x)`, scale-matched `A` (reuses Phase-2 `make_linear_map`).
- **generic_nonlinear** — `renorm(λ·x + (1−λ)·Q(x))`, scale-matched random quadratic (reuses
  Phase-2 `make_generic_nonlinear_map`).
- **random_walk** — `renorm(λ·x + (1−λ)·η)`, fresh scale-matched noise each step (reuses Phase-2
  `make_random_walk_step`).

A trajectory-returning runner `run_map_trajectory(x0, step_fn, lam, steps) -> list[Octonion]` is
added (the Phase-2 `run_map` returns only the final state; topology needs the whole trajectory).
For the octonion map this is equivalent to `octonion_walk`.

### 5.2 The gating baselines and the noise nulls
- **The three matched dynamical baselines the octonion walk must beat:** `linear`,
  `generic_nonlinear`, `random_walk`. The **random_walk is the dynamical noise null** — a matched
  diffusion on the sphere that wanders and makes loops; if the octonion walk's `max_h1` ≈ the
  random walk's, the octonion dynamics adds no loop structure beyond noise.
- **`iid_cloud` (reported sanity null, NOT gating):** `T+1` points drawn iid-uniform on S⁷, no
  dynamics. Reported for context (the purest "topology from random points"). It is deliberately
  **not** a gating baseline: a random scatter produces many short noise-loops and a different
  cloud geometry than a smooth correlated trajectory, so comparing the two would be
  apples-to-oranges. The discriminating bars are the three matched dynamical processes.

### 5.3 Verdict rule
Per map, run `N` trajectories and compute the per-trajectory `max_h1`; the map's score is the
**mean `max_h1`** over its `N` trajectories. The octonion walk **produces distinctive loop
structure** iff:
- `mean_maxH1(octonion) > mean_maxH1(best_baseline)`, where `best_baseline = argmax` over
  {linear, generic_nonlinear, random_walk}, **and**
- the paired-bootstrap 95% CI of the mean difference `maxH1(octonion) − maxH1(best_baseline)`
  (resampling the `N` paired initial states) excludes 0 on the positive side.

This is deliberately one-sided ("richer"): a walk that collapses (less H1) or merely matches the
diffusion null is a **NO**. Anything else → NO. The verdict dict reports every map's mean `max_h1`
(plus the context summaries and `iid_cloud`), the chosen best baseline, the difference CI, and the
boolean.

### 5.4 The control test (anti-castle discipline)
The pytest test asserts only: the harness runs and returns a verdict; the boolean is a bool; the
chosen best baseline is one of the three declared dynamical baselines; the reported `max_h1` values
are finite and ≥ 0; the difference CI is well-formed (`lo ≤ hi`). It must **NOT** assert the
octonion walk wins or loses. A NO is valid; forcing a YES would rebuild the castle this phase
exists to prevent.

## 6. Components & interfaces

`octonion_kernel/topology.py` (harness; ripser allowed):
- `trajectory_cloud(traj: list[Octonion]) -> np.ndarray` — stack coeffs into an `(len(traj), 8)` array.
- `persistence_summary(traj: list[Octonion]) -> dict` — `{max_h1, total_h1, n_h1, total_h0}` floats.

`octonion_kernel/topology_controls.py` (harness; ripser + sklearn/scipy allowed):
- `run_map_trajectory(x0: Octonion, step_fn, lam=0.5, steps=256) -> list[Octonion]` — trajectory
  (renorm per step, underflow halt — same iteration contract as Phase-2 `run_map`).
- `iid_cloud(rng, n_points: int) -> list[Octonion]` — `n_points` iid-uniform unit octonions.
- `_bootstrap_mean_diff_ci(a, b, n_boot=2000, seed=0) -> tuple[float,float]` — 95% CI of
  `mean(a) − mean(b)` over shared (paired) resamples of the index set.
- `run_topology_control(n=200, steps=256, lam=0.5, seed=0) -> dict` — returns
  `{"max_h1": {map: mean}, "context": {map: {total_h1,n_h1,total_h0}}, "iid_cloud_max_h1": float,
  "verdict": {octonion_adds_topology: bool, best_baseline: str, octonion_max_h1: float,
  best_baseline_max_h1: float, diff_ci: [lo,hi]}}`.

`harness_report.py`:
- `report_e(...)` — prints the per-map mean `max_h1` table (linear, generic_nonlinear, random_walk,
  iid_cloud, octonion) and the verdict; appended after `[D]`.

## 7. Testing approach
- `pytest`, seeded numpy RNG (no `hypothesis`).
- `tests/test_topology.py` — `persistence_summary` correctness on known shapes: a sampled **circle**
  yields a clearly positive `max_h1`; a **tight cluster** yields `max_h1 ≈ 0`; `trajectory_cloud`
  shape/values; determinism.
- `tests/test_topology_controls.py` — `run_map_trajectory` preserves unit norm and is deterministic;
  `iid_cloud` returns unit octonions; `_bootstrap_mean_diff_ci` brackets a known difference; the
  agnostic `run_topology_control` check (verdict produced, ranges well-formed, **no YES/NO
  assertion**). The full control run is marked `@pytest.mark.slow`.
- Full suite green, output pristine.

## 8. Definition of done (Phase 3)
- `topology.py` computes `persistence_summary` via ripser; correctness tests pass (circle vs cluster).
- `topology_controls.py` implements the trajectory runner, the iid null, the paired-bootstrap, and
  `run_topology_control`; the control test passes and is agnostic to the verdict value.
- `harness_report.py` prints the `[E]` section with the max-H1 table and a recorded verdict (YES or
  NO — whichever it is).
- Full `python -m pytest -q` green (plus the pre-existing legacy skip); `python harness_report.py`
  produces A/B/C/D/E.
- Engine purity intact: `dynamics.py` and the kernel still import no scipy/sklearn/ripser/aoi
  (AST-verified).

## 9. Deferred to later phases
The 24D/96D embedding, domain adapters (finance / bio), and the product surface. Each gets its own
spec, plan, and per-layer control gate.

## 10. Open questions
None blocking. The walk length (`steps = 256`), the verdict metric (`max_h1`), and the generator
`g` / scale-matching are the fairness-critical declared constants, all pinned in code and reported.
If the 257-point cloud proves too sparse for stable H1 (max-H1 near zero for *all* maps, making the
comparison underpowered — the Phase-2 lesson), the implementation calibrates by increasing `steps`
(declared), never to change the verdict, only to give the test power; this is reported.
