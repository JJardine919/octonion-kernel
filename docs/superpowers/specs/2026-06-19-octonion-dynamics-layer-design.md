# Octonion Dynamics Layer + Control — Design Spec

**Date:** 2026-06-19
**Author:** James Jardine (Lattice24 / 2396179 Alberta Inc.)
**Status:** Approved design — Phase 2 of the "octonion product" build.

---

## 1. Context & motivation

Phase 1 delivered a provably-correct 8-D octonion algebra kernel (`Octonion`, `multiply`,
`shadow_decompose`, `identity_residuals`) and an A/B/C control harness. Control C returned an
honest **NO**: the Jordan-Shadow associator carries no information beyond trivial statistics —
its apparent separation only tracks the `a·b` angle, and `‖associator‖` is exactly
`‖jordan‖·‖commutator‖`. That verdict is *static* — it concerns a single application of the
decomposition.

This phase, **dynamics**, asks a genuinely different question: when the kernel map is **iterated**,
does iteration create structure that a single application — and a matched trivial map — cannot?
Iteration of a nonlinear map can produce behaviour (sensitivity, manifold folding, increased
separability) absent from one step, so a static NO does not settle the dynamic question.

The governing principle from Phase 1 carries forward unchanged:

> **Every layer ships with its own control/baseline before anything is built on top of it.**
> The layer either provably does something beyond a trivial operation, or we learn that it
> doesn't — early, cheaply. A NO is a valid, expected outcome.

### Where this sits in the stack
1. Algebra kernel — *done (Phase 1)*
2. **Dynamics (this spec)** — the kernel map iterated, with feedback + decay, gated by a control
3. Topology (persistent homology over state history) — later
4. Application adapter (finance / bio) — later
5. Product surface — later

## 2. Goal & non-goals

**Goal:** a small, pure Python module implementing an autonomous, genuinely-nonlinear octonion
**walk** seeded by an initial state, plus a control harness that honestly tests whether iterating
the walk makes two declared classes of initial states **more linearly separable** than the raw
input and than matched **linear, generic-nonlinear, and random-walk** baselines.

**Non-goals (Phase 2):**
- No 24D/96D embedding; no domain data. The walk operates on octonion states directly; the
  control feeds synthetic states (Control-C style).
- No persistent homology, no domain logic — later layers.
- No input-driven / reservoir variant: the walk is **autonomous** (input sets the initial state
  only).
- No claim that the walk *means* anything; only what it provably *is* and whether it provably
  beats the baselines.

## 3. Scope boundary & dependency posture

- `octonion_kernel/dynamics.py` is a **pure** module: numpy + the Phase-1 kernel
  (`octonion`, `shadow`) only. No I/O, no global mutable state, no scipy, no sklearn, no
  `aoi_collapse`.
- `octonion_kernel/dynamics_controls.py` is **harness** code: it may import scipy and sklearn
  (the same boundary Phase 1 set — scipy/sklearn allowed in harness, never in the kernel or the
  pure engine). It is pure compute (returns dicts), no I/O.
- `harness_report.py` (repo root, the only file that prints) gains a `[D]` section.
- The dependency direction stays one-way: kernel → nothing; dynamics engine → kernel; dynamics
  control → dynamics engine + kernel; report → all.

## 4. The engine — the octonion walk

State is a **unit** octonion (renormalized every step) so that magnitude can never masquerade as
signal — the lesson Control C taught. The autonomous map, seeded by an initial unit octonion `x₀`:

```
x_{t+1} = renorm( λ · x_t  +  (1 − λ) · associator(x_t, g) )
```

- `associator(x_t, g)` = `jordan(x_t, g) · commutator(x_t, g)`, taken from
  `shadow_decompose(x_t, g)`. Each factor is linear in `x_t`, so the associator is **quadratic in
  `x_t`** → the map is genuinely nonlinear (a fixed-octonion multiply alone would be linear on ℝ⁸
  and reproducible by a linear baseline). This is why the associator, not a plain `x·x`, is the
  nonlinear core: it both guarantees nonlinearity and exercises the kernel's distinctive operation.
- `g` — a fixed, **declared** generator octonion (a specific unit octonion pinned in code, never
  tuned to a result).
- `λ ∈ [0, 1)` — decay/leak; the feedback term is `λ·x_t`. Default `0.5`.
- `renorm(y)` — `y / ‖y‖` (project to the unit sphere). If `‖y‖` underflows (degenerate step),
  the walk halts and returns the trajectory so far (documented; must not divide by zero).
- `T` — number of steps. Default `32`.

**Output:** the full trajectory `[x₀, x₁, …, x_T]` (a list of `Octonion`, so the later topology
layer can read state history) and, by convenience, the final state `x_T`. Deterministic given
`(x₀, g, λ, T)`.

### Engine correctness checks (the dynamics analog of Harness A)
- **Unit-norm preserved:** `‖x_t‖ = 1` (to ~1e-12) for every `t` in the trajectory.
- **Determinism:** identical `(x₀, g, λ, T)` → identical trajectory.
- **Genuine nonlinearity:** `associator(α·x, g) = α²·associator(x, g)` (degree-2 homogeneous),
  confirming the map is not linear. Verified over random `x`, `g`, scalars `α`.
- **Boundedness:** the trajectory never produces non-finite values (no blow-up), a direct
  consequence of per-step renormalization.

## 5. The control — built to say NO

### 5.1 The two classes (declared, not searched)
- **random:** `x₀` uniform on S⁷ (Gaussian 8-vector, normalized).
- **structured:** `x0` clustered near a **fixed declared direction** on the sphere:
  `normalize(c·μ + noise)` with `μ = ones(8)/√8` and concentration `c = 2.0` (both pinned
  in code, not tuned to a verdict). This is a **mean/location** offset — a linear signal the
  separability metric (§5.2) can detect — so the raw input is meaningfully separable and
  "separability growth over raw" is a test with real power. (The earlier variance-only
  "subspace" definition was invisible to the linear metric and is replaced.)

Equal numbers of each class, each labelled (structured = 1, random = 0).

### 5.2 Separability metric
Each `x₀` is evolved under a map to its **final state `x_T`** (8 features). On those 8-D
representations:
- Fit a plain `LogisticRegression` on a seeded train split; score the held-out test split; take
  the test **AUC** (`sklearn.metrics.roc_auc_score`). Higher AUC ⇒ the map made the two classes
  more linearly separable. The train/test split and all RNG are seeded.

### 5.3 The declared baselines — the octonion walk must beat all of them
Each baseline produces a final state the same way (same `λ`, `T`, renorm, same train/test split),
differing only in the per-step update term:
1. **raw** — no walk; classify on `x₀` directly. Establishes the starting separability; the
   octonion walk must *exceed* this to claim separability **growth**.
2. **linear walk** — `renorm(λ·x + (1−λ)·A·x)`, `A` a fixed (seeded) random linear map on ℝ⁸,
   **scale-matched** to the associator: rescale `A` so that the mean update magnitude over random
   unit octonions, `E[‖A·x‖]`, equals `E[‖associator(x, g)‖]` (expectations estimated on a fixed
   seeded sample of unit `x`). "What a linear dynamical map achieves."
3. **generic-nonlinear walk** — `renorm(λ·x + (1−λ)·Q(x))`, `Q` a generic random **quadratic** map
   (seeded random tensor, `Q(x)_i = xᵀ T_i x`), scale-matched the same way (`E[‖Q(x)‖] =
   E[‖associator(x, g)‖]` over the same seeded sample). **The decisive bar — the dynamics analog of
   `|a·b|`:** if a generic quadratic nonlinearity separates as well, the octonion/associator
   structure adds nothing; only "having a nonlinearity" did.
4. **random walk** — `renorm(λ·x + (1−λ)·η)`, fresh seeded noise `η` each step. Pure diffusion;
   does any structure survive noise.

### 5.4 Verdict rule
Compute test-AUC for `{raw, linear, generic_nonlinear, random, octonion}`. The octonion walk
**adds dynamical structure** iff:
- `AUC(octonion) > AUC(raw)` (separability grew), **and**
- the paired-bootstrap 95% CI of `AUC(octonion) − AUC(best_baseline)` excludes 0 on the positive
  side, where `best_baseline = argmax over {raw, linear, generic_nonlinear, random}`.

The paired bootstrap resamples the **test set** with shared indices and recomputes the AUC
difference (the same paired-difference method Phase 1's strengthened Control C used). Anything
else → **NO**. The verdict dict reports every AUC, the chosen best baseline, the difference CI,
and the boolean.

### 5.5 The control test (anti-castle discipline)
The pytest test asserts only: the harness runs and returns a verdict; the boolean is a bool; the
chosen best baseline is one of the four declared baselines; states remain unit-norm through the
walk; every reported AUC is in `[0, 1]`. **It does not assert the octonion walk wins or loses.**
A NO is a valid outcome; forcing a YES would rebuild the castle this phase exists to prevent.

## 6. Components & interfaces

`octonion_kernel/dynamics.py` (pure):
- `octonion_walk(x0: Octonion, g: Octonion, lam: float = 0.5, steps: int = 32) -> list[Octonion]`
  — returns the trajectory `[x0, …, x_T]`.
- `associator_step(x: Octonion, g: Octonion) -> Octonion` — the nonlinear term
  `jordan(x,g)·commutator(x,g)` (thin wrapper over `shadow_decompose`).
- `DEFAULT_GENERATOR: Octonion` — the pinned `g`.
- `renorm(x: Octonion) -> Octonion` — unit-sphere projection (raises/halts handled by the walk).

`octonion_kernel/dynamics_controls.py` (harness; scipy + sklearn allowed):
- the four baseline step-maps (linear, generic-nonlinear, random) and a generic
  `run_map(x0, step_fn, lam, steps) -> Octonion` returning the final state.
- `make_structured(rng) / make_random(rng) -> Octonion` initial-state generators.
- `run_dynamics_control(n=..., steps=..., seed=...) -> dict` — runs the experiment and returns the
  per-map AUCs, the verdict block (`octonion_adds_structure: bool`, `best_baseline`, AUC values,
  `auc_difference_ci`) via a paired bootstrap over the test set (the same paired-difference method
  Phase-1's strengthened Control C used, implemented here for the sklearn classification AUC).

`harness_report.py`:
- `report_d(...)` — prints the AUC table (raw, linear, generic_nonlinear, random, octonion) and
  the verdict, appended after `[C]`.

## 7. Testing approach
- `pytest`, seeded numpy RNG (no `hypothesis`).
- `tests/test_dynamics.py` — engine checks from §4 (unit-norm preserved, determinism, degree-2
  homogeneity / nonlinearity, finiteness) and control checks from §5.5 (verdict produced, agnostic
  to YES/NO, baselines present, AUCs in range, states unit-norm).
- The dynamics control runs at a modest `n`/`steps` in the test (seconds); the full-resolution run
  lives in `harness_report.py`. If the control test is slow, mark it `@pytest.mark.slow` (the
  marker is already registered in `pytest.ini`).
- Full suite green, output pristine.

## 8. Definition of done (Phase 2)
- `dynamics.py` implements `octonion_walk` + helpers with no scipy/sklearn/aoi imports; engine
  checks pass (unit-norm, determinism, degree-2 nonlinearity, finiteness).
- `dynamics_controls.py` implements the four baselines, the experiment, and the paired-bootstrap
  verdict; the control test passes and is agnostic to the verdict value.
- `harness_report.py` prints the `[D]` section with the AUC table and a recorded verdict (YES or
  NO — whichever it is).
- Full `python -m pytest -q` green (plus the pre-existing legacy skip); `python harness_report.py`
  produces A/B/C/D.

## 9. Deferred to later phases
Persistent homology over the trajectory (with a real declared distance metric and its own
shuffled-input null), the 24D/96D embedding, domain adapters, and the product surface. Each gets
its own spec, plan, and per-layer control gate.

## 10. Open questions
None blocking. The structured-class generator (§5.1) and the generator `g` (§4) are the
fairness-critical declared constants; both are pinned in code and reported. Additional structured
classes or generators may be explored later but are not required for Phase-2 done.
