# Octonion Optimizer Layer + Control — Design Spec

**Date:** 2026-07-17
**Author:** James Jardine (Lattice24 / 2396179 Alberta Inc.)
**Status:** Approved design — Phase 4 of the "octonion product" build.

---

## 1. Context & motivation

Phases 1-3 tested whether the Jordan-Shadow decomposition **passively carries information**:
does the associator separate structured-vs-random pairs beyond trivial statistics (Phase 1,
static, **NO**); does iterating the kernel map make initial states more separable than matched
baselines (Phase 2, dynamic, **NO**); does the walk's trajectory carry more topological structure
than a random-walk null (Phase 3, topological, **NO**). Engine is 0-for-3. The one thing that
survived all three phases is the *methodology* — pre-declared, matched-information, honest
adversarial controls that are willing to return NO.

This phase asks a **genuinely different kind of question**, not yet tested by Phases 1-3: does
the decomposition function as a **useful active search-guidance signal** inside a real optimizer,
rather than a passive descriptive statistic? Concretely: seeded with the *same* local information
a generic heuristic gets, does routing that information through the Jordan-Shadow decomposition
produce a move-proposal rule that finds better solutions, on a real combinatorial optimization
benchmark, than a matched-information non-octonion rule?

This is deliberately the literal claim under test — it is the same shape of claim a Base44-hosted
chat fallback fabricated confidently, with invented numbers and no computation behind it, on
2026-07-17 ("aligning the Jordan component with the steepest descent vector while using the
Associator component to maintain topological constraints"). This phase builds the real thing and
tests it honestly instead.

The governing principle from Phase 1 carries forward unchanged:

> **Every layer ships with its own control/baseline before anything is built on top of it.**
> The layer either provably does something beyond a trivial operation, or we learn that it
> doesn't — early, cheaply. A NO is a valid, expected outcome.

### Where this sits in the stack
1. Algebra kernel — *done (Phase 1, NO)*
2. Dynamics (autonomous walk, linear separability) — *done (Phase 2, NO)*
3. Topology (persistent homology over walk trajectory) — *done (Phase 3, NO)*
4. **Optimizer layer (this spec)** — shadow decomposition as move-proposal guidance in a real
   solver, gated by a control
5. Application adapter / product surface — later

## 2. Goal & non-goals

**Goal:** a small, pure Python module implementing simulated annealing over a random SK-model
(QUBO-equivalent) instance, with four interchangeable move-proposal strategies — random, greedy
local-field, generic-nonlinear, and shadow-guided — sharing an identical solver loop, plus a
control harness that honestly tests whether the shadow-guided strategy reaches lower energy than
the best of the other three, matched for information and compute budget.

**Non-goals (Phase 4):**
- No D-Wave / real QUBO hardware, no network, no AOI-engine imports — same boundary as every
  prior phase.
- No hybrid shadow+greedy proposal rule. The shadow arm is pure-associator, declared in advance,
  to keep the comparison to a single pre-declared rule (no post-hoc tuning toward a result).
- No claim about *why* a result holds — only whether it does, under matched information and
  matched compute.
- No attempt to make problem sizes realistic/large. Small enough to run thousands of instances in
  plain numpy; this is a controlled experiment, not a performance benchmark.

## 3. Scope boundary & dependency posture

- `octonion_kernel/optimize.py` is a **pure** module: numpy + the Phase-1 kernel (`octonion`,
  `shadow`) only. No I/O, no global mutable state, no scipy, no `aoi_collapse`.
- `octonion_kernel/optimize_controls.py` is **harness** code: may use the same paired-bootstrap
  methodology as `controls.py`/`dynamics_controls.py`. Pure compute (returns dicts), no I/O.
- `harness_report.py` gains an `[F]` section.
- Dependency direction unchanged: kernel → nothing; optimize engine → kernel; optimize control →
  optimize engine + kernel; report → all.

## 4. The engine — SK instance, energy, and the shared solver loop

**Instance:** `n = 64` spins (8 chunks of 8), coupling matrix `J` symmetric with zero diagonal,
`J_ij ~ N(0, 1/n)` for `i < j` (standard SK scaling), seeded. Energy `E(s) = -Σ_{i<j} J_ij s_i s_j`
for `s ∈ {-1, +1}^n`.

**Local field:** `h_i = Σ_j J_ij s_j` (the raw information every non-random arm receives; `ΔE` for
flipping spin `i` is `2 s_i h_i`).

**Solver:** standard single-spin-flip simulated annealing. One shared loop, parameterized only by
a `propose(state, J) -> int` function (the spin index to try flipping next):
1. Compute `ΔE` for the proposed flip.
2. Metropolis accept: always if `ΔE ≤ 0`, else with probability `exp(-ΔE / T)`.
3. Update `T` on a fixed geometric cooling schedule, `T_0 → T_min` over `steps` iterations.
4. Track best-seen energy.

Identical for all four arms: instance, initial spin configuration, cooling schedule, step budget
(`steps = 5000`), and acceptance rule. **Only `propose` differs.**

### The four proposal strategies
Chunk `c`'s current spins `a_c ∈ {-1,+1}^8` and local fields `b_c ∈ ℝ^8` are the same 16 numbers
available to every non-random arm.

1. **random** — uniform random spin index. The null.
2. **greedy** — `argmax_i |h_i|` over all `n` spins directly. No chunking, no octonion structure.
   The "trivial" baseline (Phase 1's `|a·b|` analog) — a real, legitimate heuristic, establishing
   that the setup has power to beat random at all.
3. **generic_nonlinear** — per chunk, fixed pre-declared elementwise combination
   `score_i = a_{c,i} · b_{c,i}`; propose `argmax` over all `n` spins. Same 16-numbers-per-chunk
   input as the shadow arm, same window width, no octonion algebra. The decisive bar.
4. **shadow** — per chunk, `shadow_decompose(Octonion(a_c), Octonion(b_c)).associator`; score
   `score_i = |associator_i|`; propose `argmax` over all `n` spins. The claim under test.

All four scoring functions are **pre-declared, fixed, no open search** — same discipline as
`controls.py`'s `_ALL_KEYS`.

### Engine correctness checks
- Energy function matches brute-force enumeration on `n = 8` (all 256 configurations).
- Each `propose` function returns a valid spin index in `[0, n)` for arbitrary seeded state.
- The chunk→global-spin index mapping is bijective and covers all `n` spins exactly once.
- SA loop determinism: identical `(instance, initial state, seed, propose)` → identical trajectory.
- Metropolis acceptance rate matches the closed-form expectation on a tiny fixed instance (sanity,
  not gated).

## 5. The control — built to say NO

### 5.1 Pairing
For each of `n_instances = 500` seeded random SK instances, generate one initial spin
configuration and run all four arms from the *same* instance and *same* initial state, with the
*same* cooling schedule and step budget. Only the proposal rule differs — this is what makes the
comparison paired and fair.

### 5.2 Verdict metric
Best-seen energy per arm per instance (lower is better). Paired bootstrap CI (resampling
instances, 2000 resamples, same method as `controls.py::bootstrap_diff_ci`, adapted to a mean
energy difference instead of an AUC) of `mean(best_baseline_energy) − mean(shadow_energy)` across
`{random, greedy, generic_nonlinear}` — positive and excluding zero means shadow wins.

### 5.3 Power check (Phase 2's lesson, applied up front)
Before trusting any NO: confirm `greedy` and `generic_nonlinear` each reliably beat `random` (CI of
their energy advantage over random excludes zero). If neither baseline beats random, the
experiment has no demonstrated power to separate any method from noise, and the shadow verdict is
reported as **inconclusive**, not NO.

### 5.4 Verdict rule
`shadow_finds_better_optima` is `True` iff the power check passes **and** the paired-bootstrap CI
of `(best_of_{random,greedy,generic_nonlinear} − shadow)` energy excludes zero on the favorable
(shadow-lower) side. Anything else → **NO** (or **inconclusive** if the power check fails). The
verdict dict reports every arm's mean best-energy, the chosen best baseline, the difference CI,
and the power-check booleans.

### 5.5 The control test (anti-castle discipline)
The pytest test asserts only: the harness runs and returns a verdict dict; the boolean fields are
bools; the chosen best baseline is one of the three declared baselines; every reported energy is
finite. **It does not assert the shadow arm wins or loses.** A NO is a valid outcome.

## 6. Components & interfaces

`octonion_kernel/optimize.py` (pure):
- `make_sk_instance(n: int = 64, seed: int = 0) -> np.ndarray` — symmetric `J`, zero diagonal.
- `energy(state: np.ndarray, J: np.ndarray) -> float`
- `local_fields(state: np.ndarray, J: np.ndarray) -> np.ndarray`
- `propose_random(state, J, rng) -> int`
- `propose_greedy(state, J, rng) -> int`
- `propose_generic_nonlinear(state, J, rng) -> int`
- `propose_shadow(state, J, rng) -> int` (uses `octonion_kernel.shadow.shadow_decompose`)
- `anneal(J, initial_state, propose_fn, steps: int = 5000, t0: float = 2.0, t_min: float = 0.05,
  seed: int = 0) -> dict` — returns `{"best_energy": float, "final_state": np.ndarray}`.

`octonion_kernel/optimize_controls.py` (harness):
- `run_optimize_control(n_instances: int = 500, n: int = 64, steps: int = 5000, seed: int = 0)
  -> dict` — runs all four arms paired over `n_instances`, returns per-arm mean best-energy,
  power-check results, the difference CI, and the verdict block.

`harness_report.py`:
- `report_f(...)` — prints the per-arm energy table, the power check, and the verdict, appended
  after `[E]`.

## 7. Testing approach

- `pytest`, seeded numpy RNG.
- `tests/test_optimize.py` — engine checks from §4 (brute-force energy match, valid proposal
  indices, bijective chunk mapping, determinism).
- `tests/test_optimize_controls.py` — control checks from §5.5, run at reduced `n_instances`/
  `steps` for speed (marked `@pytest.mark.slow` if needed, matching the existing marker in
  `pytest.ini`); the full-resolution run lives in `harness_report.py`.
- Full suite green, output pristine.

## 8. Definition of done (Phase 4)

- `optimize.py` implements the SK instance, energy, all four proposal strategies, and `anneal`;
  engine checks pass.
- `optimize_controls.py` implements the paired harness, the power check, and the verdict; the
  control test passes and is agnostic to the verdict value.
- `harness_report.py` prints the `[F]` section with the per-arm table and a recorded verdict
  (YES, NO, or inconclusive).
- Full `python -m pytest -q` green; `python harness_report.py` produces A/B/C/D/E/F.

## 9. Deferred to later phases

Larger/harder instance sizes, alternative cooling schedules, a hybrid shadow+greedy proposal rule,
real QUBO/D-Wave hardware comparison, the 24D/96D embedding, domain adapters, and the product
surface. Each gets its own spec, plan, and per-layer control gate.

## 10. Open questions

None blocking. The SK coupling scale, cooling schedule, step budget, and instance count (§4, §5.1)
are the fairness-critical declared constants; all are pinned in code and reported. A hybrid
shadow+greedy proposal rule was discussed and explicitly deferred (§9) to keep this phase's
comparison to a single pre-declared rule.
