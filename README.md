# octonion-kernel

A small, standalone, provably-correct library for 8-dimensional octonion algebra
and the Jordan-Shadow decomposition (DOI 18690444), with an A/B/C control harness.

This is **Phase 1** of a larger build. It ships only the math it can prove, and an
honest measurement (control C) of whether the Jordan-Shadow associator carries
information beyond *trivial* statistics. Control C compares the associator not just
against the magnitude `||product||` but against trivial non-octonion baselines —
chiefly the plain dot product `|a·b|` — and the Jordan/commutator component norms;
the associator counts as informative only if it beats the best of these (paired
bootstrap CI of the separation difference excluding zero). **Phase-1 result: it does
not.** Its apparent separation only tracks the `a·b` angle, which `|a·b|` captures as
well or better — and `||associator||` is exactly `||jordan||·||commutator||`, so it
adds nothing over its components. A NO here is a valid, expected outcome by design.
The kernel makes no claim about what the decomposition *means* — only what it
provably *is*. The 24D/96D embeddings, the multi-layer dynamics, persistent homology,
and domain adapters are later phases.

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

## Phase 2 — dynamics

`octonion_kernel/dynamics.py` iterates the kernel into an autonomous, unit-sphere walk
`x_{t+1} = renorm(λ·x_t + (1−λ)·associator(x_t, g))` (genuinely nonlinear: the associator is
quadratic in `x`). Its control (`dynamics_controls.py`, run via the `[D]` section of
`harness_report.py`) asks whether iterating the walk makes structured-vs-random initial states
more linearly separable than the raw input and than matched **linear, generic-nonlinear, and
random-walk** baselines — the generic-nonlinear map being the decisive bar (the dynamics analog
of `|a·b|`). The octonion walk counts only if it beats the best baseline with a paired-bootstrap
CI excluding zero.

**Phase-2 result:** NO — the octonion walk does not beat the best baseline (octonion AUC 0.685 vs raw 0.880). The mean/location structured class now gives the raw input real separability power (raw AUC 0.880); the walk erodes rather than grows that separation. A NO is a valid, expected outcome by design.

## Phase 3 — topology

`octonion_kernel/topology.py` computes a persistent-homology summary (via `ripser`, Euclidean
distance) of a walk's trajectory; the verdict metric is **max-H1 persistence** (the longest loop;
`total_h1` is reported but not gated on, because it rewards noise loops). The control
(`topology_controls.py`, `[E]` section of `harness_report.py`) asks whether the octonion walk's
256-step trajectory carries more loop structure than matched **linear, generic-nonlinear, and
random-walk (diffusion)** trajectories — the random-walk being the dynamical noise null. The
octonion walk counts only if its mean max-H1 beats the best baseline with a paired-bootstrap CI
excluding zero. An `iid_cloud` scatter is reported as a sanity null but does not gate.

**Phase-3 result:** NO — the octonion walk's mean max-H1 (0.4530) is below the random-walk
diffusion null (0.5338) and far below the linear baseline (1.5115); the paired-bootstrap CI
[-1.145, -0.969] excludes zero. Note: raw max-H1 is an unnormalized absolute lifetime and scales
with trajectory diameter — the octonion walk contracts toward an attractor (mean diameter 1.5355
vs linear 2.0000, random-walk 1.9657), so part of the raw gap could reflect contraction rather
than topological poverty. The scale-invariant normalized metric (max-H1/diameter) confirms the NO
is unconfounded: octonion 0.2394 remains below the random-walk null (0.2715) and far below linear
(0.7558). The NO is a valid, expected outcome by design.

## Phase 4 — optimizer

Phases 1-3 tested whether the Jordan-Shadow decomposition passively *carries information*.
Phase 4 asks a different question: does it work as a *useful search-guidance signal* inside a
real optimizer? `octonion_kernel/optimize.py` implements simulated annealing over a random
SK-model instance (`E(s) = -Σ J_ij s_i s_j`, the standard QUBO-equivalent spin-glass testbed) with
four interchangeable move-proposal strategies sharing one solver loop — random, greedy
local-field, a generic-nonlinear elementwise combination, and shadow-guided (per-chunk
`shadow_decompose` associator magnitude). The control (`optimize_controls.py`, `[F]` section of
`harness_report.py`) runs all four paired on identical instances and asks whether shadow reaches
reliably lower energy than the best of the other three, gated on a power check (greedy and
generic-nonlinear must each reliably beat random, or the run is reported inconclusive rather than
trusted as a NO).

**Two bugs found and fixed during Phase-4 development, by the power check doing its job:** (1)
hard-argmax move proposal deadlocks single-spin-flip SA — once the top-scoring spin's flip is
thermally rejected, the state hasn't changed, so the identical spin gets proposed again next step,
forever; fixed via score-weighted random sampling. (2) greedy/generic-nonlinear originally scored
by raw `|h_i|` (accidentally identical to `|state_i·h_i|`, since `|state_i|=1` always), which
conflates an already-settled spin with a genuinely frustrated one; fixed to score by actual
flip-improvement `max(0, -ΔE_i)`, the textbook greedy criterion. Both fixes were driven by the
baselines failing to beat random — exactly the "check the control has power before trusting a
verdict" discipline Phase 2 established.

**Phase-4 result:** NO — the shadow-guided proposal does not beat the best baseline (greedy local
field). Power check passes cleanly (greedy/generic-nonlinear reliably beat random). Engine 0-for-4
across all phases. Note: the spec-declared full-scale default (500 instances × 5000 steps) takes
roughly 10 hours at current performance — `propose_shadow`'s per-step `Octonion` construction
dominates the cost — so `harness_report.py`'s `[F]` section runs at a reduced demonstration scale
by default; invoke `report_f(n_instances=500, steps=5000)` directly for the full-power run.
