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

**Phase-2 result:** NO — the octonion walk does not beat the best baseline (octonion AUC 0.549 vs generic_nonlinear 0.621). A NO is a valid, expected outcome by design.
