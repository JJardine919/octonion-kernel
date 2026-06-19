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
