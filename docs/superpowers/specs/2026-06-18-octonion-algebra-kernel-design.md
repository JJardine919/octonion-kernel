# Octonion Algebra Kernel + Control Harness — Design Spec

**Date:** 2026-06-18
**Author:** James Jardine (Lattice24 / 2396179 Alberta Inc.)
**Status:** Approved design — Phase 1 of the larger "octonion product" build.

---

## 1. Context & motivation

This is Phase 1 of building, as a product, the multi-layer system sketched in a
Voodoo↔Gemini design conversation: a Fano-octonion product → Jordan-Shadow filter →
multi-layer "walk" → persistent-homology layer → application adapters (finance/bio).

That conversation is treated as a **spec-in-disguise**, not a validated result. A prior
review found its code substantively hollow in the ways that have recurred across this
research program: the "Jordan-Shadow" in the chat was `np.argsort(np.abs(x))[-3:]` (a
magnitude threshold wearing a grand name), its "octonionic distance" was actually plain
Euclidean distance, and `betti_coefficients()` is not a real gudhi call. More broadly,
the underlying AOI collapse engine has **failed a noise control in every domain tested** —
it is magnitude-driven and structure-blind. (See the engine-noise-control record.)

The governing principle for this build, therefore:

> **Every layer ships with its own control/baseline before anything is built on top of
> it.** "Test at the end" is exactly how the hollow claims survived. We build so each
> layer either provably does something beyond a trivial operation, or we learn that it
> doesn't — early, cheaply, at the kernel level.

### The full stack (decomposition, for context)

Build order (dependency stack), each a separate spec → plan → implementation cycle:

1. **Algebra kernel** ← *this spec*
2. Dynamics (multi-layer walk: kernel applied iteratively with feedback + decay)
3. Topology (persistent homology over the state history)
4. Application adapter (maps a domain — finance *or* bio — onto the kernel)
5. Product surface (CLI / API / report)

Cross-cutting: the **validation harness** is not a final phase; each layer carries its own.

Chosen build strategy: **kernel + harness first (this spec), then a thin end-to-end
vertical slice on one use case, then deepen each layer with its gate.**

## 2. Goal & non-goals

**Goal:** a small, standalone, provably-correct Python library implementing 8-dimensional
octonion algebra and the Jordan-Shadow decomposition (DOI 18690444), plus a control
harness that (a) proves the algebra is correct and (b) tests honestly whether the
Jordan-Shadow's associator carries information beyond a trivial magnitude statistic.

**Non-goals (Phase 1):**
- No 24D/96D embeddings (how domain data is packed into octonionic structures is a later
  layer's concern).
- No multi-layer walk, no persistent homology, no domain logic (finance/bio).
- No *runtime* dependency on the existing `aoi_collapse.py` / AOI engine code: the kernel
  library must build, import, and pass A/B/C with the legacy code entirely absent. (An
  optional, auto-skipped *test* may cross-check against it — see §6 — but it is never a
  library dependency.)
- No claims about what the decomposition *means* — only what it provably *is*.

## 3. Scope boundary

The kernel is a **pure library**: no I/O, no global mutable state, no domain knowledge,
deterministic. Everything in later phases imports it; it imports nothing of theirs.

The honesty line sits *above* the kernel: the algebra (the product, the lossless J/C
split, the orthogonality and Pythagorean identities) is real, provable, and
engine-independent. The semantic labels from the source material — "Jordan = stable
classical metric, commutator = quantum phase, associator = personality" — are
interpretation that later layers and the validation harness must earn. The kernel asserts
none of that; it ships the math and the honest measurement (control C) of whether the
associator is more than magnitude.

## 4. Design decisions

- **Fresh reimplementation from the mathematics** (DOI 18690444), not an import of the
  existing code. Rationale: cleanest provenance, and an independent second implementation
  that can be diffed against the legacy `octonion_shadow_decompose` as a free cross-check.
- **Octonion product via Cayley–Dickson** doubling of the quaternions, using the standard
  convention from Baez, *The Octonions* (2002): for pairs `(a,b)(c,d) = (ac − d̄b, da + bc̄)`.
  The basis labels `e0..e7` and the resulting 7-line Fano orientation are **derived** from
  this construction and pinned in a test fixture (the full 8×8 = 64-entry basis product
  table), so the convention is unambiguous and the Fano table is verified, not hand-typed.
- **Identities returned as numbers, not asserted at call sites.** The decomposition
  function does the math; a separate `identity_residuals` returns the three identity errors
  so the harness can sweep thousands of random pairs and report distributions, rather than
  hard-failing on a single tolerance.

## 5. Components & interfaces

Each unit: one purpose, a clean interface, no global state.

1. **`Octonion`** — immutable value type holding 8 real coefficients `[e0..e7]`.
   - Construction from an 8-vector; `.real` (e0), `.vec` (e1..e7, length 7).
   - Operators: `+`, `-`, scalar `*`, octonion `*` (delegates to `multiply`), unary `-`.
   - Methods: `conjugate()`, `norm()` (Euclidean on the 8 coeffs), `approx_eq(other, tol)`.
   - No mutation; arithmetic returns new instances.

2. **`multiply(a: Octonion, b: Octonion) -> Octonion`** — the octonion product, Cayley–
   Dickson per the pinned convention. This is the single source of the algebra's structure.

3. **`shadow_decompose(a: Octonion, b: Octonion) -> ShadowResult`** — returns a small record:
   - `jordan = (a*b + b*a) / 2`
   - `commutator = (a*b - b*a) / 2`
   - `associator = jordan * commutator`
   - `product = a*b`
   (`ShadowResult` is a frozen dataclass / namedtuple of four `Octonion`s.)

4. **`identity_residuals(a, b) -> dict[str, float]`** — reports, as floats:
   - `losslessness = norm((jordan + commutator) - product)`  (expected ≈ 0, exact)
   - `orthogonality = |dot(jordan.vec, commutator.vec)|`       (expected ≈ 0)
   - `pythagorean = |norm(product.vec)**2 - (norm(jordan.vec)**2 + norm(commutator.vec)**2)|`

## 6. Control harness — the trust gate

Delivered as both a `pytest` suite and a runnable report script (`harness_report.py`).

**A. Algebra correctness** (deterministic; must pass):
- Norm multiplicativity: `‖ab‖ = ‖a‖·‖b‖` (composition-algebra property) over random pairs.
- Genuine non-associativity: the associator `(ab)c − a(bc)` is non-zero for generic
  triples (confirms these are octonions, not associative quaternions).
- Moufang identities hold (the weak associativity octonions *do* satisfy), e.g.
  `(a(b a))c = a(b(ac))` forms, over random triples.
- Unit/inverse/conjugate laws: `eᵢeᵢ = −1` for i≥1; `a·conj(a) = ‖a‖²`; `a·a⁻¹ = 1`.
- Basis-table cross-check: our `multiply` reproduces the pinned 64-entry reference
  multiplication table exactly.
- (Bonus) Legacy cross-check: results agree with `aoi_collapse.octonion_shadow_decompose`
  up to a documented basis isomorphism — run if the legacy module is importable, skipped
  otherwise (never a build dependency).

**B. Jordan-Shadow identities** (must pass): over N (≥10,000) random pairs, report the
max of each residual from `identity_residuals`. Losslessness must be ≤ 1e-10;
orthogonality and Pythagorean ≤ 1e-8.

**C. The anti-castle control** (reported, not assumed — the decisive one):
*Does the associator carry information beyond a trivial magnitude statistic?*
- Build two labelled classes of octonion pairs:
  - **structured:** pairs with a deliberate internal relationship — e.g. `b` shares a
    fixed 2-plane / correlated subspace with `a`.
  - **random (magnitude-matched):** Gaussian octonions whose per-coefficient variance and
    overall norm match the structured class.
- Compute the magnitude baseline `‖product‖` and a **fixed, pre-declared** set of three
  associator summaries — `‖associator‖`, `associator.real`, `max|associator component|` —
  no open search over summaries (an open search would cherry-pick and bias the verdict
  *toward* the associator). Report all four AUCs at separating structured from random.
- **Verdict rule:** take the best AUC among the three fixed associator summaries; if it does
  not exceed `‖product‖`'s AUC by a margin outside the bootstrap 95% CI, the associator adds
  nothing over magnitude at the kernel level — recorded prominently so later layers do not
  lean on it. A "no" here is a valid, useful outcome, not a failure of the build.

## 7. Testing approach

- `pytest`, with `hypothesis` property-based tests for the algebra laws (Sections A) over
  randomly generated octonions; fixed RNG seeds for reproducibility.
- Sections B and C implemented as both deterministic tests (with thresholds for A/B) and
  the `harness_report.py` script that prints the full A/B/C report including the control-C
  AUC table and verdict.
- Target: full suite runs in seconds on one core; no network, no external services.

## 8. Definition of done (Phase 1)

- `Octonion`, `multiply`, `shadow_decompose`, `identity_residuals` implemented in a fresh
  package with no AOI-engine imports.
- Harness A passes (algebra provably correct, including the 64-entry table cross-check).
- Harness B passes (identity residuals within tolerance over ≥10k pairs).
- Harness C runs and produces a recorded verdict (whatever it is) on whether the associator
  beats the magnitude baseline.
- `harness_report.py` produces the full report; `pytest` green.

## 9. Deferred to later phases

Dynamics (multi-layer walk + feedback/decay), persistent homology layer (with a real,
declared distance metric and its own shuffled-input null), the 24D/96D embedding, the
domain adapters (finance / bio), and the product surface. Each gets its own spec, plan,
and per-layer control gate, in the build order of §1.

## 10. Open questions

None blocking. The concrete "structured" generator for control C (the correlated-subspace
construction) is fixed above as the first instance; additional structured generators may be
added later but are not required for Phase 1 done.
