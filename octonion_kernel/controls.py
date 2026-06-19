"""Control harness C — the anti-castle test.

Question: does the Jordan-Shadow associator carry information BEYOND trivial
statistics? We build a 'structured' class (pairs that share a fixed direction)
and a 'random' class (independent pairs), UNIT-NORMALIZE every octonion so all
magnitude information is destroyed (||product|| becomes constant, its AUC ~ 0.5),
then ask whether a fixed, pre-declared set of associator summaries can separate
the classes BETTER THAN the best trivial baseline.

The baselines are deliberately strong and include a purely Euclidean statistic
that uses no octonion structure at all: |a.b| (the dot product / cosine angle).
The associator counts as informative only if it beats the best of ALL declared
non-associator baselines, with the paired separation difference's bootstrap CI
excluding zero. Comparing only against magnitude would be a walkover, because
unit-normalization is designed to defeat magnitude. A 'NO' verdict is a valid,
expected result.

Pure compute: returns dicts, no I/O. Uses scipy only for tie-correct ranking.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from .octonion import Octonion
from .shadow import shadow_decompose

# fixed correlated direction shared by the structured class
_V0 = np.zeros(8)
_V0[1] = 1.0


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Directed AUC = P(score | label==1 > score | label==0), tie-corrected."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n1 = int(labels.sum())
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def bootstrap_sep_ci(scores, labels, n_boot: int = 2000, seed: int = 0):
    """95% bootstrap CI of separation = max(auc, 1 - auc)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    rng = np.random.default_rng(seed)
    n = len(scores)
    seps = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        a = auc(scores[idx], labels[idx])
        seps[i] = max(a, 1.0 - a)
    lo, hi = np.percentile(seps, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_diff_ci(scores_a, scores_b, labels, n_boot: int = 2000, seed: int = 0):
    """95% bootstrap CI of sep(scores_a) - sep(scores_b) on shared (paired) resamples.

    Positive throughout => scores_a separates better than scores_b. Used to test
    whether the associator's separation reliably exceeds the best trivial baseline's.
    """
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    labels = np.asarray(labels, dtype=int)
    rng = np.random.default_rng(seed)
    n = len(labels)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        lab = labels[idx]
        aa = auc(scores_a[idx], lab)
        ab = auc(scores_b[idx], lab)
        diffs[i] = max(aa, 1.0 - aa) - max(ab, 1.0 - ab)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def structured_pair(rng, eps: float = 0.1):
    """Pair (a, b) where b shares a's direction and a fixed direction _V0 (correlated)."""
    a = rng.standard_normal(8)
    ca, cv = rng.standard_normal(), rng.standard_normal()
    b = ca * a + cv * _V0 + eps * rng.standard_normal(8)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return Octonion(a), Octonion(b)


def random_pair(rng):
    """Independent pair, unit-normalized (magnitude-matched to the structured class)."""
    a = rng.standard_normal(8)
    b = rng.standard_normal(8)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return Octonion(a), Octonion(b)


def _pair_summaries(a: Octonion, b: Octonion) -> dict:
    """All declared scalar summaries for one (a, b) pair, grouped by what they need:

      magnitude   : ||product||                          (killed by unit-normalization)
      trivial     : |a.b| (pure Euclidean, no octonion structure),
                    ||jordan||, ||commutator||            (component norms)
      associator  : ||associator||, associator.real, max|associator component|
    """
    r = shadow_decompose(a, b)
    return {
        "prod_norm": r.product.norm(),
        "dot_abs": abs(float(a.coeffs @ b.coeffs)),
        "jordan_norm": r.jordan.norm(),
        "commutator_norm": r.commutator.norm(),
        "assoc_norm": r.associator.norm(),
        "assoc_real": r.associator.real,
        "assoc_maxabs": float(np.max(np.abs(r.associator.coeffs))),
    }


# non-associator baselines the associator must beat (magnitude + trivial + components)
_BASELINE_KEYS = ("prod_norm", "dot_abs", "jordan_norm", "commutator_norm")
# the associator summaries under test (fixed, pre-declared — no open search)
_ASSOC_KEYS = ("assoc_norm", "assoc_real", "assoc_maxabs")
_ALL_KEYS = _BASELINE_KEYS + _ASSOC_KEYS


def run_control_c(n: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    cols = {k: [] for k in _ALL_KEYS}
    labels = []
    for label, make in ((1, structured_pair), (0, random_pair)):
        for _ in range(n):
            a, b = make(rng)
            s = _pair_summaries(a, b)
            for k in _ALL_KEYS:
                cols[k].append(s[k])
            labels.append(label)
    labels = np.array(labels)
    scores = {k: np.array(cols[k]) for k in _ALL_KEYS}

    out = {}
    for k in _ALL_KEYS:
        a = auc(scores[k], labels)
        lo, hi = bootstrap_sep_ci(scores[k], labels, seed=seed + 1)
        out[k] = {"auc": a, "sep": max(a, 1.0 - a), "ci_lo": lo, "ci_hi": hi}

    best_assoc = max(_ASSOC_KEYS, key=lambda k: out[k]["sep"])
    best_baseline = max(_BASELINE_KEYS, key=lambda k: out[k]["sep"])
    # paired bootstrap of sep(best associator) - sep(best baseline): the associator
    # adds information only if this difference is reliably positive (CI excludes 0)
    # AND the associator clears chance.
    diff_lo, diff_hi = bootstrap_diff_ci(scores[best_assoc], scores[best_baseline],
                                         labels, seed=seed + 2)
    adds_info = bool(out[best_assoc]["ci_lo"] > 0.5 and diff_lo > 0.0)
    out["verdict"] = {
        "associator_adds_information": adds_info,
        "best_associator_summary": best_assoc,
        "best_baseline_summary": best_baseline,
        "assoc_sep": out[best_assoc]["sep"],
        "best_baseline_sep": out[best_baseline]["sep"],
        "sep_difference_ci": [diff_lo, diff_hi],
        "magnitude_sep": out["prod_norm"]["sep"],
    }
    return out
