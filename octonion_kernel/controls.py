"""Control harness C — the anti-castle test.

Question: does the Jordan-Shadow associator carry information BEYOND a trivial
magnitude statistic? We build a 'structured' class (pairs that share a fixed
direction) and a 'random' class (independent pairs), UNIT-NORMALIZE every
octonion so all magnitude information is destroyed (||product|| becomes constant,
its AUC ~ 0.5), then ask whether a fixed, pre-declared set of associator summaries
can still separate the classes. No open search over summaries (that would
cherry-pick toward the associator). A 'no' verdict is a valid result.

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


# fixed, pre-declared associator summaries (no open search)
def _summaries(result):
    assoc = result.associator
    return {
        "prod_norm": result.product.norm(),
        "assoc_norm": assoc.norm(),
        "assoc_real": assoc.real,
        "assoc_maxabs": float(np.max(np.abs(assoc.coeffs))),
    }


_ASSOC_KEYS = ("assoc_norm", "assoc_real", "assoc_maxabs")
_ALL_KEYS = ("prod_norm",) + _ASSOC_KEYS


def run_control_c(n: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    cols = {k: [] for k in _ALL_KEYS}
    labels = []
    for label, make in ((1, structured_pair), (0, random_pair)):
        for _ in range(n):
            a, b = make(rng)
            s = _summaries(shadow_decompose(a, b))
            for k in _ALL_KEYS:
                cols[k].append(s[k])
            labels.append(label)
    labels = np.array(labels)

    out = {}
    for k in _ALL_KEYS:
        scores = np.array(cols[k])
        a = auc(scores, labels)
        lo, hi = bootstrap_sep_ci(scores, labels, seed=seed + 1)
        out[k] = {"auc": a, "sep": max(a, 1.0 - a), "ci_lo": lo, "ci_hi": hi}

    best_key = max(_ASSOC_KEYS, key=lambda k: out[k]["sep"])
    magnitude_sep = out["prod_norm"]["sep"]
    # associator carries info beyond magnitude iff its best summary's CI lower
    # bound clears chance AND clears the magnitude baseline's separation.
    beats = bool(out[best_key]["ci_lo"] > 0.5 and out[best_key]["ci_lo"] > magnitude_sep)
    out["verdict"] = {
        "associator_beats_magnitude": beats,
        "best_summary": best_key,
        "magnitude_sep": magnitude_sep,
    }
    return out
