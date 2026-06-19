"""Control harness for the octonion dynamics layer (Phase 2).

Harness code: numpy + scipy + sklearn allowed (never imported by the pure
dynamics engine or the kernel). Pure compute: returns dicts, no I/O.

Builds matched baselines (linear, generic-nonlinear, random walk) and asks
whether iterating the octonion walk makes structured-vs-random initial states
more linearly separable than the raw input and than every baseline.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .octonion import Octonion
from .dynamics import associator_step, renorm, DEFAULT_GENERATOR

# declared structured-class subspace dimension (a "great subsphere")
_STRUCTURED_DIM = 3


def make_random(rng) -> Octonion:
    """Uniform on S^7."""
    v = rng.standard_normal(8)
    return Octonion(v / np.linalg.norm(v))


def make_structured(rng) -> Octonion:
    """Unit octonion supported only on the first _STRUCTURED_DIM basis axes."""
    v = np.zeros(8)
    v[:_STRUCTURED_DIM] = rng.standard_normal(_STRUCTURED_DIM)
    return Octonion(v / np.linalg.norm(v))


def _mean_assoc_norm(g: Octonion, seed: int = 12345, n: int = 2000) -> float:
    """E[||associator(x, g)||] over random unit octonions x (for scale matching)."""
    rng = np.random.default_rng(seed)
    tot = 0.0
    for _ in range(n):
        tot += associator_step(make_random(rng), g).norm()
    return tot / n


def _mean_step_norm(step_fn, seed: int = 999, n: int = 2000) -> float:
    """E[||step_fn(x)||] over random unit octonions x."""
    rng = np.random.default_rng(seed)
    tot = 0.0
    for _ in range(n):
        tot += step_fn(make_random(rng)).norm()
    return tot / n


def make_linear_map(g: Octonion, seed: int = 0):
    """Scale-matched random linear step term x -> A x (A rescaled so
    E[||A x||] == E[||associator(x, g)||])."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((8, 8))

    def raw_step(x: Octonion) -> Octonion:
        return Octonion(A @ x.coeffs)

    scale = _mean_assoc_norm(g) / _mean_step_norm(raw_step)
    A = A * scale

    def step(x: Octonion) -> Octonion:
        return Octonion(A @ x.coeffs)

    return step


def make_generic_nonlinear_map(g: Octonion, seed: int = 0):
    """Scale-matched generic random quadratic step term Q(x)_i = x^T T_i x
    (rescaled so E[||Q(x)||] == E[||associator(x, g)||])."""
    rng = np.random.default_rng(seed)
    T = rng.standard_normal((8, 8, 8))  # T[i] is the matrix for output coord i

    def raw_step(x: Octonion) -> Octonion:
        xc = x.coeffs
        return Octonion(np.einsum("ijk,j,k->i", T, xc, xc))

    scale = _mean_assoc_norm(g) / _mean_step_norm(raw_step)

    def step(x: Octonion) -> Octonion:
        xc = x.coeffs
        return Octonion(scale * np.einsum("ijk,j,k->i", T, xc, xc))

    return step


def make_random_walk_step(g: Octonion, seed: int = 0):
    """Fresh scale-matched noise each call (pure diffusion baseline). The emitted
    step term has norm == E[||associator(x, g)||] and ignores x."""
    target = _mean_assoc_norm(g)
    rng = np.random.default_rng(seed)

    def step(x: Octonion) -> Octonion:
        eta = rng.standard_normal(8)
        return Octonion(eta / np.linalg.norm(eta) * target)

    return step


def octonion_step(g: Octonion):
    """The octonion associator step term as a step_fn (uniform interface)."""
    def step(x: Octonion) -> Octonion:
        return associator_step(x, g)
    return step


def run_map(x0: Octonion, step_fn, lam: float = 0.5, steps: int = 32) -> Octonion:
    """Iterate renorm(lam*x + (1-lam)*step_fn(x)) from x0; return the final state.
    Halts (returns current state) if a step underflows (||y|| < 1e-12)."""
    x = renorm(x0)
    for _ in range(steps):
        y = Octonion(lam * x.coeffs + (1.0 - lam) * step_fn(x).coeffs)
        if y.norm() < 1e-12:
            break
        x = renorm(y)
    return x


def _final_features(step_or_none, x0_list, lam, steps):
    """8-vector final state for each x0 under a step map (raw = no walk)."""
    feats = []
    for x0 in x0_list:
        xf = renorm(x0) if step_or_none is None else run_map(x0, step_or_none, lam, steps)
        feats.append(xf.coeffs)
    return np.asarray(feats)


def _auc_on_split(X, y, train_idx, test_idx):
    """Fit logistic regression on the train split, return (test AUC, test scores)."""
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X[train_idx], y[train_idx])
    scores = clf.decision_function(X[test_idx])
    return float(roc_auc_score(y[test_idx], scores)), scores


def _bootstrap_auc_diff_ci(scores_a, scores_b, y, n_boot: int = 2000, seed: int = 0):
    """95% CI of AUC(scores_a) - AUC(scores_b) on shared (paired) test-set resamples."""
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    y = np.asarray(y, dtype=int)
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ys = y[idx]
        if ys.min() == ys.max():
            continue  # degenerate resample (single class) — skip
        diffs.append(roc_auc_score(ys, scores_a[idx]) - roc_auc_score(ys, scores_b[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def run_dynamics_control(n: int = 400, lam: float = 0.5, steps: int = 32, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    g = DEFAULT_GENERATOR

    x0s, labels = [], []
    for _ in range(n):
        x0s.append(make_structured(rng))
        labels.append(1)
    for _ in range(n):
        x0s.append(make_random(rng))
        labels.append(0)
    labels = np.asarray(labels)

    maps = {
        "raw": None,
        "linear": make_linear_map(g, seed=seed + 1),
        "generic_nonlinear": make_generic_nonlinear_map(g, seed=seed + 2),
        "random_walk": make_random_walk_step(g, seed=seed + 3),
        "octonion": octonion_step(g),
    }
    feats = {k: _final_features(m, x0s, lam, steps) for k, m in maps.items()}

    # fixed seeded 60/40 train/test split
    idx = np.arange(2 * n)
    np.random.default_rng(seed + 100).shuffle(idx)
    cut = int(0.6 * len(idx))
    train_idx, test_idx = idx[:cut], idx[cut:]
    y_test = labels[test_idx]

    aucs, scores = {}, {}
    for k in maps:
        aucs[k], scores[k] = _auc_on_split(feats[k], labels, train_idx, test_idx)

    baseline_keys = ("raw", "linear", "generic_nonlinear", "random_walk")
    best_baseline = max(baseline_keys, key=lambda k: aucs[k])
    diff_lo, diff_hi = _bootstrap_auc_diff_ci(
        scores["octonion"], scores[best_baseline], y_test, seed=seed + 200)
    adds = bool(aucs["octonion"] > aucs["raw"] and diff_lo > 0.0)

    return {
        "aucs": aucs,
        "verdict": {
            "octonion_adds_structure": adds,
            "best_baseline": best_baseline,
            "octonion_auc": aucs["octonion"],
            "best_baseline_auc": aucs[best_baseline],
            "auc_difference_ci": [diff_lo, diff_hi],
        },
    }
