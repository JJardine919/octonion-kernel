"""Runnable A/B/C control report. Run from the repo root: python harness_report.py

A: algebra correctness  (delegates to pytest tests/test_algebra.py)
B: Jordan-Shadow identity residuals over random pairs
C: the anti-castle control — does the associator beat the magnitude baseline?
"""
import numpy as np

from octonion_kernel import Octonion, identity_residuals
from octonion_kernel.controls import run_control_c


def _rand_oct(rng):
    return Octonion(rng.standard_normal(8))


def report_b(n=10_000, seed=2):
    rng = np.random.default_rng(seed)
    mx = {"losslessness": 0.0, "orthogonality": 0.0, "pythagorean": 0.0}
    for _ in range(n):
        res = identity_residuals(_rand_oct(rng), _rand_oct(rng))
        for k in mx:
            mx[k] = max(mx[k], res[k])
    print(f"\n[B] Jordan-Shadow identities over {n} random pairs (max residual):")
    for k, v in mx.items():
        print(f"    {k:<14} {v:.3e}")


def report_c(n=2000, seed=0):
    out = run_control_c(n=n, seed=seed)
    print(f"\n[C] Anti-castle control ({n} structured + {n} random pairs, all unit-normalized):")
    print(f"    {'summary':<14} {'AUC':>7} {'sep':>7} {'95% CI':>20}")
    for k in ("prod_norm", "assoc_norm", "assoc_real", "assoc_maxabs"):
        r = out[k]
        print(f"    {k:<14} {r['auc']:>7.3f} {r['sep']:>7.3f} "
              f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]")
    v = out["verdict"]
    print(f"\n    magnitude baseline separation: {v['magnitude_sep']:.3f} "
          f"(should be ~0.5 — magnitude was killed by normalization)")
    print(f"    best associator summary: {v['best_summary']}")
    verdict = ("YES — associator carries information beyond magnitude"
               if v["associator_beats_magnitude"]
               else "NO — associator adds nothing over magnitude at the kernel level")
    print(f"    VERDICT: {verdict}")


if __name__ == "__main__":
    print("=" * 64)
    print("Octonion kernel control report")
    print("[A] algebra correctness: run `python -m pytest tests/test_algebra.py`")
    report_b()
    report_c()
    print("=" * 64)
