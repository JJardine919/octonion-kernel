"""Runnable A/B/C control report. Run from the repo root: python harness_report.py

A: algebra correctness  (delegates to pytest tests/test_algebra.py)
B: Jordan-Shadow identity residuals over random pairs
C: the anti-castle control — does the associator beat the best trivial baseline?
"""
import numpy as np

from octonion_kernel import Octonion, identity_residuals
from octonion_kernel.controls import run_control_c
from octonion_kernel.dynamics_controls import run_dynamics_control
from octonion_kernel.topology_controls import run_topology_control


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
    groups = {
        "prod_norm": "magnitude", "dot_abs": "trivial", "jordan_norm": "trivial",
        "commutator_norm": "trivial", "assoc_norm": "associator",
        "assoc_real": "associator", "assoc_maxabs": "associator",
    }
    print(f"\n[C] Anti-castle control ({n} structured + {n} random pairs, all unit-normalized):")
    print(f"    {'summary':<16} {'group':<11} {'AUC':>7} {'sep':>7} {'95% CI':>18}")
    for k in ("prod_norm", "dot_abs", "jordan_norm", "commutator_norm",
              "assoc_norm", "assoc_real", "assoc_maxabs"):
        r = out[k]
        print(f"    {k:<16} {groups[k]:<11} {r['auc']:>7.3f} {r['sep']:>7.3f} "
              f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]")
    v = out["verdict"]
    print(f"\n    magnitude baseline separation: {v['magnitude_sep']:.3f} "
          f"(~0.5 => magnitude killed by unit-normalization)")
    print(f"    best associator summary:      {v['best_associator_summary']} (sep {v['assoc_sep']:.3f})")
    print(f"    best non-associator baseline: {v['best_baseline_summary']} (sep {v['best_baseline_sep']:.3f})")
    print(f"    sep(associator) - sep(best baseline), 95% CI: "
          f"[{v['sep_difference_ci'][0]:.3f}, {v['sep_difference_ci'][1]:.3f}]")
    if v["associator_adds_information"]:
        print("    VERDICT: YES - the associator separates the classes better than every")
        print("             declared trivial baseline (including the plain dot product |a.b|).")
    else:
        print("    VERDICT: NO - the associator does NOT beat the best trivial baseline.")
        print("             Its separation only reflects the a-b angle, which the plain dot")
        print("             product |a.b| (and the Jordan/commutator norms) capture as well or")
        print("             better. The associator carries no information beyond trivial statistics here.")


def report_d(n=400, steps=32, seed=0):
    out = run_dynamics_control(n=n, steps=steps, seed=seed)
    print(f"\n[D] Dynamics control ({n} structured + {n} random initial states, "
          f"{steps}-step unit-sphere walks):")
    print(f"    {'map':<18} {'test AUC':>9}")
    for k in ("raw", "linear", "generic_nonlinear", "random_walk", "octonion"):
        print(f"    {k:<18} {out['aucs'][k]:>9.3f}")
    v = out["verdict"]
    print(f"\n    best baseline:  {v['best_baseline']} (AUC {v['best_baseline_auc']:.3f})")
    print(f"    octonion AUC:   {v['octonion_auc']:.3f}")
    print(f"    AUC(octonion) - AUC(best baseline), 95% CI: "
          f"[{v['auc_difference_ci'][0]:.3f}, {v['auc_difference_ci'][1]:.3f}]")
    if v["octonion_adds_structure"]:
        print("    VERDICT: YES - iterating the octonion walk separates the classes better")
        print("             than the raw input AND every declared baseline (incl. a matched")
        print("             generic quadratic map).")
    else:
        print("    VERDICT: NO - the octonion walk does NOT beat the best baseline. Iteration")
        print("             adds no separability beyond a matched linear / generic-nonlinear /")
        print("             random map; the octonion structure is not doing the work here.")


def report_e(n=200, steps=256, seed=0):
    out = run_topology_control(n=n, steps=steps, seed=seed)
    print(f"\n[E] Topology control ({n} random initial states, {steps}-step walks, "
          f"max-H1 persistence over the trajectory):")
    print(f"    {'map':<18} {'mean max-H1':>12}")
    for k in ("linear", "generic_nonlinear", "random_walk", "octonion"):
        print(f"    {k:<18} {out['max_h1'][k]:>12.4f}")
    print(f"    {'iid_cloud (null)':<18} {out['iid_cloud_max_h1']:>12.4f}")
    v = out["verdict"]
    print(f"\n    best baseline:  {v['best_baseline']} (mean max-H1 {v['best_baseline_max_h1']:.4f})")
    print(f"    octonion mean max-H1: {v['octonion_max_h1']:.4f}")
    print(f"    mean[maxH1(octonion) - maxH1(best baseline)], 95% CI: "
          f"[{v['diff_ci'][0]:.4f}, {v['diff_ci'][1]:.4f}]")
    if v["octonion_adds_topology"]:
        print("    VERDICT: YES - the octonion walk's trajectory carries more persistent loop")
        print("             structure than every matched baseline (linear / generic-nonlinear /")
        print("             random-walk diffusion).")
    else:
        print("    VERDICT: NO - the octonion walk does NOT produce more loop structure than the")
        print("             best matched baseline. Its trajectory topology is not distinctive")
        print("             beyond a linear / generic-nonlinear / diffusion process.")


if __name__ == "__main__":
    print("=" * 64)
    print("Octonion kernel control report")
    print("[A] algebra correctness: run `python -m pytest tests/test_algebra.py`")
    report_b()
    report_c()
    report_d()
    report_e()
    print("=" * 64)
