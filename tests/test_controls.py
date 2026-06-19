import numpy as np
from octonion_kernel.controls import (
    auc, bootstrap_sep_ci, structured_pair, random_pair, run_control_c,
)


def test_auc_perfect_separation():
    scores = np.array([0.0, 0.1, 0.2, 1.0, 1.1, 1.2])
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert abs(auc(scores, labels) - 1.0) < 1e-12


def test_auc_chance():
    scores = np.array([0.0, 1.0, 0.0, 1.0])
    labels = np.array([0, 0, 1, 1])
    assert abs(auc(scores, labels) - 0.5) < 1e-12


def test_bootstrap_ci_brackets_separation():
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.normal(0, 1, 200), rng.normal(3, 1, 200)])
    labels = np.concatenate([np.zeros(200), np.ones(200)])
    lo, hi = bootstrap_sep_ci(scores, labels, n_boot=500, seed=1)
    assert 0.5 <= lo <= hi <= 1.0
    assert lo > 0.5  # clearly separated data => CI lower bound above chance


def test_pair_generators_are_unit_normalized():
    rng = np.random.default_rng(2)
    for _ in range(50):
        a, b = structured_pair(rng)
        assert abs(a.norm() - 1.0) < 1e-9 and abs(b.norm() - 1.0) < 1e-9
        a, b = random_pair(rng)
        assert abs(a.norm() - 1.0) < 1e-9 and abs(b.norm() - 1.0) < 1e-9


def test_control_c_runs_kills_magnitude_and_produces_verdict():
    out = run_control_c(n=1500, seed=0)
    # a verdict is produced regardless of its value
    assert "verdict" in out
    assert isinstance(out["verdict"]["associator_beats_magnitude"], bool)
    assert out["verdict"]["best_summary"] in ("assoc_norm", "assoc_real", "assoc_maxabs")
    # SANITY: magnitude is killed by unit-normalization, so ||product|| can't separate.
    # This guards the experiment's validity; it does NOT assert the associator wins.
    assert out["prod_norm"]["sep"] < 0.6
    # every reported summary has a well-formed CI
    for key in ("prod_norm", "assoc_norm", "assoc_real", "assoc_maxabs"):
        assert 0.5 <= out[key]["ci_lo"] <= out[key]["ci_hi"] <= 1.0
