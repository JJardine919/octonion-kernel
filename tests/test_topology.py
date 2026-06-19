import numpy as np
from octonion_kernel import Octonion
from octonion_kernel.topology import trajectory_cloud, persistence_summary


def _circle_traj(n=60):
    # n points on a circle in the first 2 octonion coords (the rest zero)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [Octonion(np.array([np.cos(a), np.sin(a), 0, 0, 0, 0, 0, 0])) for a in t]


def _cluster_traj(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return [Octonion(np.concatenate([[1.0], 0.001 * rng.standard_normal(7)])) for _ in range(n)]


def test_trajectory_cloud_shape_and_values():
    cloud = trajectory_cloud(_circle_traj(10))
    assert cloud.shape == (10, 8)
    assert np.allclose(cloud[:, 2:], 0.0)


def test_circle_has_a_persistent_loop():
    s = persistence_summary(_circle_traj(60))
    assert s["n_h1"] >= 1
    assert s["max_h1"] > 0.5  # the circle's loop is long-lived at radius scale


def test_tight_cluster_has_no_real_loop():
    s = persistence_summary(_cluster_traj(60))
    assert s["max_h1"] < 0.05  # a tight blob carries no real loop


def test_persistence_summary_is_deterministic():
    traj = _circle_traj(40)
    assert persistence_summary(traj) == persistence_summary(traj)


def test_summary_keys_and_types():
    s = persistence_summary(_circle_traj(30))
    assert set(s) == {"max_h1", "total_h1", "n_h1", "total_h0"}
    assert all(isinstance(s[k], float) for k in ("max_h1", "total_h1", "total_h0"))
    assert isinstance(s["n_h1"], int)
