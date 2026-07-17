import numpy as np
from octonion_kernel import Octonion
from octonion_kernel.topology import trajectory_cloud, persistence_summary, persistence_summary_from_cloud


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
    assert set(s) == {"max_h1", "total_h1", "n_h1", "total_h0", "diameter", "max_h1_norm"}
    assert all(isinstance(s[k], float) for k in ("max_h1", "total_h1", "total_h0", "diameter", "max_h1_norm"))
    assert isinstance(s["n_h1"], int)


def test_max_h1_norm_is_scale_invariant():
    # scaling the whole cloud must not change normalized max-H1 (loops/diameter both scale)
    base = _circle_traj(60)
    scaled = [Octonion(3.7 * x.coeffs) for x in base]
    sb = persistence_summary(base)
    ss = persistence_summary(scaled)
    assert sb["max_h1_norm"] > 0.3          # a real loop, normalized
    assert abs(sb["max_h1_norm"] - ss["max_h1_norm"]) < 1e-6   # invariant to cloud scale
    assert ss["max_h1"] > sb["max_h1"]      # raw max-H1 DID scale up (sanity)


def test_persistence_summary_from_cloud_matches_octonion_wrapper():
    traj = _circle_traj(40)
    cloud = trajectory_cloud(traj)
    assert persistence_summary_from_cloud(cloud) == persistence_summary(traj)


def test_persistence_summary_from_cloud_works_on_lower_dimensional_clouds():
    rng = np.random.default_rng(0)
    cloud_3d = rng.standard_normal((30, 3))
    s = persistence_summary_from_cloud(cloud_3d)
    assert set(s) == {"max_h1", "total_h1", "n_h1", "total_h0", "diameter", "max_h1_norm"}
