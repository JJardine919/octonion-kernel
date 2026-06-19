"""Persistent-homology summary of an octonion walk's trajectory.

Harness-tier: imports ripser (and numpy). NEVER imported by the kernel
(octonion, shadow) or the pure engine (dynamics.py). No aoi_collapse.
"""
from __future__ import annotations

import numpy as np
from ripser import ripser

from .octonion import Octonion


def trajectory_cloud(traj: list[Octonion]) -> np.ndarray:
    """Stack a trajectory's coefficients into an (len(traj), 8) array."""
    return np.array([x.coeffs for x in traj], dtype=float)


def persistence_summary(traj: list[Octonion]) -> dict:
    """Persistent-homology summary of the trajectory point cloud (Euclidean, maxdim=1).

    Returns:
      max_h1   - single longest H1 (loop) lifetime, 0.0 if none [VERDICT METRIC]
      total_h1 - sum of all H1 lifetimes (context)
      n_h1     - number of H1 features (context)
      total_h0 - sum of finite H0 lifetimes (context)
    """
    cloud = trajectory_cloud(traj)
    dgms = ripser(cloud, maxdim=1)["dgms"]
    h0, h1 = dgms[0], dgms[1]
    if len(h1):
        life1 = h1[:, 1] - h1[:, 0]
        max_h1 = float(np.max(life1))
        total_h1 = float(np.sum(life1))
        n_h1 = int(len(h1))
    else:
        max_h1 = 0.0
        total_h1 = 0.0
        n_h1 = 0
    life0 = h0[:, 1] - h0[:, 0]
    life0 = life0[np.isfinite(life0)]  # drop the one infinite H0 bar
    total_h0 = float(np.sum(life0))
    return {"max_h1": max_h1, "total_h1": total_h1, "n_h1": n_h1, "total_h0": total_h0}
