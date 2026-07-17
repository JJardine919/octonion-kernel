"""Persistent-homology summary of an octonion walk's trajectory.

Harness-tier: imports ripser (and numpy). NEVER imported by the kernel
(octonion, shadow) or the pure engine (dynamics.py). No aoi_collapse.
"""
from __future__ import annotations

import numpy as np
from ripser import ripser
from scipy.spatial.distance import pdist

from .octonion import Octonion


def trajectory_cloud(traj: list[Octonion]) -> np.ndarray:
    """Stack a trajectory's coefficients into an (len(traj), 8) array."""
    return np.array([x.coeffs for x in traj], dtype=float)


def persistence_summary_from_cloud(cloud: np.ndarray) -> dict:
    """Persistent-homology summary of a raw point cloud (Euclidean, maxdim=1). Same
    computation as persistence_summary, on any (N, d) array directly -- used by
    Phase 5's compression codes, which aren't always 8-dimensional (e.g. PCA's
    k-dimensional codes).

    Returns:
      max_h1   - single longest H1 (loop) lifetime, 0.0 if none [VERDICT METRIC]
      total_h1 - sum of all H1 lifetimes (context)
      n_h1     - number of H1 features (context)
      total_h0 - sum of finite H0 lifetimes (context)
    """
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
    diameter = float(pdist(cloud).max()) if len(cloud) > 1 else 0.0
    max_h1_norm = float(max_h1 / diameter) if diameter > 1e-12 else 0.0
    return {"max_h1": max_h1, "total_h1": total_h1, "n_h1": n_h1, "total_h0": total_h0,
            "diameter": diameter, "max_h1_norm": max_h1_norm}


def persistence_summary(traj: list[Octonion]) -> dict:
    """Persistent-homology summary of the trajectory point cloud (Euclidean, maxdim=1).
    Thin wrapper over persistence_summary_from_cloud -- see that function for the
    return-value docs."""
    return persistence_summary_from_cloud(trajectory_cloud(traj))
