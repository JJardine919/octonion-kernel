"""Autonomous octonion dynamics — an iterated, genuinely-nonlinear walk.

Pure: numpy + the Phase-1 kernel only. No I/O, no scipy/sklearn, no aoi_collapse.

The map (unit-sphere, feedback + decay):
    x_{t+1} = renorm( lam * x_t + (1 - lam) * associator(x_t, g) )
where associator(x, g) = jordan(x, g) * commutator(x, g) from shadow_decompose.
The associator is quadratic in x, so the map is genuinely nonlinear: a fixed-octonion
multiply alone is linear on R^8 and a linear baseline would reproduce it.
"""
from __future__ import annotations

import numpy as np

from .octonion import Octonion
from .shadow import shadow_decompose


def associator_step(x: Octonion, g: Octonion) -> Octonion:
    """The nonlinear term jordan(x,g) * commutator(x,g) (quadratic in x)."""
    return shadow_decompose(x, g).associator


def renorm(x: Octonion) -> Octonion:
    """Project onto the unit sphere. Caller ensures ||x|| is not ~0."""
    return Octonion(x.coeffs / x.norm())


# A fixed, declared generator octonion (unit norm). Pinned, never tuned to a result.
DEFAULT_GENERATOR: Octonion = Octonion(
    np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) / np.sqrt(7.0)
)


def octonion_walk(x0: Octonion, g: Octonion = DEFAULT_GENERATOR,
                  lam: float = 0.5, steps: int = 32) -> list[Octonion]:
    """Iterate the autonomous octonion map from x0, returning [x0, x1, ..., x_T].

    State is renormalized to the unit sphere each step. If a step underflows
    (||y|| < 1e-12), the walk halts and returns the trajectory so far.
    """
    traj = [renorm(x0)]
    for _ in range(steps):
        x = traj[-1]
        y = Octonion(lam * x.coeffs + (1.0 - lam) * associator_step(x, g).coeffs)
        if y.norm() < 1e-12:
            break
        traj.append(renorm(y))
    return traj
