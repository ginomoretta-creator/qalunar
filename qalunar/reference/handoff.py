"""Helpers for Phase 2 -> Phase 3 handoff (cislunar arc to lunar orbit).

The end of a Phase-2 cislunar correction leaves the spacecraft on a
synodic-frame state somewhere near the Moon, typically on a hyperbolic
flyby trajectory. To enter the Phase-3 Moon-descending Edelbaum spiral
we need three pieces of information at the perilune of that flyby:

1. The Moon-relative radius ``r_perilune`` (becomes ``r1`` for Phase 3).
2. The Moon-relative speed ``v_perilune`` at perilune.
3. The local circular speed ``v_circ = sqrt(mu / r_perilune)``.

The capture Delta-v is the gap between the actual flyby speed and the
circular speed at perilune: it is what the spacecraft must shed (with
anti-tangential thrust near perilune) to transition from the Phase-2
hyperbolic flyby into a closed orbit around the Moon. The Phase-3
analytical Edelbaum spiral then takes that closed orbit at ``r_perilune``
and lowers it to the target LLO at ``r_LLO``.

The Moon-relative state is computed in the *inertial* frame of the
CR3BP, since Edelbaum's two-body assumption ignores the Moon's own
motion: in that picture the Moon is fixed and the spacecraft's
"actual" velocity relative to the Moon is the inertial velocity minus
the Moon's inertial velocity ``omega x r_moon``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from qalunar.dynamics.cr3bp import EARTH_MOON_MU, PlanarCR3BP


__all__ = [
    "moon_relative_inertial",
    "find_perilune",
    "capture_delta_v",
    "PerilunePass",
]


def moon_relative_inertial(
    state_synodic: NDArray[np.float64],
    mu: float = EARTH_MOON_MU,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the spacecraft's Moon-relative position and inertial velocity.

    Given a synodic-frame state ``[x, y, vx, vy]``, this converts to the
    inertial frame and subtracts the Moon's own inertial velocity to
    obtain the velocity of the spacecraft relative to the Moon (still
    expressed in inertial Cartesian components).

    The Moon-relative position is just ``r_spacecraft - r_moon`` and is
    frame-invariant for a single instant since both frames coincide at
    that instant.

    Parameters
    ----------
    state_synodic : (4,) ndarray
        Synodic-frame state ``[x, y, vx, vy]``.
    mu : float, optional
        CR3BP mass parameter.

    Returns
    -------
    r_moon_rel : (2,) ndarray
        Position relative to the Moon.
    v_moon_rel_inertial : (2,) ndarray
        Velocity relative to the Moon, in the inertial frame.
    """
    state = np.asarray(state_synodic, dtype=np.float64)
    r = state[:2]
    v_syn = state[2:4]

    moon_pos = np.array([1.0 - mu, 0.0])

    # Synodic -> inertial: v_inertial = v_syn + omega x r, omega x r = (-y, x).
    omega_cross_r = np.array([-r[1], r[0]])
    v_inertial = v_syn + omega_cross_r

    # Moon's inertial velocity at t = 0 (frames aligned): omega x r_moon.
    v_moon_inertial = np.array([-moon_pos[1], moon_pos[0]])

    r_rel = r - moon_pos
    v_rel = v_inertial - v_moon_inertial

    return r_rel, v_rel


@dataclass(frozen=True)
class PerilunePass:
    """Closest-approach pass to the Moon along an unforced trajectory.

    Attributes
    ----------
    t : float
        Time of perilune (relative to the start of the propagation).
    state_synodic : (4,) ndarray
        Synodic-frame state at perilune.
    r_moon_rel : (2,) ndarray
        Position vector from the Moon to the spacecraft at perilune.
    v_moon_rel_inertial : (2,) ndarray
        Velocity of the spacecraft relative to the Moon, in the
        inertial frame.
    r_perilune : float
        Distance from the Moon at perilune, ``|r_moon_rel|``.
    v_perilune : float
        Speed relative to the Moon at perilune,
        ``|v_moon_rel_inertial|``.
    v_circ : float
        Local circular speed at ``r_perilune`` around the Moon,
        ``sqrt(mu / r_perilune)``.
    is_bound : bool
        ``True`` iff the two-body energy w.r.t. the Moon is negative
        (already on a closed orbit).
    """

    t: float
    state_synodic: NDArray[np.float64]
    r_moon_rel: NDArray[np.float64]
    v_moon_rel_inertial: NDArray[np.float64]
    r_perilune: float
    v_perilune: float
    v_circ: float
    is_bound: bool


def find_perilune(
    dynamics: PlanarCR3BP,
    state0_synodic: NDArray[np.float64],
    t_span: tuple[float, float],
    n_steps: int = 4_000,
) -> PerilunePass:
    """Locate the perilune (closest approach to the Moon) along an
    unforced CR3BP propagation.

    Propagates the state with no thrust and finds the time at which the
    distance to the Moon is minimised. Returns a :class:`PerilunePass`
    summarising the geometry and the Moon-relative kinematics there.

    Parameters
    ----------
    dynamics : PlanarCR3BP
        The CR3BP dynamics model.
    state0_synodic : (4,) ndarray
        Initial synodic-frame state.
    t_span : (t0, tf)
        Time interval over which to search.
    n_steps : int, optional
        RK4 propagation steps. Higher values give a finer perilune
        time. Default 4000.

    Returns
    -------
    PerilunePass
    """
    t, traj = dynamics.propagate(state0_synodic, t_span, n_steps=n_steps)
    moon = np.array([1.0 - dynamics.mu, 0.0])
    dist = np.linalg.norm(traj[:, :2] - moon, axis=1)
    idx = int(np.argmin(dist))

    state_p = traj[idx]
    t_p = float(t[idx])

    r_rel, v_rel = moon_relative_inertial(state_p, mu=dynamics.mu)
    r_perilune = float(np.linalg.norm(r_rel))
    v_perilune = float(np.linalg.norm(v_rel))
    v_circ = float(np.sqrt(dynamics.mu / r_perilune))

    # Two-body energy with respect to the Moon.
    energy = 0.5 * v_perilune ** 2 - dynamics.mu / r_perilune
    is_bound = bool(energy < 0.0)

    return PerilunePass(
        t=t_p,
        state_synodic=state_p,
        r_moon_rel=r_rel,
        v_moon_rel_inertial=v_rel,
        r_perilune=r_perilune,
        v_perilune=v_perilune,
        v_circ=v_circ,
        is_bound=is_bound,
    )


def capture_delta_v(pass_: PerilunePass) -> float:
    """Delta-v required to circularise at perilune.

    The classical capture cost: subtract the local circular speed from
    the actual perilune speed. Positive values mean the flyby is
    hyperbolic and requires deceleration; negative values (capped to
    zero in practice) mean the spacecraft is already moving slower
    than circular and would need to be sped up -- a regime that does
    not occur for normal flybys.

    Parameters
    ----------
    pass_ : PerilunePass

    Returns
    -------
    float
        Capture Δv in nondimensional units. Multiply by
        ``qalunar.reference.edelbaum.VELOCITY_M_S`` for m/s.
    """
    return float(max(pass_.v_perilune - pass_.v_circ, 0.0))
