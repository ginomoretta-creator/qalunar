"""Numerical Phase-1 Earth-escape spiral *with the Moon included*.

The closed-form :func:`qalunar.reference.edelbaum.edelbaum_spiral` deliberately
drops the Moon: it is a pure two-body Earth-centered Edelbaum spiral, valid
because the lunar pull is tiny near Earth. That approximation degrades as the
parking orbit grows toward the lunar distance --- a 200,000 km parking orbit
sits at ~0.52 nondimensional units, more than halfway to the Moon, where the
Moon's direct pull is already ~1-2 % of Earth's and accumulates over the
revolutions of the spiral.

This module provides the Moon-aware counterpart: it integrates the *full*
planar CR3BP (:class:`qalunar.dynamics.cr3bp.PlanarCR3BP`, which carries both
the Earth and the Moon) under continuous tangential thrust, starting from a
circular Earth orbit at radius ``r1`` and terminating when the Earth-relative
radius first reaches ``r2``. Because the dynamics now include the Moon there is
no closed form; we propagate with adaptive-free fixed-step RK4 and recompute the
tangential thrust direction at every stage.

The thrust direction convention matches the rest of the codebase
(:func:`qalunar.qubo.thrust_scheduling._thrust_direction_at_step`): ``tangential``
means *along the synodic-frame velocity* ``state[2:4]``, so the Phase-1 arc is
consistent with the Phase-2/3 binary schedules that follow it.

The returned :class:`NumericSpiralResult` exposes the same handoff fields as the
analytical :class:`~qalunar.reference.edelbaum.EdelbaumSpiral` (``dv``, ``tof``,
``n_revs``, and ``handoff_state_synodic``) so it is a drop-in replacement at the
Phase-1 -> Phase-2 boundary, plus the full propagated trajectory for diagnostics
and for comparing the lunar-perturbation miss against the closed form.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from qalunar.dynamics.cr3bp import EARTH_MOON_MU, PlanarCR3BP


__all__ = [
    "NumericSpiralResult",
    "phase1_spiral_numeric",
]


@dataclass(frozen=True)
class NumericSpiralResult:
    """Result of a Moon-aware numerical Phase-1 spiral.

    All quantities are in CR3BP nondimensional units. Mirrors the handoff
    interface of :class:`qalunar.reference.edelbaum.EdelbaumSpiral` so callers
    can swap the analytical spiral for this one at the Phase-2 boundary.

    Attributes
    ----------
    r1, r2 : float
        Initial and (achieved) final Earth-relative circular-orbit radius.
        ``r2_achieved`` may slightly overshoot the requested target because
        termination is detected on a discrete grid and refined by linear
        interpolation.
    v1 : float
        Initial circular inertial speed at ``r1`` (``sqrt(GM_earth / r1)``).
    dv : float
        Total Delta-v expended, equal to ``a_thrust * tof`` for a continuous
        full-on arc.
    tof : float
        Time of flight to reach ``r2``.
    n_revs : float
        Number of inertial revolutions about the Earth during the spiral.
    a_thrust : float
        Constant tangential thrust-acceleration magnitude that was applied.
    mu : float
        CR3BP mass parameter used by the dynamics.
    handoff_state : (4,) ndarray
        Synodic-frame state ``[x, y, vx, vy]`` at the end of the spiral ---
        the Moon-perturbed handoff into Phase 2.
    t_grid : (n+1,) ndarray
        Time grid of the propagation (truncated at the ``r2`` crossing).
    trajectory : (n+1, 4) ndarray
        Synodic-frame state at each time on ``t_grid``.
    converged : bool
        ``True`` if the spiral reached ``r2`` within the step budget.
    """

    r1: float
    r2: float
    v1: float
    dv: float
    tof: float
    n_revs: float
    a_thrust: float
    mu: float
    handoff_state: NDArray[np.float64]
    t_grid: NDArray[np.float64]
    trajectory: NDArray[np.float64]
    converged: bool

    def handoff_state_synodic(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Position and velocity at the end of the spiral, in synodic coords.

        Matches the signature of
        :meth:`qalunar.reference.edelbaum.EdelbaumSpiral.handoff_state_synodic`
        so this result is a drop-in replacement at the Phase-1 -> Phase-2
        boundary.
        """
        return self.handoff_state[:2].copy(), self.handoff_state[2:4].copy()


def _circular_earth_state(
    r1: float, mu: float
) -> NDArray[np.float64]:
    """Synodic-frame state of a prograde circular Earth orbit at radius ``r1``.

    Identical convention to
    :meth:`EdelbaumSpiral.handoff_state_synodic` for ``body="earth"``:
    the spacecraft crosses the ``+x`` direction relative to Earth with
    inertial velocity ``(0, v1)``; the synodic velocity subtracts the
    frame rotation ``omega x r``.
    """
    v1 = float(np.sqrt((1.0 - mu) / r1))
    r_syn = np.array([-mu + r1, 0.0])
    v_inertial = np.array([0.0, v1])
    omega_cross_r = np.array([-r_syn[1], r_syn[0]])
    v_syn = v_inertial - omega_cross_r
    return np.array([r_syn[0], r_syn[1], v_syn[0], v_syn[1]])


def phase1_spiral_numeric(
    *,
    r1: float,
    r2: float,
    a_thrust: float,
    dynamics: PlanarCR3BP | None = None,
    max_tof: float | None = None,
    n_steps: int = 40_000,
) -> NumericSpiralResult:
    """Moon-aware numerical Phase-1 Earth-escape spiral.

    Propagates the full planar CR3BP (Earth *and* Moon) from a circular Earth
    orbit at radius ``r1`` under continuous tangential thrust of magnitude
    ``a_thrust``, terminating when the Earth-relative radius first reaches
    ``r2``. This is the Moon-included counterpart to the closed-form
    :func:`qalunar.reference.edelbaum.edelbaum_spiral`.

    Parameters
    ----------
    r1, r2 : float
        Initial and target Earth-relative circular-orbit radius, in CR3BP
        nondimensional units (1 unit = Earth-Moon distance). Requires
        ``r2 > r1 > 0`` (ascending spiral).
    a_thrust : float
        Constant tangential thrust acceleration magnitude, CR3BP units.
    dynamics : PlanarCR3BP, optional
        Dynamics model carrying the Moon. Defaults to ``PlanarCR3BP()``
        (Earth-Moon ``mu``).
    max_tof : float, optional
        Hard cap on the integrated time. Defaults to ``3 *`` the analytical
        Edelbaum time of flight, which comfortably brackets the true (slightly
        perturbed) arc while bounding runaway integrations.
    n_steps : int, optional
        Number of fixed RK4 steps spanning ``[0, max_tof]``. The thrust
        direction is recomputed at every RK4 stage. Default 40000.

    Returns
    -------
    NumericSpiralResult

    Raises
    ------
    ValueError
        If radii are non-positive or ``r2 <= r1``, or if ``a_thrust <= 0``.
    """
    if r1 <= 0.0 or r2 <= 0.0:
        raise ValueError(f"r1 and r2 must be positive; got r1={r1}, r2={r2}")
    if r2 <= r1:
        raise ValueError(
            f"Ascending spiral requires r2 > r1; got r1={r1}, r2={r2}"
        )
    if a_thrust <= 0.0:
        raise ValueError(f"a_thrust must be positive; got {a_thrust}")

    dyn = dynamics if dynamics is not None else PlanarCR3BP()
    mu = dyn.mu
    gm_earth = 1.0 - mu
    v1 = float(np.sqrt(gm_earth / r1))
    v2 = float(np.sqrt(gm_earth / r2))

    if max_tof is None:
        # Analytical Edelbaum TOF = |v1 - v2| / a_thrust; pad generously so
        # the (slower, Moon-perturbed) arc has room to reach r2.
        max_tof = 3.0 * abs(v1 - v2) / a_thrust

    def rhs(state: NDArray[np.float64]) -> NDArray[np.float64]:
        v = state[2:4]
        v_norm = float(np.linalg.norm(v))
        vhat = v / v_norm if v_norm > 1e-15 else np.array([1.0, 0.0])
        return dyn.rhs(state, a_thrust * vhat)

    earth = np.array([-mu, 0.0])
    dt = max_tof / n_steps

    states = np.empty((n_steps + 1, 4))
    t_grid = np.empty(n_steps + 1)
    state = _circular_earth_state(r1, mu)
    states[0] = state
    t_grid[0] = 0.0

    # Accumulated inertial angle about the Earth, for the revolution count.
    # Inertial angle = synodic angle + t (the frame rotates at unit rate).
    d0 = state[:2] - earth
    prev_syn_angle = float(np.arctan2(d0[1], d0[0]))
    accumulated_syn_angle = 0.0

    converged = False
    n_used = n_steps
    for i in range(n_steps):
        k1 = rhs(state)
        k2 = rhs(state + 0.5 * dt * k1)
        k3 = rhs(state + 0.5 * dt * k2)
        k4 = rhs(state + dt * k3)
        new_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Unwrapped synodic angle increment about the Earth.
        d = new_state[:2] - earth
        syn_angle = float(np.arctan2(d[1], d[0]))
        delta = syn_angle - prev_syn_angle
        if delta > np.pi:
            delta -= 2.0 * np.pi
        elif delta < -np.pi:
            delta += 2.0 * np.pi
        accumulated_syn_angle += delta
        prev_syn_angle = syn_angle

        r_earth_prev = float(np.linalg.norm(state[:2] - earth))
        r_earth_new = float(np.linalg.norm(d))

        states[i + 1] = new_state
        t_grid[i + 1] = t_grid[i] + dt

        if r_earth_new >= r2:
            # Linear interpolation to the r2 crossing for a clean handoff.
            denom = r_earth_new - r_earth_prev
            frac = (r2 - r_earth_prev) / denom if denom != 0.0 else 1.0
            frac = float(np.clip(frac, 0.0, 1.0))
            state_cross = state + frac * (new_state - state)
            t_cross = t_grid[i] + frac * dt
            states[i + 1] = state_cross
            t_grid[i + 1] = t_cross
            # Correct the angle accumulation for the partial last step.
            accumulated_syn_angle += (frac - 1.0) * delta
            n_used = i + 1
            converged = True
            break

        state = new_state

    t_grid = t_grid[: n_used + 1]
    states = states[: n_used + 1]
    handoff = states[-1].copy()
    tof = float(t_grid[-1])
    r2_achieved = float(np.linalg.norm(handoff[:2] - earth))

    # Inertial revolutions = (swept synodic angle + elapsed time) / 2 pi,
    # since the synodic frame itself rotates one revolution per 2 pi of time.
    total_inertial_angle = accumulated_syn_angle + tof
    n_revs = float(abs(total_inertial_angle) / (2.0 * np.pi))

    return NumericSpiralResult(
        r1=r1,
        r2=r2_achieved,
        v1=v1,
        dv=float(a_thrust * tof),
        tof=tof,
        n_revs=n_revs,
        a_thrust=a_thrust,
        mu=mu,
        handoff_state=handoff,
        t_grid=t_grid,
        trajectory=states,
        converged=converged,
    )
