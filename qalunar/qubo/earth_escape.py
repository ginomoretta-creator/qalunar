"""Sliding-window QUBO scheduler for the Phase-1 Earth-escape spiral.

This is the mirror image of :mod:`qalunar.qubo.lunar_capture`. Where the
lunar-capture solver chains short anti-tangential QUBO windows to *dissipate*
Moon-relative energy until the spacecraft is bound, this module chains short
**tangential** QUBO windows to *inject* Earth-relative energy until the
osculating semi-major axis reaches a target parking-orbit radius.

The motivation is the "QUBO everywhere, three bodies everywhere" upgrade: rather
than carving Phase 1 out as a closed-form two-body Edelbaum spiral, the
semi-major-axis raise is solved with exactly the same naturally-binary QUBO
machinery as Phases 2 and 3, in the full planar CR3BP (Earth *and* Moon present
in every step). Phase 1 then differs from the other phases only in its
per-window objective (raise energy, ``tangential``) and thrust level, not in any
of the underlying formulation.

Implementation note
-------------------
The window-chaining loop is the phase-agnostic
:func:`qalunar.qubo.receding_horizon.solve_receding_horizon`. This module is the
Earth-side adapter: Earth as the
:class:`~qalunar.qubo.receding_horizon.CentralBody`, a tangential channel, a
boosting target speed (``boost_factor > 1``), and a stop rule on the
Earth-relative semi-major axis. :mod:`qalunar.qubo.lunar_capture` is the dual
(Moon, anti-tangential, braking) adapter over the same driver.

Why a single global QUBO does not work for Phase 1
--------------------------------------------------
Raising the orbit from GEO (~0.11 nondim) to a 200,000 km parking orbit
(~0.52 nondim) is a large-excursion, many-revolution, bulk-Delta-v problem. A
single linearisation about one reference cannot survive a 5x radius change, and
resolving the thrust modulation over many revolutions in one binary vector would
need far more qubits than any sampler can handle. The receding-horizon
(sliding-window) structure is what makes it tractable: each window is short
enough that the local linearisation holds, and the windows chain to cover the
whole climb.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from qalunar.dynamics.cr3bp import EARTH_MOON_MU, PlanarCR3BP
from qalunar.qubo.receding_horizon import (
    CentralBody,
    RecedingHorizonWindow,
    solve_receding_horizon,
)
from qalunar.qubo.thrust_scheduling import IterativeSchedulingResult


__all__ = [
    "EarthEscapeWindow",
    "EarthEscapeResult",
    "earth_relative_inertial",
    "earth_two_body_energy",
    "earth_orbit_apoapsis_periapsis",
    "solve_earth_escape_sliding_window",
]


# ---------------------------------------------------------------------------
# Earth-relative orbital diagnostics (delegated to CentralBody so Earth and
# Moon diagnostics share one implementation).
# ---------------------------------------------------------------------------


def earth_relative_inertial(
    state_synodic: NDArray[np.float64],
    mu: float = EARTH_MOON_MU,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Earth-relative position and inertial velocity from a synodic state.

    Earth sits at ``(-mu, 0)`` in the synodic frame; its inertial velocity is
    ``omega x r_earth = (0, -mu)``.
    """
    return CentralBody.earth(mu).relative_inertial(state_synodic)


def earth_two_body_energy(
    state_synodic: NDArray[np.float64],
    mu: float = EARTH_MOON_MU,
) -> float:
    """Specific two-body orbital energy w.r.t. Earth (``GM_earth = 1 - mu``).

    Negative -> bound; the Earth-escape spiral drives this *up* toward the
    (less negative) energy of the target parking orbit.
    """
    return CentralBody.earth(mu).two_body_energy(state_synodic)


def earth_orbit_apoapsis_periapsis(
    state_synodic: NDArray[np.float64],
    mu: float = EARTH_MOON_MU,
) -> tuple[float, float, float]:
    """Two-body apoapsis, periapsis, and semi-major axis around Earth.

    Returns ``(r_apo, r_peri, a)`` in nondimensional units. For an unbound
    state ``a`` and ``r_apo`` are ``+inf``.
    """
    return CentralBody.earth(mu).apoapsis_periapsis_sma(state_synodic)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class EarthEscapeWindow:
    """One sliding-window step of the Earth-escape solver."""

    index: int
    state_initial: NDArray[np.float64]
    state_final: NDArray[np.float64]
    schedule: NDArray[np.int64]
    n_burns: int
    delta_v: float
    energy_initial: float
    energy_final: float
    sma_initial: float
    sma_final: float
    predicted_miss_norm: float
    true_miss_norm: float
    iterative_result: IterativeSchedulingResult


@dataclass
class EarthEscapeResult:
    """Concatenated result of the sliding-window Earth-escape solver."""

    windows: list[EarthEscapeWindow] = field(default_factory=list)
    final_state: NDArray[np.float64] = field(default_factory=lambda: np.zeros(4))
    total_delta_v: float = 0.0
    total_burns: int = 0
    total_tof: float = 0.0
    reached_target: bool = False
    full_schedule: NDArray[np.int64] = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )

    def sma_history(self) -> NDArray[np.float64]:
        if not self.windows:
            return np.array([])
        a = [self.windows[0].sma_initial]
        for w in self.windows:
            a.append(w.sma_final)
        return np.array(a)


def _to_escape_window(w: RecedingHorizonWindow) -> EarthEscapeWindow:
    """Adapt a generic driver window onto the Earth-escape window type."""
    return EarthEscapeWindow(
        index=w.index,
        state_initial=w.state_initial,
        state_final=w.state_final,
        schedule=w.schedule,
        n_burns=w.n_burns,
        delta_v=w.delta_v,
        energy_initial=w.energy_initial,
        energy_final=w.energy_final,
        sma_initial=w.sma_initial,
        sma_final=w.sma_final,
        predicted_miss_norm=w.predicted_miss_norm,
        true_miss_norm=w.true_miss_norm,
        iterative_result=w.iterative_result,
    )


# ---------------------------------------------------------------------------
# Main solver (thin adapter over the phase-agnostic driver)
# ---------------------------------------------------------------------------


def solve_earth_escape_sliding_window(
    dynamics: PlanarCR3BP,
    state0_synodic: NDArray[np.float64],
    sampler,
    *,
    target_sma: float,
    n_windows: int = 200,
    window_revs: float = 0.25,
    window_t_max: float | None = None,
    n_decision_steps: int = 10,
    thrust_magnitude: float = 1.286,
    boost_factor: float = 1.05,
    target_position_weight: float = 0.0,
    target_velocity_weight: float = 1.0,
    fuel_weight: float = 0.0,
    n_integration_substeps: int = 60,
    n_truth_substeps: int = 200,
    max_inner_iters: int = 4,
    verbose: bool = False,
) -> EarthEscapeResult:
    """Raise the orbit from ``state0`` to ``target_sma`` with binary tangential
    thrust over chained QUBO windows in the full CR3BP.

    Parameters
    ----------
    dynamics : PlanarCR3BP
    state0_synodic : (4,) ndarray
        Initial synodic-frame state (e.g. a circular Earth orbit at GEO).
    sampler : callable
        QUBO sampler ``(qubo) -> bitstring``.
    target_sma : float
        Target Earth-relative semi-major axis (nondim). The loop stops once
        the osculating SMA reaches this value.
    n_windows : int
        Maximum number of windows.
    window_revs : float
        Window length as a fraction of the local Earth-relative circular
        period.
    n_decision_steps : int
        Per-window QUBO size (qubits).
    thrust_magnitude : float
        Tangential thrust acceleration (nondim). Defaults to main power.
    boost_factor : float
        Target speed multiple of local circular speed (``> 1``).
    verbose : bool

    Returns
    -------
    EarthEscapeResult
    """
    earth = CentralBody.earth(dynamics.mu)

    def _reached(state: NDArray[np.float64], body: CentralBody) -> bool:
        _, _, sma = body.apoapsis_periapsis_sma(state)
        return bool(np.isfinite(sma) and sma >= target_sma)

    rh = solve_receding_horizon(
        dynamics, state0_synodic, sampler,
        body=earth,
        thrust_direction="tangential",
        target_factor=boost_factor,
        thrust_magnitude=thrust_magnitude,
        n_windows=n_windows,
        window_revs=window_revs,
        window_t_max=window_t_max,
        n_decision_steps=n_decision_steps,
        target_radius=None,
        target_position_weight=target_position_weight,
        target_velocity_weight=target_velocity_weight,
        fuel_weight=fuel_weight,
        action_radius=None,
        terminate=_reached,
        n_integration_substeps=n_integration_substeps,
        n_truth_substeps=n_truth_substeps,
        max_inner_iters=max_inner_iters,
        verbose=verbose,
    )

    return EarthEscapeResult(
        windows=[_to_escape_window(w) for w in rh.windows],
        final_state=rh.final_state,
        total_delta_v=rh.total_delta_v,
        total_burns=rh.total_burns,
        total_tof=rh.total_tof,
        reached_target=rh.terminated,
        full_schedule=rh.full_schedule,
    )
