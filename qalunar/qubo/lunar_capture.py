"""Sliding-window QUBO scheduler for lunar capture and stabilisation.

The Phase-2 cislunar correction QUBO produces a hyperbolic flyby state
near the Moon: the spacecraft passes close to the Moon but does not
become bound to it. To convert that flyby into a closed orbit using
on/off binary thrust, a single QUBO arc is not enough -- the
linearisation gap over the many revolutions needed to dissipate the
hyperbolic excess velocity is far too wide.

This module implements a *sliding-window* QUBO. Each window covers a
short arc (a fraction of a Moon-relative revolution); within the
window the iterative re-linearisation driver of the paper applies, and
between windows the spacecraft state is propagated and used as the
input to the next window. The target for each window is constructed
from the spacecraft's current Moon-relative state: the QUBO is asked
to drive the spacecraft toward a circular orbit at its current radius
(velocity-only weighting), which the anti-tangential binary thrust
achieves by firing during fast-moving portions of the orbit.

Implementation note
-------------------
Since the "QUBO everywhere" upgrade, the actual window-chaining loop lives in
the phase-agnostic :func:`qalunar.qubo.receding_horizon.solve_receding_horizon`.
This module is now a thin adapter: it picks the Moon as the
:class:`~qalunar.qubo.receding_horizon.CentralBody`, an anti-tangential channel,
and a braking target speed (``brake_factor <= 1``), then maps the generic
windows back onto the lunar-capture result type used throughout the codebase.
The Earth-escape Phase 1 (:mod:`qalunar.qubo.earth_escape`) is the dual adapter
over the same driver.

Capture is judged from the spacecraft's two-body energy with respect to the
Moon *and* its distance: a negative Keplerian energy is only meaningful
inside the lunar Hill sphere, so ``captured`` requires both. The loop runs
the full window budget unless ``energy_tol`` is given, in which case it
stops as soon as the Moon-relative energy drops below that threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from qalunar.dynamics.cr3bp import EARTH_MOON_MU, PlanarCR3BP
from qalunar.qubo.receding_horizon import (
    CentralBody,
    RecedingHorizonSegment,
    RecedingHorizonWindow,
    solve_receding_horizon,
)
from qalunar.qubo.thrust_scheduling import (
    IterativeSchedulingResult,
    ThrustSchedulingConfig,
    propagate_schedule,
)


__all__ = [
    "LunarCaptureWindow",
    "LunarCaptureResult",
    "solve_lunar_capture_sliding_window",
    "moon_two_body_energy",
    "moon_orbit_apolune_perilune",
    "moon_hill_radius",
    "is_captured",
]


# ---------------------------------------------------------------------------
# Orbital diagnostics relative to the Moon (delegated to CentralBody so the
# Earth and Moon diagnostics share one implementation).
# ---------------------------------------------------------------------------


def moon_two_body_energy(
    state_synodic: NDArray[np.float64],
    mu: float = EARTH_MOON_MU,
) -> float:
    """Specific orbital energy of the spacecraft w.r.t. the Moon.

    Negative -> bound orbit; zero -> parabolic; positive -> hyperbolic.
    """
    return CentralBody.moon(mu).two_body_energy(state_synodic)


def moon_hill_radius(mu: float = EARTH_MOON_MU) -> float:
    """Lunar Hill radius in nondimensional units, ``(mu/3)^(1/3)``
    (about 0.159 LU = 61,500 km)."""
    return float((mu / 3.0) ** (1.0 / 3.0))


def is_captured(
    state_synodic: NDArray[np.float64],
    mu: float = EARTH_MOON_MU,
    distance_limit: float | None = None,
) -> bool:
    """Bound to the Moon: negative two-body energy *and* inside the Hill
    sphere (or ``distance_limit``). A bare energy sign test returns True
    for states far outside the region where the Moon-relative Keplerian
    elements mean anything."""
    moon = CentralBody.moon(mu)
    limit = moon_hill_radius(mu) if distance_limit is None else distance_limit
    return bool(
        moon.two_body_energy(state_synodic) < 0.0
        and moon.distance(state_synodic) < limit
    )


def moon_orbit_apolune_perilune(
    state_synodic: NDArray[np.float64],
    mu: float = EARTH_MOON_MU,
) -> tuple[float, float, float]:
    """Two-body apolune, perilune, and semi-major axis around the Moon.

    For an unbound state the apolune is reported as ``+inf``. The
    perilune and semi-major axis are still well defined as long as
    angular momentum is non-zero.
    """
    return CentralBody.moon(mu).apoapsis_periapsis_sma(state_synodic)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class LunarCaptureWindow:
    """One sliding-window step of the lunar-capture solver."""

    index: int
    state_initial: NDArray[np.float64]
    state_final: NDArray[np.float64]
    schedule: NDArray[np.int64]
    n_burns: int
    delta_v: float
    energy_initial: float
    energy_final: float
    apolune_initial: float
    apolune_final: float
    iterative_result: IterativeSchedulingResult


@dataclass
class LunarCaptureResult:
    """Concatenated result of the sliding-window capture solver.

    Attributes
    ----------
    windows : list of LunarCaptureWindow
    final_state : (4,) ndarray
        Synodic-frame state after the last window.
    total_delta_v : float
        Total nondimensional Δv across all windows.
    total_burns : int
        Total number of active burn slots.
    captured : bool
        True if the spacecraft ended bound to the Moon: energy < 0 *and*
        inside the lunar Hill sphere (see :func:`is_captured`).
    full_schedule : (M_total,) ndarray
        Concatenation of all per-window schedules.
    total_tof : float
        Elapsed time including free-drift and coast-fallback arcs (nondim).
    segments : list of RecedingHorizonSegment
        Every flown piece in order (QUBO windows, drifts, coasts).
    """

    windows: list[LunarCaptureWindow] = field(default_factory=list)
    final_state: NDArray[np.float64] = field(default_factory=lambda: np.zeros(4))
    total_delta_v: float = 0.0
    total_burns: int = 0
    captured: bool = False
    full_schedule: NDArray[np.int64] = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )
    total_tof: float = 0.0
    segments: list[RecedingHorizonSegment] = field(default_factory=list)

    def energy_history(self) -> NDArray[np.float64]:
        """Two-body energy at each window boundary."""
        if not self.windows:
            return np.array([])
        e = [self.windows[0].energy_initial]
        for w in self.windows:
            e.append(w.energy_final)
        return np.array(e)

    def apolune_history(self) -> NDArray[np.float64]:
        if not self.windows:
            return np.array([])
        a = [self.windows[0].apolune_initial]
        for w in self.windows:
            a.append(w.apolune_final)
        return np.array(a)


def _to_capture_window(w: RecedingHorizonWindow) -> LunarCaptureWindow:
    """Adapt a generic driver window onto the lunar-capture window type."""
    return LunarCaptureWindow(
        index=w.index,
        state_initial=w.state_initial,
        state_final=w.state_final,
        schedule=w.schedule,
        n_burns=w.n_burns,
        delta_v=w.delta_v,
        energy_initial=w.energy_initial,
        energy_final=w.energy_final,
        apolune_initial=w.apo_initial,
        apolune_final=w.apo_final,
        iterative_result=w.iterative_result,
    )


# ---------------------------------------------------------------------------
# Main solver (thin adapter over the phase-agnostic driver)
# ---------------------------------------------------------------------------


def solve_lunar_capture_sliding_window(
    dynamics: PlanarCR3BP,
    state0_synodic: NDArray[np.float64],
    sampler,
    *,
    n_windows: int = 30,
    window_revs: float = 1.0,
    window_t_max: float | None = None,
    n_decision_steps: int = 12,
    thrust_magnitude: float = 0.02,
    target_position_weight: float = 0.0,
    target_velocity_weight: float = 1.0,
    fuel_weight: float = 0.0,
    brake_factor: float = 0.95,
    target_radius: float | None = None,
    moon_action_radius: float | None = None,
    drift_t_max: float = 0.5,
    n_integration_substeps: int = 60,
    n_truth_substeps: int = 200,
    max_inner_iters: int = 4,
    energy_tol: float | None = None,
    verbose: bool = False,
) -> LunarCaptureResult:
    """Capture a hyperbolic flyby into a stable lunar orbit using
    binary anti-tangential thrust over multiple sliding windows.

    At each window the QUBO target is "circular Moon-relative velocity
    in the current direction at the unforced end-of-window position".
    The QUBO with single anti-tangential channel naturally fires
    during fast-moving portions of the orbit (near perilune of an
    elliptical pass) and dissipates energy.

    Parameters
    ----------
    dynamics : PlanarCR3BP
    state0_synodic : (4,) ndarray
        Initial synodic-frame state. Should be near the Moon (e.g. at
        the perilune of a Phase-2 corrected flyby).
    sampler : callable
        QUBO sampler ``(qubo) -> bitstring``. Same interface as
        :func:`qalunar.qubo.thrust_scheduling.solve_iterative`.
    n_windows : int
        Maximum number of sliding windows to run.
    window_revs : float
        Window length expressed as a fraction of the local Moon-relative
        circular period at the spacecraft's current radius. Smaller
        values keep the linearisation faithful but require more windows.
    n_decision_steps : int
        Per-window QUBO size (qubits).
    thrust_magnitude : float
        Anti-tangential thrust acceleration (CR3BP nondim).
    target_position_weight, target_velocity_weight : float
        Diagonal entries of the target weighting ``W``. Position is
        usually unweighted (``0``) so the QUBO does not chase
        position, only modulates velocity.
    fuel_weight : float
        Fuel penalty in the QUBO objective.
    brake_factor : float
        Target speed as a multiple of local circular speed (``<= 1``).
    moon_action_radius : float, optional
        Skip QUBO scheduling (free-drift) when the spacecraft is farther
        than this from the Moon, where a Moon-relative circular target is
        meaningless.
    drift_t_max : float
        Max time of one free-drift segment.
    n_integration_substeps, n_truth_substeps : int
        Pass-through to ``solve_iterative``.
    max_inner_iters : int
        ``max_iters`` passed to each window's ``solve_iterative`` call.
    energy_tol : float or None
        Stop braking once the Moon-relative two-body energy is below this
        value (e.g. ``-1e-5``). ``None`` (default) runs the full window
        budget, which keeps braking after the orbit is already bound.
    verbose : bool

    Returns
    -------
    LunarCaptureResult
    """
    moon = CentralBody.moon(dynamics.mu)
    terminate = None
    if energy_tol is not None:
        def terminate(state, body, _tol=float(energy_tol)):
            return bool(body.two_body_energy(state) < _tol)
    rh = solve_receding_horizon(
        dynamics, state0_synodic, sampler,
        body=moon,
        thrust_direction="antitangential",
        target_factor=brake_factor,
        thrust_magnitude=thrust_magnitude,
        n_windows=n_windows,
        window_revs=window_revs,
        window_t_max=window_t_max,
        n_decision_steps=n_decision_steps,
        target_radius=target_radius,
        target_position_weight=target_position_weight,
        target_velocity_weight=target_velocity_weight,
        fuel_weight=fuel_weight,
        action_radius=moon_action_radius,
        drift_t_max=drift_t_max,
        terminate=terminate,
        n_integration_substeps=n_integration_substeps,
        n_truth_substeps=n_truth_substeps,
        max_inner_iters=max_inner_iters,
        verbose=verbose,
    )

    result = LunarCaptureResult(
        windows=[_to_capture_window(w) for w in rh.windows],
        final_state=rh.final_state,
        total_delta_v=rh.total_delta_v,
        total_burns=rh.total_burns,
        captured=is_captured(rh.final_state, dynamics.mu),
        full_schedule=rh.full_schedule,
        total_tof=rh.total_tof,
        segments=list(rh.segments),
    )
    return result


def reconstruct_full_trajectory(
    dynamics: PlanarCR3BP,
    result: LunarCaptureResult,
    thrust_magnitude: float,
    n_integration_substeps: int = 200,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Concatenate the flown timeline into a single (t, x).

    Uses ``result.segments`` when available so free-drift and coast
    arcs are propagated too (no teleport across the action-radius gate);
    falls back to window-only concatenation for results built without
    segments.
    """
    times: list[NDArray[np.float64]] = []
    states: list[NDArray[np.float64]] = []
    t0 = 0.0
    cfg = ThrustSchedulingConfig(
        thrust_magnitude=thrust_magnitude,
        thrust_direction="antitangential",
        fuel_weight=0.0,
        target_weights=np.array([0.0, 0.0, 1.0, 1.0]),
    )
    pieces: list[tuple[NDArray[np.float64], float, NDArray[np.int64] | None]] = []
    if result.segments:
        for seg in result.segments:
            sched = (result.windows[seg.window_position].schedule
                     if seg.kind == "qubo" else None)
            pieces.append((seg.state_initial, seg.duration, sched))
    else:
        for w in result.windows:
            q = w.iterative_result.final_qubo
            pieces.append((w.state_initial, q.dt_decision * q.n_steps, w.schedule))

    for state_i, t_window, sched in pieces:
        if sched is None:
            t_seg, x_seg = dynamics.propagate(
                state_i, (0.0, t_window), n_steps=n_integration_substeps,
            )
        else:
            t_seg, x_seg = propagate_schedule(
                dynamics, state_i, (0.0, t_window),
                sched, cfg,
                n_integration_substeps=n_integration_substeps,
                return_trajectory=True,
            )
        if times:
            t_seg = t_seg + t0
            t_seg = t_seg[1:]
            x_seg = x_seg[1:]
        times.append(t_seg)
        states.append(x_seg)
        t0 = float(t_seg[-1])

    return np.concatenate(times), np.concatenate(states)
