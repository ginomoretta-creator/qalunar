"""Phase-agnostic receding-horizon QUBO driver in the full planar CR3BP.

Phases 1 and 3 of the mission were originally written as two separate
sliding-window solvers -- :mod:`qalunar.qubo.earth_escape` (raise the
Earth-relative semi-major axis with **tangential** thrust until a parking
orbit) and :mod:`qalunar.qubo.lunar_capture` (dissipate Moon-relative energy
with **anti-tangential** thrust until the orbit is bound). They are near mirror
images: the only essential differences are

  * which body the per-window target is built around (Earth vs Moon),
  * the thrust direction (tangential to inject energy vs anti-tangential to
    dissipate it), and
  * the per-window target speed (``factor * v_circ`` with ``factor > 1`` to
    boost, ``factor < 1`` (or ``=1``) to brake), and
  * the stop condition.

This module factors out everything they share into a single driver,
:func:`solve_receding_horizon`, parameterised by a :class:`CentralBody` and a
per-window *target factor*, thrust direction, and termination predicate. The
two phase solvers in :mod:`earth_escape` and :mod:`lunar_capture` are thin
adapters over this driver, and any future phase (e.g. a mid-cruise station-keep)
can reuse it by supplying its own body/target/termination.

The receding-horizon structure -- short windows, each linearised and solved by
the iterative QUBO of the paper, chained through the true propagated state -- is
what makes the full-CR3BP, many-revolution problem tractable: a single global
linearisation cannot survive a large-excursion climb or a hyperbolic capture,
but a fraction-of-a-revolution window can.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from qalunar.dynamics.cr3bp import EARTH_MOON_MU, PlanarCR3BP
from qalunar.qubo.thrust_scheduling import (
    IterativeSchedulingResult,
    ThrustSchedulingConfig,
    solve_iterative,
)


__all__ = [
    "CentralBody",
    "RecedingHorizonWindow",
    "RecedingHorizonResult",
    "solve_receding_horizon",
]


# ---------------------------------------------------------------------------
# Central body: one object that knows where a primary sits and its GM, and can
# build all the two-body diagnostics and the per-window circular-speed target
# relative to it. Earth and Moon are the two instances used by the phases.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CentralBody:
    """A CR3BP primary used as the reference for per-window targeting.

    Attributes
    ----------
    position : (2,) ndarray
        Synodic-frame position of the body (Earth at ``(-mu, 0)``, Moon at
        ``(1 - mu, 0)``).
    gm : float
        Gravitational parameter of the body in CR3BP nondimensional units
        (Earth ``1 - mu``, Moon ``mu``).
    name : str
    """

    position: NDArray[np.float64]
    gm: float
    name: str

    @classmethod
    def earth(cls, mu: float = EARTH_MOON_MU) -> "CentralBody":
        return cls(np.array([-mu, 0.0]), 1.0 - mu, "Earth")

    @classmethod
    def moon(cls, mu: float = EARTH_MOON_MU) -> "CentralBody":
        return cls(np.array([1.0 - mu, 0.0]), mu, "Moon")

    # -- frame conversions / diagnostics ----------------------------------

    def relative_inertial(
        self, state_synodic: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Body-relative position and inertial velocity from a synodic state."""
        state = np.asarray(state_synodic, dtype=np.float64)
        r = state[:2]
        v_syn = state[2:4]
        omega_cross_r = np.array([-r[1], r[0]])
        v_inertial = v_syn + omega_cross_r
        v_body_inertial = np.array([-self.position[1], self.position[0]])
        r_rel = r - self.position
        v_rel = v_inertial - v_body_inertial
        return r_rel, v_rel

    def distance(self, state_synodic: NDArray[np.float64]) -> float:
        r_rel, _ = self.relative_inertial(state_synodic)
        return float(np.linalg.norm(r_rel))

    def two_body_energy(self, state_synodic: NDArray[np.float64]) -> float:
        """Specific two-body orbital energy w.r.t. this body.

        Negative -> bound; zero -> parabolic; positive -> hyperbolic.
        """
        r_rel, v_rel = self.relative_inertial(state_synodic)
        return float(0.5 * np.dot(v_rel, v_rel) - self.gm / np.linalg.norm(r_rel))

    def apoapsis_periapsis_sma(
        self, state_synodic: NDArray[np.float64]
    ) -> tuple[float, float, float]:
        """Two-body ``(r_apo, r_peri, a)`` w.r.t. this body (nondim).

        For an unbound state ``a`` and ``r_apo`` are ``+inf``; ``r_peri`` is
        still well defined while angular momentum is non-zero.
        """
        r_rel, v_rel = self.relative_inertial(state_synodic)
        gm = self.gm
        r = float(np.linalg.norm(r_rel))
        v_sq = float(np.dot(v_rel, v_rel))
        energy = 0.5 * v_sq - gm / r
        h = float(r_rel[0] * v_rel[1] - r_rel[1] * v_rel[0])
        # Eccentricity is finite for every conic (e >= 1 when unbound),
        # so the periapsis r_peri = p / (1 + e) is always well defined.
        e = float(np.sqrt(max(1.0 + 2.0 * energy * h ** 2 / gm ** 2, 0.0)))
        if energy >= 0.0:
            a = float("inf")
            r_apo = float("inf")
        else:
            a = -gm / (2.0 * energy)
            r_apo = a * (1.0 + e)
        r_peri = (h ** 2 / gm) / (1.0 + e)
        return r_apo, r_peri, a

    def circular_period(self, radius: float) -> float:
        """Local two-body circular period at ``radius``."""
        return 2.0 * np.pi * np.sqrt(radius ** 3 / self.gm)

    def circular_speed_target(
        self,
        state_synodic: NDArray[np.float64],
        factor: float,
        target_radius: float | None = None,
    ) -> NDArray[np.float64]:
        """Target state: same position, body-relative speed ``factor*v_circ``.

        The target body-relative inertial velocity points along the current
        velocity direction with magnitude ``factor * v_circ(ref_radius)``. With
        ``factor > 1`` and a *tangential* channel the QUBO fires to raise the
        speed (energy injection, the Earth-escape boost); with ``factor <= 1``
        and an *anti-tangential* channel it fires where the spacecraft moves
        faster than the target (energy dissipation, the lunar-capture brake).

        ``target_radius`` overrides the reference radius used for ``v_circ``;
        ``None`` uses the current body-relative radius.
        """
        r_rel, v_rel = self.relative_inertial(state_synodic)
        r_now = float(np.linalg.norm(r_rel))
        v_norm = float(np.linalg.norm(v_rel))
        ref_radius = target_radius if target_radius is not None else r_now
        v_circ = float(np.sqrt(self.gm / ref_radius))

        v_dir = v_rel / v_norm if v_norm > 1e-15 else np.array([0.0, 1.0])
        target_v_rel_inertial = factor * v_circ * v_dir

        v_body_inertial = np.array([-self.position[1], self.position[0]])
        target_v_inertial = target_v_rel_inertial + v_body_inertial

        pos_syn = np.asarray(state_synodic, dtype=np.float64)[:2]
        omega_cross_r = np.array([-pos_syn[1], pos_syn[0]])
        target_v_syn = target_v_inertial - omega_cross_r
        return np.concatenate([pos_syn, target_v_syn])


# ---------------------------------------------------------------------------
# Generic result containers
# ---------------------------------------------------------------------------


@dataclass
class RecedingHorizonWindow:
    """One window of the generic receding-horizon driver.

    Carries the union of the diagnostics the phase-specific windows need:
    two-body energy, apoapsis/periapsis/SMA (all w.r.t. the driver's
    :class:`CentralBody`) before and after the window, the schedule, and the
    underlying iterative solve.
    """

    index: int
    state_initial: NDArray[np.float64]
    state_final: NDArray[np.float64]
    schedule: NDArray[np.int64]
    n_burns: int
    delta_v: float
    energy_initial: float
    energy_final: float
    apo_initial: float
    apo_final: float
    peri_initial: float
    peri_final: float
    sma_initial: float
    sma_final: float
    predicted_miss_norm: float
    true_miss_norm: float
    iterative_result: IterativeSchedulingResult


@dataclass
class RecedingHorizonResult:
    """Concatenated result of the generic receding-horizon driver."""

    body_name: str = ""
    windows: list[RecedingHorizonWindow] = field(default_factory=list)
    final_state: NDArray[np.float64] = field(default_factory=lambda: np.zeros(4))
    total_delta_v: float = 0.0
    total_burns: int = 0
    total_tof: float = 0.0
    terminated: bool = False
    full_schedule: NDArray[np.int64] = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------


def solve_receding_horizon(
    dynamics: PlanarCR3BP,
    state0_synodic: NDArray[np.float64],
    sampler,
    *,
    body: CentralBody,
    thrust_direction: str,
    target_factor: float,
    thrust_magnitude: float,
    n_windows: int,
    window_revs: float,
    window_t_max: float | None = None,
    n_decision_steps: int = 12,
    target_radius: float | None = None,
    target_position_weight: float = 0.0,
    target_velocity_weight: float = 1.0,
    fuel_weight: float = 0.0,
    action_radius: float | None = None,
    drift_t_max: float = 0.5,
    terminate=None,
    n_integration_substeps: int = 60,
    n_truth_substeps: int = 200,
    max_inner_iters: int = 4,
    verbose: bool = False,
) -> RecedingHorizonResult:
    """Chain short QUBO windows in the full CR3BP toward a body-relative goal.

    Each window builds a target of "same unforced end-of-window position with a
    body-relative speed of ``target_factor * v_circ``" via
    :meth:`CentralBody.circular_speed_target`, runs the iterative re-linearised
    QUBO over the window, and advances the true state. Windows whose start lies
    outside ``action_radius`` of the body (if given) are skipped with a free
    drift.

    Parameters
    ----------
    body : CentralBody
        Reference primary for targeting and diagnostics.
    thrust_direction : {"tangential", "antitangential"}
        Single-channel thrust direction passed to the QUBO.
    target_factor : float
        Multiple of local circular speed used as the per-window target
        (``> 1`` boosts/raises energy, ``<= 1`` brakes/dissipates).
    n_windows : int
        Maximum number of windows.
    window_revs : float
        Window length as a fraction of the local body-relative circular period.
    window_t_max : float, optional
        Hard cap on a window's time length (essential once the orbit loosens
        and the period explodes).
    action_radius : float, optional
        If set, windows starting farther than this from the body free-drift
        (the local "circular" target is meaningless far from the body).
    drift_t_max : float
        Max time of a single free-drift segment.
    terminate : callable, optional
        ``terminate(state, body) -> bool`` checked at the start of each window;
        when it returns ``True`` the loop stops and ``result.terminated`` is set.
    verbose : bool

    Returns
    -------
    RecedingHorizonResult
    """
    if thrust_direction not in ("tangential", "antitangential"):
        raise ValueError(
            f"thrust_direction must be 'tangential' or 'antitangential', "
            f"got {thrust_direction!r}"
        )

    state = np.asarray(state0_synodic, dtype=np.float64).copy()
    moon_or_body_pos = body.position
    result = RecedingHorizonResult(body_name=body.name)
    schedule_list: list[NDArray[np.int64]] = []
    total_tof = 0.0

    target_weights = np.array([
        target_position_weight, target_position_weight,
        target_velocity_weight, target_velocity_weight,
    ])

    for w_idx in range(n_windows):
        # ----- Termination check (start-of-window) -----
        if terminate is not None and terminate(state, body):
            result.terminated = True
            break

        r_now = body.distance(state)

        # ----- Free drift when far from the body -----
        if action_radius is not None and r_now > action_radius:
            n_drift = 1_000
            t_drift, drift_traj = dynamics.propagate(
                state, (0.0, drift_t_max), n_steps=n_drift,
            )
            r_drift = np.linalg.norm(
                drift_traj[:, :2] - moon_or_body_pos, axis=1,
            )
            mask_inside = r_drift <= action_radius
            if np.any(mask_inside):
                idx_in = int(np.argmax(mask_inside))
                state = drift_traj[idx_in]
                total_tof += float(t_drift[idx_in])
            else:
                state = drift_traj[-1]
                total_tof += drift_t_max
            if verbose:
                print(f"  window {w_idx:3d}: drift "
                      f"(r={r_now:.3f} > {action_radius:.3f})")
            continue

        # ----- Window length: fraction of local period, capped -----
        t_window = window_revs * body.circular_period(r_now)
        if window_t_max is not None:
            t_window = min(t_window, window_t_max)
        t_span = (0.0, t_window)

        # ----- Diagnostics at window start -----
        e_init = body.two_body_energy(state)
        apo_init, peri_init, sma_init = body.apoapsis_periapsis_sma(state)

        # ----- Target = unforced end-of-window position, scaled speed -----
        _, traj_unforced = dynamics.propagate(state, t_span, n_steps=400)
        end_state = traj_unforced[-1]
        target_state = body.circular_speed_target(
            end_state, factor=target_factor, target_radius=target_radius,
        )

        cfg = ThrustSchedulingConfig(
            thrust_magnitude=thrust_magnitude,
            thrust_direction=thrust_direction,
            fuel_weight=fuel_weight,
            target_weights=target_weights,
        )
        try:
            iter_res = solve_iterative(
                dynamics, state, target_state, t_span,
                n_decision_steps=n_decision_steps,
                sampler=sampler, config=cfg,
                n_integration_substeps=n_integration_substeps,
                n_truth_substeps=n_truth_substeps,
                max_iters=max_inner_iters,
            )
        except np.linalg.LinAlgError:
            if verbose:
                print(f"  window {w_idx:3d}: STM singular -> coast fallback")
            _, traj_seg = dynamics.propagate(
                state, t_span, n_steps=n_truth_substeps,
            )
            state = traj_seg[-1]
            total_tof += t_window
            continue

        new_state = iter_res.true_final_state
        n_burns = int(iter_res.schedule.sum())
        dv_window = float(iter_res.final_qubo.delta_v(iter_res.schedule))
        e_final = body.two_body_energy(new_state)
        apo_final, peri_final, sma_final = body.apoapsis_periapsis_sma(new_state)
        predicted_miss = float(np.linalg.norm(
            iter_res.final_qubo.miss_distance(iter_res.schedule)))
        true_miss = (
            iter_res.true_miss_norm_history[-1]
            if iter_res.true_miss_norm_history else float("nan")
        )

        result.windows.append(RecedingHorizonWindow(
            index=w_idx,
            state_initial=state.copy(),
            state_final=new_state.copy(),
            schedule=iter_res.schedule.copy(),
            n_burns=n_burns,
            delta_v=dv_window,
            energy_initial=e_init,
            energy_final=e_final,
            apo_initial=apo_init,
            apo_final=apo_final,
            peri_initial=peri_init,
            peri_final=peri_final,
            sma_initial=sma_init,
            sma_final=sma_final,
            predicted_miss_norm=predicted_miss,
            true_miss_norm=true_miss,
            iterative_result=iter_res,
        ))
        schedule_list.append(iter_res.schedule.copy())
        total_tof += t_window

        if verbose:
            apo_str = f"{apo_final:.4f}" if np.isfinite(apo_final) else "inf"
            sma_str = f"{sma_final:.4f}" if np.isfinite(sma_final) else "inf"
            print(
                f"  window {w_idx:3d}: "
                f"e {e_init:+.3e} -> {e_final:+.3e}, "
                f"sma {sma_init:.4f} -> {sma_str}, apo={apo_str}, "
                f"burns={n_burns}/{n_decision_steps}, dV={dv_window:.3e}"
            )

        state = new_state

    result.final_state = state
    result.total_delta_v = float(sum(w.delta_v for w in result.windows))
    result.total_burns = int(sum(w.n_burns for w in result.windows))
    result.total_tof = float(total_tof)
    result.full_schedule = (
        np.concatenate(schedule_list) if schedule_list
        else np.zeros(0, dtype=np.int64)
    )
    return result
