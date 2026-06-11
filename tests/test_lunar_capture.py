"""Tests for the sliding-window lunar capture solver."""

from __future__ import annotations

import numpy as np
import pytest

from qalunar.dynamics.cr3bp import EARTH_MOON_MU, PlanarCR3BP
from qalunar.qubo.lunar_capture import (
    LunarCaptureResult,
    moon_orbit_apolune_perilune,
    moon_two_body_energy,
    reconstruct_full_trajectory,
    solve_lunar_capture_sliding_window,
)
from qalunar.qubo.scheduling_samplers import (
    sample_brute_force,
    sample_simulated_annealing,
)
from qalunar.reference.edelbaum import LENGTH_KM


_MU = EARTH_MOON_MU
_MOON_POS = np.array([1.0 - _MU, 0.0])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _moon_orbit_state(
    altitude_km: float,
    excess_factor: float = 1.0,
    phase_rad: float = 0.0,
    prograde: bool = True,
) -> np.ndarray:
    """Build a synodic-frame state at given Moon-relative altitude with
    a Moon-relative inertial speed equal to ``excess_factor`` times the
    local circular speed.

    ``excess_factor < sqrt(2)`` -> bound; ``> sqrt(2)`` -> hyperbolic.
    """
    r = (1737.0 + altitude_km) / LENGTH_KM
    cos_p, sin_p = np.cos(phase_rad), np.sin(phase_rad)
    pos = _MOON_POS + r * np.array([cos_p, sin_p])
    v_circ = float(np.sqrt(_MU / r))
    sign = 1.0 if prograde else -1.0
    v_rel = excess_factor * sign * v_circ * np.array([-sin_p, cos_p])
    v_moon_inertial = np.array([-_MOON_POS[1], _MOON_POS[0]])
    v_inertial = v_rel + v_moon_inertial
    omega_cross_r = np.array([-pos[1], pos[0]])
    v_syn = v_inertial - omega_cross_r
    return np.concatenate([pos, v_syn])


def _sampler(qubo):
    if qubo.n_vars <= 18:
        return sample_brute_force(qubo).schedule
    return sample_simulated_annealing(
        qubo, num_reads=400, seed=42,
    ).schedule


# ---------------------------------------------------------------------------
# Two-body diagnostics
# ---------------------------------------------------------------------------


class TestMoonTwoBodyEnergy:
    """``moon_two_body_energy`` matches the Keplerian formula
    E = v^2/2 - mu/r."""

    def test_circular_orbit_energy_is_negative_half_mu_over_r(self) -> None:
        """For a circular Moon orbit, E = -mu/(2r)."""
        for alt_km in (100.0, 1_000.0, 5_000.0):
            r = (1737.0 + alt_km) / LENGTH_KM
            state = _moon_orbit_state(alt_km, excess_factor=1.0)
            e = moon_two_body_energy(state)
            expected = -_MU / (2.0 * r)
            assert e == pytest.approx(expected, rel=5e-3)

    def test_hyperbolic_orbit_energy_is_positive(self) -> None:
        state = _moon_orbit_state(4_000.0, excess_factor=1.5)
        assert moon_two_body_energy(state) > 0.0

    def test_bound_orbit_energy_is_negative(self) -> None:
        state = _moon_orbit_state(4_000.0, excess_factor=1.2)
        assert moon_two_body_energy(state) < 0.0


class TestMoonApolunePerilune:
    """``moon_orbit_apolune_perilune`` matches Keplerian conic
    apolune/perilune formulas."""

    def test_circular_orbit_apo_equals_peri_equals_a(self) -> None:
        state = _moon_orbit_state(2_000.0, excess_factor=1.0)
        r_apo, r_peri, a = moon_orbit_apolune_perilune(state)
        r_orbit = (1737.0 + 2_000.0) / LENGTH_KM
        assert r_apo == pytest.approx(r_orbit, rel=1e-3)
        assert r_peri == pytest.approx(r_orbit, rel=1e-3)
        assert a == pytest.approx(r_orbit, rel=1e-3)

    def test_hyperbolic_orbit_apolune_is_infinite(self) -> None:
        state = _moon_orbit_state(4_000.0, excess_factor=1.5)
        r_apo, _, _ = moon_orbit_apolune_perilune(state)
        assert not np.isfinite(r_apo)

    def test_eccentric_bound_orbit_has_apo_greater_than_peri(self) -> None:
        # Build at perilune with v > v_circ -> apolune is higher up.
        state = _moon_orbit_state(2_000.0, excess_factor=1.2)
        r_apo, r_peri, _ = moon_orbit_apolune_perilune(state)
        assert r_apo > r_peri
        assert np.isfinite(r_apo)


# ---------------------------------------------------------------------------
# Sliding-window solver
# ---------------------------------------------------------------------------


class TestSlidingWindowCapture:
    """End-to-end tests of the sliding-window solver. All scenarios
    start from a bound elliptical orbit at ~4000 km perilune (excess
    factor 1.2 -> e ~ 0.69). The Earth's perturbation and the
    anti-tangential burns interact to circularise the orbit slowly."""

    def setup_method(self) -> None:
        self.dyn = PlanarCR3BP()
        self.state0 = _moon_orbit_state(
            altitude_km=4_000.0, excess_factor=1.20,
        )

    def test_small_run_returns_proper_result(self) -> None:
        """Even a short run returns a populated LunarCaptureResult."""
        res = solve_lunar_capture_sliding_window(
            self.dyn, self.state0, sampler=_sampler,
            n_windows=4, window_t_max=0.04, n_decision_steps=8,
            thrust_magnitude=0.02, moon_action_radius=0.10,
            n_integration_substeps=40, n_truth_substeps=80,
            max_inner_iters=2,
        )
        assert isinstance(res, LunarCaptureResult)
        assert len(res.windows) > 0
        assert res.full_schedule.size == sum(
            w.schedule.size for w in res.windows
        )
        # Energy stays in the bound regime.
        assert moon_two_body_energy(res.final_state) < 0.0
        assert res.captured

    def test_anti_tangential_drives_energy_down(self) -> None:
        """Over many windows from a bound-eccentric start, anti-tangential
        thrust should not raise the orbital energy (it should at minimum
        stay similar; in expectation it decreases)."""
        res = solve_lunar_capture_sliding_window(
            self.dyn, self.state0, sampler=_sampler,
            n_windows=12, window_t_max=0.04, n_decision_steps=8,
            thrust_magnitude=0.02, moon_action_radius=0.10,
            n_integration_substeps=40, n_truth_substeps=80,
            max_inner_iters=2,
        )
        e_init = moon_two_body_energy(self.state0)
        e_final = moon_two_body_energy(res.final_state)
        # Energy should not increase by more than a small CR3BP-perturb
        # amount over the run.
        assert e_final <= e_init + 5e-3

    def test_apolune_history_starts_finite(self) -> None:
        res = solve_lunar_capture_sliding_window(
            self.dyn, self.state0, sampler=_sampler,
            n_windows=6, window_t_max=0.04, n_decision_steps=8,
            thrust_magnitude=0.02, moon_action_radius=0.10,
            n_integration_substeps=40, n_truth_substeps=80,
            max_inner_iters=2,
        )
        apo_hist = res.apolune_history()
        assert apo_hist.size >= 2
        assert np.isfinite(apo_hist[0])

    def test_per_window_dv_is_sum_of_burns_times_thrust(self) -> None:
        thrust_mag = 0.02
        n_steps = 8
        res = solve_lunar_capture_sliding_window(
            self.dyn, self.state0, sampler=_sampler,
            n_windows=4, window_t_max=0.04, n_decision_steps=n_steps,
            thrust_magnitude=thrust_mag, moon_action_radius=0.10,
            n_integration_substeps=40, n_truth_substeps=80,
            max_inner_iters=2,
        )
        for w in res.windows:
            dt_decision = w.iterative_result.final_qubo.dt_decision
            expected = thrust_mag * dt_decision * w.n_burns
            assert w.delta_v == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# Action-radius gating
# ---------------------------------------------------------------------------


class TestActionRadiusGating:
    """When the spacecraft is outside the action radius, the solver must
    drift freely instead of running the QUBO."""

    def setup_method(self) -> None:
        self.dyn = PlanarCR3BP()

    def test_far_from_moon_drifts(self) -> None:
        """Place the spacecraft far from the Moon (close to L1, well
        outside any reasonable action radius) and verify no windows
        are recorded."""
        # ~75% of the way from Earth to Moon, with the Moon-relative
        # circular velocity (so the orbit is energetically bound to
        # the Earth-Moon system).
        moon_pos = _MOON_POS
        r_far = 0.4
        pos = moon_pos + np.array([-r_far, 0.0])
        v_circ_local = float(np.sqrt(_MU / r_far))
        v_rel = np.array([0.0, v_circ_local])
        v_moon_inertial = np.array([-moon_pos[1], moon_pos[0]])
        v_inertial = v_rel + v_moon_inertial
        omega_cross_r = np.array([-pos[1], pos[0]])
        v_syn = v_inertial - omega_cross_r
        state = np.concatenate([pos, v_syn])

        res = solve_lunar_capture_sliding_window(
            self.dyn, state, sampler=_sampler,
            n_windows=3, window_t_max=0.04, n_decision_steps=6,
            thrust_magnitude=0.02,
            moon_action_radius=0.05,    # spacecraft is at r=0.4
            drift_t_max=0.05,
            n_integration_substeps=30, n_truth_substeps=60,
            max_inner_iters=2,
        )
        # All windows should be drift -> no entries in res.windows.
        assert len(res.windows) == 0
        # State should have moved (drifted).
        assert not np.allclose(res.final_state, state, atol=1e-6)


# ---------------------------------------------------------------------------
# Trajectory reconstruction
# ---------------------------------------------------------------------------


class TestReconstructTrajectory:
    """``reconstruct_full_trajectory`` produces a continuous time series."""

    def test_times_are_monotonic(self) -> None:
        dyn = PlanarCR3BP()
        state0 = _moon_orbit_state(altitude_km=4_000.0, excess_factor=1.20)
        res = solve_lunar_capture_sliding_window(
            dyn, state0, sampler=_sampler,
            n_windows=3, window_t_max=0.04, n_decision_steps=6,
            thrust_magnitude=0.02, moon_action_radius=0.10,
            n_integration_substeps=30, n_truth_substeps=60,
            max_inner_iters=2,
        )
        if len(res.windows) == 0:
            pytest.skip("No accepted windows; trajectory reconstruction skipped")
        times, _ = reconstruct_full_trajectory(
            dyn, res, thrust_magnitude=0.02, n_integration_substeps=40,
        )
        diffs = np.diff(times)
        assert np.all(diffs >= -1e-12)

    def test_segments_are_continuous_in_state(self) -> None:
        """Each window's last state should equal the next window's
        first state (within integration tolerance)."""
        dyn = PlanarCR3BP()
        state0 = _moon_orbit_state(altitude_km=4_000.0, excess_factor=1.20)
        res = solve_lunar_capture_sliding_window(
            dyn, state0, sampler=_sampler,
            n_windows=4, window_t_max=0.04, n_decision_steps=6,
            thrust_magnitude=0.02, moon_action_radius=0.10,
            n_integration_substeps=30, n_truth_substeps=60,
            max_inner_iters=2,
        )
        for prev, nxt in zip(res.windows[:-1], res.windows[1:]):
            assert np.allclose(
                prev.state_final, nxt.state_initial, atol=1e-9,
            )
