"""Tests for the Phase 2 -> Phase 3 handoff helpers."""

from __future__ import annotations

import numpy as np
import pytest

from qalunar.dynamics.cr3bp import EARTH_MOON_MU, PlanarCR3BP
from qalunar.reference import (
    capture_delta_v,
    edelbaum_spiral_moon_descending,
    find_perilune,
    moon_relative_inertial,
)
from qalunar.reference.edelbaum import LENGTH_KM


_MU = EARTH_MOON_MU
_MOON_POS = np.array([1.0 - _MU, 0.0])


# ---------------------------------------------------------------------------
# Moon-relative inertial conversion
# ---------------------------------------------------------------------------


class TestMoonRelativeInertial:
    """Synodic -> Moon-relative inertial conversion is self-consistent."""

    def test_circular_lunar_orbit_has_circular_speed(self) -> None:
        """A spacecraft built on a circular Moon-relative orbit (using the
        same construction as the station-keeping demo) must have
        Moon-relative speed equal to ``sqrt(mu / r)``."""
        r_lunar = (1737.0 + 100.0) / LENGTH_KM
        # Build the synodic state by reversing moon_relative_inertial.
        # Place spacecraft at (+x relative to Moon) with prograde tangential
        # Moon-relative velocity.
        pos = _MOON_POS + np.array([r_lunar, 0.0])
        v_rel_inertial = np.array([0.0, np.sqrt(_MU / r_lunar)])
        v_moon_inertial = np.array([-_MOON_POS[1], _MOON_POS[0]])
        v_inertial = v_rel_inertial + v_moon_inertial
        omega_cross_r = np.array([-pos[1], pos[0]])
        v_syn = v_inertial - omega_cross_r
        state_syn = np.concatenate([pos, v_syn])

        r_rel, v_rel = moon_relative_inertial(state_syn, mu=_MU)
        assert np.linalg.norm(r_rel) == pytest.approx(r_lunar, rel=1e-12)
        assert np.linalg.norm(v_rel) == pytest.approx(
            np.sqrt(_MU / r_lunar), rel=1e-12
        )

    def test_position_relative_to_moon_is_difference(self) -> None:
        """``r_moon_rel = r_spacecraft - r_moon``."""
        state_syn = np.array([0.7, 0.1, 0.0, 0.0])
        r_rel, _ = moon_relative_inertial(state_syn, mu=_MU)
        expected = state_syn[:2] - _MOON_POS
        assert np.allclose(r_rel, expected, atol=1e-15)

    def test_zero_synodic_velocity_gives_pure_corotation(self) -> None:
        """If synodic velocity is zero, the spacecraft is at rest in the
        rotating frame -- in the inertial frame it co-rotates with
        ``omega x r``, and the Moon-relative inertial velocity is just
        ``omega x (r - r_moon)``."""
        r = np.array([0.7, 0.1])
        state_syn = np.concatenate([r, np.zeros(2)])
        _, v_rel = moon_relative_inertial(state_syn, mu=_MU)
        # omega x (r - r_moon) = (-(r-r_moon)_y, (r-r_moon)_x).
        expected = np.array([-(r[1] - 0.0), r[0] - (1.0 - _MU)])
        assert np.allclose(v_rel, expected, atol=1e-14)


# ---------------------------------------------------------------------------
# Perilune detection
# ---------------------------------------------------------------------------


class TestFindPerilune:
    """``find_perilune`` locates the closest approach correctly."""

    def setup_method(self) -> None:
        self.dyn = PlanarCR3BP()

    def test_circular_lunar_orbit_perilune_is_at_starting_radius(self) -> None:
        """On a (Keplerian-circular) Moon orbit, the CR3BP perilune over
        one period stays close to the starting radius. Earth's perturbation
        makes it slightly non-circular even when the construction is
        Keplerian-circular, so the tolerance is loose."""
        r_orbit = (1737.0 + 5_000.0) / LENGTH_KM
        pos = _MOON_POS + np.array([r_orbit, 0.0])
        v_rel = np.array([0.0, np.sqrt(_MU / r_orbit)])
        v_moon = np.array([-_MOON_POS[1], _MOON_POS[0]])
        v_inertial = v_rel + v_moon
        omega_cross_r = np.array([-pos[1], pos[0]])
        v_syn = v_inertial - omega_cross_r
        state0 = np.concatenate([pos, v_syn])

        period = 2.0 * np.pi * np.sqrt(r_orbit ** 3 / _MU)
        peri = find_perilune(self.dyn, state0, (0.0, period), n_steps=4_000)

        assert peri.r_perilune == pytest.approx(r_orbit, rel=1e-2)
        assert peri.is_bound

    def test_perilune_speed_close_to_circular_for_circular_orbit(self) -> None:
        r_orbit = (1737.0 + 5_000.0) / LENGTH_KM
        pos = _MOON_POS + np.array([r_orbit, 0.0])
        v_rel = np.array([0.0, np.sqrt(_MU / r_orbit)])
        v_moon = np.array([-_MOON_POS[1], _MOON_POS[0]])
        v_inertial = v_rel + v_moon
        omega_cross_r = np.array([-pos[1], pos[0]])
        v_syn = v_inertial - omega_cross_r
        state0 = np.concatenate([pos, v_syn])

        period = 2.0 * np.pi * np.sqrt(r_orbit ** 3 / _MU)
        peri = find_perilune(self.dyn, state0, (0.0, period), n_steps=4_000)

        # Capture Δv from a (Keplerian-)circular orbit is small but
        # non-zero in CR3BP because Earth perturbs the orbit; the
        # spacecraft is still approximately captured.
        assert capture_delta_v(peri) < 0.05 * peri.v_circ

    def test_hyperbolic_flyby_is_unbound_and_has_positive_capture(self) -> None:
        """A flyby with v_perilune > v_circ at perilune is hyperbolic
        with respect to the Moon: the two-body energy is positive and
        the capture Δv is strictly positive."""
        r_perilune = (1737.0 + 6_000.0) / LENGTH_KM
        v_circ = np.sqrt(_MU / r_perilune)
        # Inject the spacecraft at perilune with 1.5x circular speed --
        # comfortably hyperbolic.
        pos = _MOON_POS + np.array([r_perilune, 0.0])
        v_rel = np.array([0.0, 1.5 * v_circ])
        v_moon = np.array([-_MOON_POS[1], _MOON_POS[0]])
        v_inertial = v_rel + v_moon
        omega_cross_r = np.array([-pos[1], pos[0]])
        v_syn = v_inertial - omega_cross_r
        state0 = np.concatenate([pos, v_syn])

        # Short search window -- spacecraft is at perilune at t = 0.
        peri = find_perilune(self.dyn, state0, (0.0, 0.05), n_steps=4_000)

        assert peri.r_perilune == pytest.approx(r_perilune, rel=1e-3)
        assert not peri.is_bound
        assert capture_delta_v(peri) > 0.0
        assert capture_delta_v(peri) == pytest.approx(0.5 * v_circ, rel=1e-2)


# ---------------------------------------------------------------------------
# End-to-end Phase 2 -> Phase 3 handoff sanity
# ---------------------------------------------------------------------------


class TestPhase2to3Handoff:
    """End-to-end: a hyperbolic flyby's perilune feeds the Edelbaum
    descending spiral with a sensible Δv breakdown."""

    def test_total_dv_breakdown_is_positive(self) -> None:
        """Capture Δv plus Edelbaum-descending Δv from the perilune
        radius down to LLO is the total Phase-3 cost."""
        dyn = PlanarCR3BP()
        r_perilune_input = (1737.0 + 6_000.0) / LENGTH_KM
        v_circ_p = np.sqrt(_MU / r_perilune_input)

        pos = _MOON_POS + np.array([r_perilune_input, 0.0])
        v_rel = np.array([0.0, 1.3 * v_circ_p])  # 30% above circular
        v_moon = np.array([-_MOON_POS[1], _MOON_POS[0]])
        v_inertial = v_rel + v_moon
        omega_cross_r = np.array([-pos[1], pos[0]])
        v_syn = v_inertial - omega_cross_r
        state0 = np.concatenate([pos, v_syn])

        peri = find_perilune(dyn, state0, (0.0, 0.05), n_steps=4_000)
        dv_capture = capture_delta_v(peri)

        r_LLO = (1737.0 + 100.0) / LENGTH_KM
        spiral = edelbaum_spiral_moon_descending(
            r1=peri.r_perilune, r2=r_LLO, a_thrust=0.02,
        )

        total_dv = dv_capture + spiral.dv
        assert total_dv > spiral.dv  # capture adds something
        assert dv_capture == pytest.approx(0.3 * v_circ_p, rel=1e-2)
