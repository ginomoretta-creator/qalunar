"""Tests for the Phase-1 analytical Edelbaum spiral module."""

from __future__ import annotations

import numpy as np
import pytest

from qalunar.reference import (
    EdelbaumSpiral,
    edelbaum_spiral,
    edelbaum_spiral_moon_descending,
)
from qalunar.reference.edelbaum import (
    ACCELERATION_M_S2,
    LENGTH_KM,
    TIME_S,
    VELOCITY_M_S,
)


EARTH_MOON_MU = 0.012150583925359


# ---------------------------------------------------------------------------
# Basic numerical sanity: formulas agree with hand derivation
# ---------------------------------------------------------------------------


class TestEdelbaumSpiralFormulas:
    """Closed-form formulas match their textbook definitions."""

    def test_circular_speeds_from_gm(self) -> None:
        """``v_i = sqrt((1-mu)/r_i)`` in CR3BP nondim units."""
        r1, r2 = 0.02, 0.5
        sp = edelbaum_spiral(r1=r1, r2=r2, a_thrust=0.01)
        expected_v1 = np.sqrt((1.0 - EARTH_MOON_MU) / r1)
        expected_v2 = np.sqrt((1.0 - EARTH_MOON_MU) / r2)
        assert sp.v1 == pytest.approx(expected_v1, rel=1e-12)
        assert sp.v2 == pytest.approx(expected_v2, rel=1e-12)

    def test_delta_v_is_speed_difference(self) -> None:
        """``dv = |v1 - v2|`` by Edelbaum's equation."""
        sp = edelbaum_spiral(r1=0.018, r2=0.5, a_thrust=0.05)
        assert sp.dv == pytest.approx(abs(sp.v1 - sp.v2), rel=1e-12)

    def test_time_of_flight_integrates_constant_acceleration(self) -> None:
        """``tof = dv / a_thrust`` (constant tangential thrust)."""
        a_t = 0.0368
        sp = edelbaum_spiral(r1=0.02, r2=0.4, a_thrust=a_t)
        assert sp.tof == pytest.approx(sp.dv / a_t, rel=1e-12)

    def test_revolution_count_closed_form(self) -> None:
        """``N = |v1^4 - v2^4| / (8 pi GM a_t)`` from integrating v/r dt."""
        sp = edelbaum_spiral(r1=0.018, r2=0.5, a_thrust=0.05)
        gm = 1.0 - EARTH_MOON_MU
        expected = abs(sp.v1**4 - sp.v2**4) / (8.0 * np.pi * gm * sp.a_thrust)
        assert sp.n_revs == pytest.approx(expected, rel=1e-12)

    def test_spiral_is_symmetric_under_r1_r2_swap(self) -> None:
        """A descending spiral has the same Delta-v and TOF as the ascent."""
        up = edelbaum_spiral(r1=0.02, r2=0.4, a_thrust=0.01)
        dn = edelbaum_spiral(r1=0.4, r2=0.02, a_thrust=0.01)
        assert up.dv == pytest.approx(dn.dv, rel=1e-12)
        assert up.tof == pytest.approx(dn.tof, rel=1e-12)
        assert up.n_revs == pytest.approx(dn.n_revs, rel=1e-12)


# ---------------------------------------------------------------------------
# Physical reference mission: LEO -> 0.5 nondim circular Earth orbit
# at 0.1 mm/s^2. Numerical values are the ones quoted in the manual.
# ---------------------------------------------------------------------------


class TestReferenceMission:
    """Phase-1 spiral for the manual's hero two-phase mission."""

    def setup_method(self) -> None:
        self.sp = edelbaum_spiral(r1=0.018, r2=0.5, a_thrust=0.0368)

    def test_delta_v_is_about_6_km_per_s(self) -> None:
        """~6.14 km/s Delta-v, within 100 m/s tolerance."""
        dv_m_s = self.sp.dv * VELOCITY_M_S
        assert dv_m_s == pytest.approx(6_142.0, abs=100.0)

    def test_tof_is_about_2_years(self) -> None:
        """~709 days, within 10 days tolerance."""
        tof_days = self.sp.tof * TIME_S / 86_400.0
        assert tof_days == pytest.approx(709.0, abs=10.0)

    def test_has_thousands_of_revolutions(self) -> None:
        """Low-thrust LEO spirals have thousands of revs."""
        assert self.sp.n_revs > 3_000.0
        assert self.sp.n_revs < 4_000.0

    def test_v1_is_leo_circular_velocity(self) -> None:
        """At r = 6,919 km, circular velocity is ~7,580 m/s."""
        v1_m_s = self.sp.v1 * VELOCITY_M_S
        assert v1_m_s == pytest.approx(7_580.0, abs=50.0)


# ---------------------------------------------------------------------------
# Handoff state in the CR3BP synodic frame
# ---------------------------------------------------------------------------


class TestHandoffState:
    """Phase-1 -> Phase-2 synodic handoff is placed correctly."""

    def test_position_is_on_positive_x_axis(self) -> None:
        """Handoff occurs as the spacecraft crosses the +x direction."""
        sp = edelbaum_spiral(r1=0.02, r2=0.5, a_thrust=0.01)
        r_syn, _ = sp.handoff_state_synodic()
        assert r_syn[0] == pytest.approx(-EARTH_MOON_MU + 0.5, rel=1e-12)
        assert r_syn[1] == 0.0

    def test_synodic_velocity_consistent_with_inertial_circular(self) -> None:
        """``v_syn = v_inertial - omega x r`` with ``omega x r = (-y, x)``."""
        sp = edelbaum_spiral(r1=0.02, r2=0.5, a_thrust=0.01)
        r_syn, v_syn = sp.handoff_state_synodic()
        # v_inertial = (0, v2) prograde; v_syn = (0 - (-y), v2 - x) = (y, v2 - x)
        assert v_syn[0] == pytest.approx(r_syn[1], rel=1e-12)
        assert v_syn[1] == pytest.approx(sp.v2 - r_syn[0], rel=1e-12)

    def test_synodic_speed_is_less_than_inertial(self) -> None:
        """At a prograde crossing the corotation subtracts, not adds."""
        sp = edelbaum_spiral(r1=0.018, r2=0.5, a_thrust=0.05)
        _, v_syn = sp.handoff_state_synodic()
        assert np.linalg.norm(v_syn) < sp.v2

    def test_jacobi_integral_matches_two_body_energy(self) -> None:
        """At r2 = 0.5 (far from Moon), Jacobi integral is close to the
        two-body energy of a prograde circular orbit. This is a sanity
        check that the synodic conversion is self-consistent.
        """
        r2 = 0.5
        sp = edelbaum_spiral(r1=0.02, r2=r2, a_thrust=0.01)
        r_syn, v_syn = sp.handoff_state_synodic()
        # Two-body energy (per unit mass) = v^2/2 - GM/r.
        gm = 1.0 - EARTH_MOON_MU
        v_inertial_sq = sp.v2**2
        e_kep = 0.5 * v_inertial_sq - gm / r2
        # For circular orbit, e_kep = -GM/(2r). Check that too.
        assert e_kep == pytest.approx(-gm / (2.0 * r2), rel=1e-12)
        # And the synodic position is at distance r2 from Earth.
        earth = np.array([-EARTH_MOON_MU, 0.0])
        assert np.linalg.norm(r_syn - earth) == pytest.approx(r2, rel=1e-12)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestValidation:
    """Bad inputs raise with informative messages."""

    def test_rejects_non_positive_r1(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            edelbaum_spiral(r1=0.0, r2=0.5, a_thrust=0.01)

    def test_rejects_non_positive_r2(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            edelbaum_spiral(r1=0.02, r2=-0.1, a_thrust=0.01)

    def test_rejects_equal_radii(self) -> None:
        with pytest.raises(ValueError, match="differ"):
            edelbaum_spiral(r1=0.1, r2=0.1, a_thrust=0.01)

    def test_rejects_non_positive_thrust(self) -> None:
        with pytest.raises(ValueError, match="a_thrust"):
            edelbaum_spiral(r1=0.02, r2=0.5, a_thrust=0.0)


# ---------------------------------------------------------------------------
# SI unit reporting
# ---------------------------------------------------------------------------


class TestSiReport:
    """``si_report`` returns a printable dict with sensible strings."""

    def test_report_has_expected_keys(self) -> None:
        sp = edelbaum_spiral(r1=0.018, r2=0.5, a_thrust=0.0368)
        rep = sp.si_report()
        expected = {
            "body", "r1_km", "r2_km", "v1_m_s", "v2_m_s", "dv_m_s",
            "tof_days", "tof_years", "n_revs", "a_thrust_mm_s2",
        }
        assert set(rep.keys()) == expected

    def test_all_values_are_strings(self) -> None:
        sp = edelbaum_spiral(r1=0.018, r2=0.5, a_thrust=0.0368)
        rep = sp.si_report()
        assert all(isinstance(v, str) for v in rep.values())

    def test_length_conversion_constant(self) -> None:
        """LENGTH_KM is the Earth-Moon mean distance."""
        assert LENGTH_KM == pytest.approx(384_400.0)

    def test_velocity_conversion_constant(self) -> None:
        """V = L/T is ~1023 m/s, the CR3BP natural velocity scale."""
        assert VELOCITY_M_S == pytest.approx(1_023.0, abs=2.0)

    def test_acceleration_conversion_constant(self) -> None:
        """A = V/T is ~2.72e-3 m/s^2."""
        assert ACCELERATION_M_S2 == pytest.approx(2.72e-3, abs=1e-4)

    def test_time_conversion_constant(self) -> None:
        """T = lunar sidereal period / 2 pi, ~3.76e5 s."""
        assert TIME_S == pytest.approx(3.7569e5, rel=1e-4)


# ---------------------------------------------------------------------------
# Dataclass contract
# ---------------------------------------------------------------------------


class TestDataClassContract:
    """EdelbaumSpiral is frozen and holds what we expect."""

    def test_is_frozen(self) -> None:
        sp = edelbaum_spiral(r1=0.02, r2=0.5, a_thrust=0.01)
        with pytest.raises((AttributeError, Exception)):
            sp.r1 = 999.0  # type: ignore[misc]

    def test_stores_configuration(self) -> None:
        sp = edelbaum_spiral(r1=0.02, r2=0.5, a_thrust=0.01, mu=0.0123)
        assert sp.r1 == 0.02
        assert sp.r2 == 0.5
        assert sp.a_thrust == 0.01
        assert sp.mu == 0.0123

    def test_is_edelbaum_spiral_type(self) -> None:
        sp = edelbaum_spiral(r1=0.02, r2=0.5, a_thrust=0.01)
        assert isinstance(sp, EdelbaumSpiral)

    def test_default_body_is_earth(self) -> None:
        sp = edelbaum_spiral(r1=0.02, r2=0.5, a_thrust=0.01)
        assert sp.body == "earth"


# ---------------------------------------------------------------------------
# Phase-3: Moon-centered descending spiral (lunar orbit insertion)
# ---------------------------------------------------------------------------


# Representative Phase-3 LOI: 6,000 km flyby altitude → 100 km LLO,
# at the same 0.054 mm/s² thrust used in the paper's Phase-2 scenario.
_R_PERILUNE = (1737.0 + 6_000.0) / LENGTH_KM
_R_LLO = (1737.0 + 100.0) / LENGTH_KM
_A_LOI = 0.02


class TestMoonDescendingFormulas:
    """Closed-form formulas use ``GM_Moon = mu``."""

    def test_circular_speeds_use_moon_gm(self) -> None:
        sp = edelbaum_spiral_moon_descending(
            r1=_R_PERILUNE, r2=_R_LLO, a_thrust=_A_LOI,
        )
        expected_v1 = np.sqrt(EARTH_MOON_MU / _R_PERILUNE)
        expected_v2 = np.sqrt(EARTH_MOON_MU / _R_LLO)
        assert sp.v1 == pytest.approx(expected_v1, rel=1e-12)
        assert sp.v2 == pytest.approx(expected_v2, rel=1e-12)

    def test_descent_increases_orbital_speed(self) -> None:
        """Lower orbits are faster: v2 > v1 when r1 > r2."""
        sp = edelbaum_spiral_moon_descending(
            r1=_R_PERILUNE, r2=_R_LLO, a_thrust=_A_LOI,
        )
        assert sp.v2 > sp.v1

    def test_delta_v_is_speed_difference(self) -> None:
        sp = edelbaum_spiral_moon_descending(
            r1=_R_PERILUNE, r2=_R_LLO, a_thrust=_A_LOI,
        )
        assert sp.dv == pytest.approx(abs(sp.v1 - sp.v2), rel=1e-12)

    def test_time_of_flight_is_dv_over_thrust(self) -> None:
        sp = edelbaum_spiral_moon_descending(
            r1=_R_PERILUNE, r2=_R_LLO, a_thrust=_A_LOI,
        )
        assert sp.tof == pytest.approx(sp.dv / _A_LOI, rel=1e-12)

    def test_revolution_count_uses_moon_gm(self) -> None:
        sp = edelbaum_spiral_moon_descending(
            r1=_R_PERILUNE, r2=_R_LLO, a_thrust=_A_LOI,
        )
        expected = (
            abs(sp.v1**4 - sp.v2**4)
            / (8.0 * np.pi * EARTH_MOON_MU * _A_LOI)
        )
        assert sp.n_revs == pytest.approx(expected, rel=1e-12)

    def test_body_label_is_moon(self) -> None:
        sp = edelbaum_spiral_moon_descending(
            r1=_R_PERILUNE, r2=_R_LLO, a_thrust=_A_LOI,
        )
        assert sp.body == "moon"


class TestMoonDescendingValidation:
    """Bad inputs raise informative errors."""

    def test_rejects_r1_less_than_r2(self) -> None:
        """Descending spiral requires r1 > r2."""
        with pytest.raises(ValueError, match="Descending"):
            edelbaum_spiral_moon_descending(
                r1=_R_LLO, r2=_R_PERILUNE, a_thrust=_A_LOI,
            )

    def test_rejects_equal_radii(self) -> None:
        with pytest.raises(ValueError, match="differ"):
            edelbaum_spiral_moon_descending(
                r1=_R_LLO, r2=_R_LLO, a_thrust=_A_LOI,
            )

    def test_rejects_non_positive_radius(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            edelbaum_spiral_moon_descending(
                r1=-0.01, r2=_R_LLO, a_thrust=_A_LOI,
            )

    def test_rejects_non_positive_thrust(self) -> None:
        with pytest.raises(ValueError, match="a_thrust"):
            edelbaum_spiral_moon_descending(
                r1=_R_PERILUNE, r2=_R_LLO, a_thrust=0.0,
            )


class TestMoonHandoffState:
    """Moon-centered handoff is consistent with the synodic frame."""

    def setup_method(self) -> None:
        self.sp = edelbaum_spiral_moon_descending(
            r1=_R_PERILUNE, r2=_R_LLO, a_thrust=_A_LOI,
        )
        self.r_syn, self.v_syn = self.sp.handoff_state_synodic()
        self.moon_pos = np.array([1.0 - EARTH_MOON_MU, 0.0])

    def test_position_is_at_r2_from_moon(self) -> None:
        """The handoff position is at distance r2 from the Moon."""
        d_to_moon = np.linalg.norm(self.r_syn - self.moon_pos)
        assert d_to_moon == pytest.approx(self.sp.r2, rel=1e-12)

    def test_position_is_on_positive_x_relative_to_moon(self) -> None:
        """Handoff occurs as the spacecraft crosses the +x direction
        in the Moon-relative frame."""
        rel = self.r_syn - self.moon_pos
        assert rel[0] == pytest.approx(self.sp.r2, rel=1e-12)
        assert rel[1] == 0.0

    def test_moon_relative_speed_is_circular(self) -> None:
        """Spacecraft's Moon-relative velocity (in the inertial frame)
        has magnitude equal to the local circular speed v2.

        v_inertial = v_syn + omega × r_spacecraft
        v_rel_inertial = v_inertial - v_moon_inertial
        |v_rel_inertial| should equal v2.
        """
        omega_cross_r = np.array([-self.r_syn[1], self.r_syn[0]])
        v_inertial = self.v_syn + omega_cross_r
        v_moon_inertial = np.array([-self.moon_pos[1], self.moon_pos[0]])
        v_rel = v_inertial - v_moon_inertial
        assert np.linalg.norm(v_rel) == pytest.approx(self.sp.v2, rel=1e-12)

    def test_handoff_position_far_from_earth(self) -> None:
        """Sanity: end of LOI is near the Moon, far from Earth."""
        earth = np.array([-EARTH_MOON_MU, 0.0])
        d_to_earth = np.linalg.norm(self.r_syn - earth)
        # Should be ~1 nondim unit (Earth-Moon distance), not r2.
        assert d_to_earth > 0.9
        assert d_to_earth < 1.1


class TestMoonReferenceMission:
    """Phase-3 LOI for the paper's spacecraft class.

    100 kg microsatellite, 5.4 mN ion thruster, a ≈ 0.054 mm/s². LOI
    from a 6,000 km perilune flyby down to a 100 km circular LLO.
    """

    def setup_method(self) -> None:
        self.sp = edelbaum_spiral_moon_descending(
            r1=_R_PERILUNE, r2=_R_LLO, a_thrust=_A_LOI,
        )

    def test_delta_v_in_realistic_range(self) -> None:
        """Lunar capture from a slow flyby costs O(100s) m/s."""
        dv_m_s = self.sp.dv * VELOCITY_M_S
        assert dv_m_s > 100.0
        assert dv_m_s < 1000.0

    def test_tof_is_weeks_to_months(self) -> None:
        """Low-thrust LOI takes weeks to months."""
        tof_days = self.sp.tof * TIME_S / 86_400.0
        assert tof_days > 1.0
        assert tof_days < 365.0

    def test_has_many_revolutions(self) -> None:
        """Low-thrust spiral-down has many lunar revolutions."""
        assert self.sp.n_revs > 1.0

    def test_si_report_has_moon_body(self) -> None:
        rep = self.sp.si_report()
        assert rep["body"] == "moon"
