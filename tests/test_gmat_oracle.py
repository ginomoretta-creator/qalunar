"""Tests for the GMAT truth oracle (frame bridge + script generation).

The frame-bridge and script-generation tests are pure Python. The
integration tests require a local GmatConsole install and are skipped
automatically when it is absent (CI machines).
"""

from __future__ import annotations

import numpy as np
import pytest

from qalunar.dynamics import PlanarCR3BP
from qalunar.dynamics.cr3bp import EARTH_MOON_MU
from qalunar.highfidelity.gmat_oracle import (
    GmatOracleConfig,
    _schedule_segments,
    build_schedule_script,
    epoch_plus_seconds,
    find_gmat_console,
    propagate_schedule_gmat,
    rotating_km_to_synodic,
    synodic_to_rotating_km,
)
from qalunar.qubo.thrust_scheduling import ThrustSchedulingConfig
from qalunar.reference.edelbaum import LENGTH_KM, TIME_S


_GMAT_AVAILABLE = find_gmat_console().exists()
needs_gmat = pytest.mark.skipif(
    not _GMAT_AVAILABLE, reason="GmatConsole.exe not found on this machine"
)


class TestFrameBridge:
    def test_round_trip_identity(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            state = rng.normal(0.0, 0.7, size=4)
            back = rotating_km_to_synodic(synodic_to_rotating_km(state))
            np.testing.assert_allclose(back, state, rtol=0, atol=1e-14)

    def test_earth_centre_maps_to_origin(self):
        earth_synodic = np.array([-EARTH_MOON_MU, 0.0, 0.0, 0.0])
        rot = synodic_to_rotating_km(earth_synodic)
        np.testing.assert_allclose(rot, np.zeros(4), atol=1e-12)

    def test_moon_maps_to_canonical_distance(self):
        moon_synodic = np.array([1.0 - EARTH_MOON_MU, 0.0, 0.0, 0.0])
        rot = synodic_to_rotating_km(moon_synodic)
        assert rot[0] == pytest.approx(LENGTH_KM)
        assert rot[1] == 0.0


class TestEpochArithmetic:
    def test_zero_offset_is_identity(self):
        e = "08 Jan 2026 00:00:00.000"
        assert epoch_plus_seconds(e, 0.0) == e

    def test_day_and_fraction(self):
        assert (epoch_plus_seconds("08 Jan 2026 00:00:00.000", 86_400.5)
                == "09 Jan 2026 00:00:00.500")

    def test_month_and_year_rollover(self):
        assert (epoch_plus_seconds("31 Dec 2026 23:59:59.999", 0.0015)
                == "01 Jan 2027 00:00:00.000")

    def test_composition(self):
        e = "08 Jan 2026 12:34:56.789"
        once = epoch_plus_seconds(e, 5000.0 + 7200.0)
        twice = epoch_plus_seconds(epoch_plus_seconds(e, 5000.0), 7200.0)
        assert once == twice


class TestScheduleSegments:
    def test_all_coast_merges(self):
        segs = _schedule_segments(np.zeros(10, dtype=np.int64))
        assert segs == [(False, 10)]

    def test_burns_stay_individual(self):
        q = np.array([1, 1, 0, 0, 0, 1], dtype=np.int64)
        segs = _schedule_segments(q)
        assert segs == [(True, 1), (True, 1), (False, 3), (True, 1)]

    def test_total_interval_count_preserved(self):
        rng = np.random.default_rng(3)
        q = (rng.random(40) < 0.3).astype(np.int64)
        segs = _schedule_segments(q)
        assert sum(n for _, n in segs) == q.size


class TestScriptGeneration:
    def _script(self, schedule, direction="tangential"):
        cfg = ThrustSchedulingConfig(
            thrust_magnitude=0.02, thrust_direction=direction,
        )
        state0 = np.array([0.5, 0.0, 0.0, 1.1])
        return build_schedule_script(
            state0, (0.0, 4.0), schedule, cfg,
            GmatOracleConfig(), report_name="test_report.txt",
        )

    def test_burn_count_matches_schedule(self):
        q = np.array([1, 0, 0, 1, 0, 1, 0, 0, 0, 0], dtype=np.int64)
        script = self._script(q)
        assert script.count("BeginFiniteBurn") == 3
        assert script.count("EndFiniteBurn") == 3

    def test_pure_ascii(self):
        q = np.array([1, 0, 1], dtype=np.int64)
        script = self._script(q)
        script.encode("ascii")  # raises if non-ASCII slipped in

    def test_segment_durations_sum_to_total(self):
        import re
        q = np.array([0, 1, 1, 0, 0, 0, 1, 0], dtype=np.int64)
        script = self._script(q)
        seconds = [
            float(m) for m in re.findall(r"ElapsedSecs = ([0-9.]+)", script)
        ]
        assert sum(seconds) == pytest.approx(4.0 * TIME_S, rel=1e-12)

    def test_antitangential_flips_sign(self):
        q = np.array([1, 0], dtype=np.int64)
        script = self._script(q, direction="antitangential")
        assert "-1 * Sat.EarthMoonRot.VX" in script

    def test_multi_channel_rejected(self):
        cfg = ThrustSchedulingConfig(
            thrust_magnitude=0.02,
            thrust_channels=("tangential", "antitangential"),
        )
        with pytest.raises(NotImplementedError):
            build_schedule_script(
                np.array([0.5, 0.0, 0.0, 1.1]), (0.0, 1.0),
                np.zeros(4, dtype=np.int64), cfg,
            )


@needs_gmat
class TestGmatIntegration:
    """End-to-end runs through GmatConsole (skipped without a local GMAT)."""

    def test_short_coast_close_to_cr3bp(self):
        # Over a short arc the ephemeris truth should stay close to the
        # CR3BP propagation: this catches gross bridge errors (wrong
        # units, frame, or stop-condition semantics) without being
        # sensitive to the genuine model gap that grows on long arcs.
        mu = EARTH_MOON_MU
        r = 200_000.0 / LENGTH_KM
        v_circ = np.sqrt((1 - mu) / r)
        r_syn = np.array([-mu + r, 0.0])
        v_syn = np.array([0.0, v_circ * 1.187]) - np.array([-r_syn[1], r_syn[0]])
        state0 = np.concatenate([r_syn, v_syn])

        t_span = (0.0, 0.2)  # ~0.87 days
        sched_cfg = ThrustSchedulingConfig(thrust_magnitude=0.02)
        coast = np.zeros(5, dtype=np.int64)

        x_gmat = propagate_schedule_gmat(state0, t_span, coast, sched_cfg)

        dyn = PlanarCR3BP(mu=mu)
        _, traj = dyn.propagate(state0, t_span, n_steps=2000)
        x_cr3bp = traj[-1]

        pos_gap_km = np.linalg.norm((x_gmat - x_cr3bp)[:2]) * LENGTH_KM
        assert np.all(np.isfinite(x_gmat))
        assert pos_gap_km < 2_000.0

    def test_burn_changes_final_state(self):
        mu = EARTH_MOON_MU
        r = 200_000.0 / LENGTH_KM
        v_circ = np.sqrt((1 - mu) / r)
        r_syn = np.array([-mu + r, 0.0])
        v_syn = np.array([0.0, v_circ * 1.187]) - np.array([-r_syn[1], r_syn[0]])
        state0 = np.concatenate([r_syn, v_syn])

        t_span = (0.0, 0.2)
        sched_cfg = ThrustSchedulingConfig(thrust_magnitude=0.02)
        coast = np.zeros(5, dtype=np.int64)
        burn = np.array([1, 0, 0, 0, 0], dtype=np.int64)

        x_coast = propagate_schedule_gmat(state0, t_span, coast, sched_cfg)
        x_burn = propagate_schedule_gmat(state0, t_span, burn, sched_cfg)

        gap_km = np.linalg.norm((x_burn - x_coast)[:2]) * LENGTH_KM
        assert gap_km > 10.0  # the burn must visibly move the endpoint
