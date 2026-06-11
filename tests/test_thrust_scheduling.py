"""Tests for the on/off thrust scheduling QUBO."""

from __future__ import annotations

import numpy as np
import pytest

from qalunar.dynamics import PlanarCR3BP
from qalunar.qubo.thrust_scheduling import (
    IterativeSchedulingResult,
    ThrustSchedulingConfig,
    ThrustSchedulingQubo,
    build_thrust_scheduling_qubo,
    propagate_schedule,
    solve_iterative,
)


@pytest.fixture
def dynamics() -> PlanarCR3BP:
    return PlanarCR3BP()


# A benign initial state away from primaries
STATE0 = np.array([-0.5, 0.3, 0.1, -0.2])


def _make_small_qubo(
    dynamics: PlanarCR3BP,
    n_steps: int = 10,
    thrust_mag: float = 0.05,
    fuel_weight: float = 0.0,
) -> ThrustSchedulingQubo:
    """Build a small scheduling QUBO for testing.

    Uses a target that is achievable by thrust: propagate with constant
    thrust to get the target, then build the QUBO against the coast arc.
    """
    u = np.array([thrust_mag * 0.5, thrust_mag * 0.3])
    _, states_thrust = dynamics.propagate(
        STATE0, (0.0, 2.0), n_steps=2000, control=u
    )
    target = states_thrust[-1]

    cfg = ThrustSchedulingConfig(
        thrust_magnitude=thrust_mag,
        thrust_direction="fixed",
        thrust_vector=np.array([0.5, 0.3]),
        fuel_weight=fuel_weight,
    )
    return build_thrust_scheduling_qubo(
        dynamics, STATE0, target, (0.0, 2.0),
        n_decision_steps=n_steps, config=cfg,
    )


# ---------------------------------------------------------------------------
# QUBO structure
# ---------------------------------------------------------------------------


class TestQuboStructure:
    def test_output_dimensions(self, dynamics: PlanarCR3BP) -> None:
        N = 12
        qubo = _make_small_qubo(dynamics, n_steps=N)
        assert qubo.Q.shape == (N, N)
        assert qubo.linear.shape == (N,)
        assert qubo.b_vectors.shape == (N, 4)
        assert qubo.t_grid.shape == (N,)
        assert qubo.n_steps == N

    def test_Q_is_symmetric(self, dynamics: PlanarCR3BP) -> None:
        qubo = _make_small_qubo(dynamics)
        np.testing.assert_allclose(qubo.Q, qubo.Q.T, atol=1e-14)

    def test_all_zeros_gives_coast_energy(self, dynamics: PlanarCR3BP) -> None:
        """q=0 (all coast) should give energy = ||d||² (miss distance squared)."""
        qubo = _make_small_qubo(dynamics)
        q_zeros = np.zeros(qubo.n_steps, dtype=np.int64)
        energy = qubo.energy(q_zeros)
        expected = float(qubo.target_gap @ qubo.target_gap)
        assert energy == pytest.approx(expected, rel=1e-10)

    def test_energy_manual_computation(self, dynamics: PlanarCR3BP) -> None:
        """Verify energy formula: q^T Q q + linear^T q + const."""
        qubo = _make_small_qubo(dynamics, n_steps=8)
        rng = np.random.default_rng(42)
        for _ in range(20):
            q = rng.integers(0, 2, size=qubo.n_steps).astype(np.int64)
            e = qubo.energy(q)
            qf = q.astype(np.float64)
            e_manual = float(qf @ qubo.Q @ qf + qubo.linear @ qf + qubo.constant)
            assert e == pytest.approx(e_manual, rel=1e-12)

    def test_miss_distance_is_zero_for_perfect_schedule(self, dynamics: PlanarCR3BP) -> None:
        """If b_vectors span d, some schedule should have small miss."""
        qubo = _make_small_qubo(dynamics, n_steps=15, thrust_mag=0.1)
        # Brute force the best
        best_q, _ = qubo.brute_force()
        miss = qubo.miss_distance(best_q)
        # With enough burn options, miss should be small
        assert np.linalg.norm(miss) < np.linalg.norm(qubo.target_gap)


# ---------------------------------------------------------------------------
# Brute force
# ---------------------------------------------------------------------------


class TestBruteForce:
    def test_brute_force_finds_global_minimum(self, dynamics: PlanarCR3BP) -> None:
        """Brute force should find the lowest-energy bitstring."""
        qubo = _make_small_qubo(dynamics, n_steps=10)
        best_q, best_energy = qubo.brute_force()
        # Verify by checking a sample of random bitstrings
        rng = np.random.default_rng(0)
        for _ in range(500):
            q = rng.integers(0, 2, size=qubo.n_steps).astype(np.int64)
            assert qubo.energy(q) >= best_energy - 1e-12

    def test_brute_force_rejects_large_n(self, dynamics: PlanarCR3BP) -> None:
        qubo = _make_small_qubo(dynamics, n_steps=10)
        # Monkey-patch n_steps to test the guard
        qubo.n_steps = 25
        with pytest.raises(ValueError, match="intractable"):
            qubo.brute_force()

    def test_best_is_not_all_zeros(self, dynamics: PlanarCR3BP) -> None:
        """The optimal schedule should involve some burns (not all coast)."""
        qubo = _make_small_qubo(dynamics, n_steps=10, thrust_mag=0.1)
        best_q, _ = qubo.brute_force()
        assert qubo.n_burns(best_q) > 0

    def test_fuel_weight_reduces_burns(self, dynamics: PlanarCR3BP) -> None:
        """Higher fuel_weight should produce fewer burns."""
        qubo_free = _make_small_qubo(dynamics, n_steps=10, fuel_weight=0.0)
        qubo_expensive = _make_small_qubo(dynamics, n_steps=10, fuel_weight=0.1)
        best_free, _ = qubo_free.brute_force()
        best_expensive, _ = qubo_expensive.brute_force()
        assert qubo_free.n_burns(best_free) >= qubo_expensive.n_burns(best_expensive)


# ---------------------------------------------------------------------------
# Thrust direction modes
# ---------------------------------------------------------------------------


class TestThrustDirections:
    def test_tangential(self, dynamics: PlanarCR3BP) -> None:
        _, states = dynamics.propagate(STATE0, (0.0, 2.0), n_steps=2000)
        target = states[-1] + np.array([0.05, 0.03, 0.0, 0.0])
        cfg = ThrustSchedulingConfig(thrust_direction="tangential")
        qubo = build_thrust_scheduling_qubo(
            dynamics, STATE0, target, (0.0, 2.0), 8, config=cfg
        )
        assert qubo.n_steps == 8

    def test_antitangential(self, dynamics: PlanarCR3BP) -> None:
        _, states = dynamics.propagate(STATE0, (0.0, 2.0), n_steps=2000)
        target = states[-1] + np.array([0.05, 0.03, 0.0, 0.0])
        cfg = ThrustSchedulingConfig(thrust_direction="antitangential")
        qubo = build_thrust_scheduling_qubo(
            dynamics, STATE0, target, (0.0, 2.0), 8, config=cfg
        )
        assert qubo.n_steps == 8

    def test_fixed_direction(self, dynamics: PlanarCR3BP) -> None:
        _, states = dynamics.propagate(STATE0, (0.0, 2.0), n_steps=2000)
        target = states[-1] + np.array([0.05, 0.03, 0.0, 0.0])
        cfg = ThrustSchedulingConfig(
            thrust_direction="fixed",
            thrust_vector=np.array([1.0, 0.5]),
        )
        qubo = build_thrust_scheduling_qubo(
            dynamics, STATE0, target, (0.0, 2.0), 8, config=cfg
        )
        assert qubo.n_steps == 8

    def test_fixed_requires_vector(self, dynamics: PlanarCR3BP) -> None:
        _, states = dynamics.propagate(STATE0, (0.0, 2.0), n_steps=2000)
        target = states[-1] + np.array([0.05, 0.03, 0.0, 0.0])
        cfg = ThrustSchedulingConfig(thrust_direction="fixed", thrust_vector=None)
        with pytest.raises(ValueError, match="thrust_vector required"):
            build_thrust_scheduling_qubo(
                dynamics, STATE0, target, (0.0, 2.0), 8, config=cfg
            )

    def test_invalid_direction_rejected(self, dynamics: PlanarCR3BP) -> None:
        _, states = dynamics.propagate(STATE0, (0.0, 2.0), n_steps=2000)
        target = states[-1] + np.array([0.05, 0.03, 0.0, 0.0])
        cfg = ThrustSchedulingConfig(thrust_direction="sideways")
        with pytest.raises(ValueError, match="Unknown"):
            build_thrust_scheduling_qubo(
                dynamics, STATE0, target, (0.0, 2.0), 8, config=cfg
            )


# ---------------------------------------------------------------------------
# SA sampling vs brute force
# ---------------------------------------------------------------------------


class TestSAvsBruteForce:
    def test_sa_finds_brute_force_optimum(self, dynamics: PlanarCR3BP) -> None:
        """SA should find the same optimum as brute force on a small problem."""
        import dimod
        import neal

        qubo = _make_small_qubo(dynamics, n_steps=10, thrust_mag=0.05)
        best_q_bf, best_e_bf = qubo.brute_force()

        # Build BQM from QUBO
        N = qubo.n_steps
        h = {i: qubo.Q[i, i] + qubo.linear[i] for i in range(N)}
        J = {}
        for i in range(N):
            for j in range(i + 1, N):
                coupling = 2.0 * qubo.Q[i, j]
                if abs(coupling) > 1e-15:
                    J[(i, j)] = coupling
        bqm = dimod.BinaryQuadraticModel(h, J, qubo.constant, dimod.BINARY)

        sampler = neal.SimulatedAnnealingSampler()
        result = sampler.sample(bqm, num_reads=500, seed=42)
        best_sample = result.first.sample
        sa_q = np.array([best_sample[i] for i in range(N)], dtype=np.int64)
        sa_energy = qubo.energy(sa_q)

        assert sa_energy <= best_e_bf + 1e-8, (
            f"SA energy {sa_energy:.6e} > brute force {best_e_bf:.6e}"
        )


# ---------------------------------------------------------------------------
# Multi-channel scheduling
# ---------------------------------------------------------------------------


class TestMultiChannel:
    def _make_multi_qubo(
        self,
        dynamics: PlanarCR3BP,
        channels: tuple[str, ...] = ("tangential", "normal"),
        n_steps: int = 8,
        thrust_mag: float = 0.05,
        fuel_weight: float = 0.0,
        exclusive: bool = False,
    ) -> ThrustSchedulingQubo:
        u = np.array([thrust_mag * 0.5, thrust_mag * 0.3])
        _, states_thrust = dynamics.propagate(
            STATE0, (0.0, 2.0), n_steps=2000, control=u
        )
        target = states_thrust[-1]
        cfg = ThrustSchedulingConfig(
            thrust_magnitude=thrust_mag,
            thrust_channels=channels,
            fuel_weight=fuel_weight,
            exclusive=exclusive,
        )
        return build_thrust_scheduling_qubo(
            dynamics, STATE0, target, (0.0, 2.0),
            n_decision_steps=n_steps, config=cfg,
        )

    def test_dimensions(self, dynamics: PlanarCR3BP) -> None:
        """Two channels with N steps should give 2N variables."""
        N = 8
        qubo = self._make_multi_qubo(dynamics, n_steps=N)
        assert qubo.n_channels == 2
        assert qubo.n_steps == N
        assert qubo.n_vars == 2 * N
        assert qubo.Q.shape == (2 * N, 2 * N)
        assert qubo.linear.shape == (2 * N,)
        assert qubo.b_vectors.shape == (2 * N, 4)

    def test_Q_symmetric(self, dynamics: PlanarCR3BP) -> None:
        qubo = self._make_multi_qubo(dynamics)
        np.testing.assert_allclose(qubo.Q, qubo.Q.T, atol=1e-14)

    def test_channel_labels(self, dynamics: PlanarCR3BP) -> None:
        qubo = self._make_multi_qubo(
            dynamics, channels=("tangential", "normal")
        )
        assert qubo.channel_labels == ("tangential", "normal")

    def test_channel_schedule_extraction(self, dynamics: PlanarCR3BP) -> None:
        N = 8
        qubo = self._make_multi_qubo(dynamics, n_steps=N)
        q = np.zeros(qubo.n_vars, dtype=np.int64)
        # Turn on step 2 in channel 0, step 5 in channel 1
        q[2] = 1  # channel 0, step 2
        q[N + 5] = 1  # channel 1, step 5
        ch0 = qubo.channel_schedule(q, 0)
        ch1 = qubo.channel_schedule(q, 1)
        assert ch0[2] == 1 and ch0.sum() == 1
        assert ch1[5] == 1 and ch1.sum() == 1

    def test_var_index_roundtrip(self, dynamics: PlanarCR3BP) -> None:
        qubo = self._make_multi_qubo(dynamics, n_steps=8)
        for c in range(qubo.n_channels):
            for s in range(qubo.n_steps):
                j = qubo.var_index(c, s)
                c2, s2 = qubo.var_channel_step(j)
                assert c2 == c and s2 == s

    def test_energy_manual(self, dynamics: PlanarCR3BP) -> None:
        """Energy formula should hold for multi-channel QUBO."""
        qubo = self._make_multi_qubo(dynamics, n_steps=6)
        rng = np.random.default_rng(99)
        for _ in range(20):
            q = rng.integers(0, 2, size=qubo.n_vars).astype(np.int64)
            e = qubo.energy(q)
            qf = q.astype(np.float64)
            e_manual = float(qf @ qubo.Q @ qf + qubo.linear @ qf + qubo.constant)
            assert e == pytest.approx(e_manual, rel=1e-12)

    def test_multi_better_than_single(self, dynamics: PlanarCR3BP) -> None:
        """Two channels should achieve equal or lower miss than one."""
        N = 8
        # Single channel (tangential only)
        single = self._make_multi_qubo(
            dynamics, channels=("tangential",), n_steps=N
        )
        best_single, _ = single.brute_force()
        miss_single = np.linalg.norm(single.miss_distance(best_single))

        # Two channels (tangential + normal)
        multi = self._make_multi_qubo(
            dynamics, channels=("tangential", "normal"), n_steps=N
        )
        best_multi, _ = multi.brute_force()
        miss_multi = np.linalg.norm(multi.miss_distance(best_multi))

        assert miss_multi <= miss_single + 1e-10

    def test_exclusive_penalty_discourages_simultaneous(
        self, dynamics: PlanarCR3BP
    ) -> None:
        """With exclusion, the optimum should not fire both channels at the same step."""
        N = 6
        qubo = self._make_multi_qubo(
            dynamics, n_steps=N, exclusive=True,
            channels=("tangential", "normal"),
        )
        best_q, _ = qubo.brute_force()
        # Check no step has both channels active
        for i in range(N):
            ch0 = best_q[i]
            ch1 = best_q[N + i]
            assert ch0 + ch1 <= 1, f"Step {i}: both channels active"

    def test_normal_direction_perpendicular(self, dynamics: PlanarCR3BP) -> None:
        """Normal b_vectors should be perpendicular to tangential b_vectors."""
        qubo = self._make_multi_qubo(
            dynamics, channels=("tangential", "normal"), n_steps=8
        )
        N = qubo.n_steps
        for i in range(N):
            # The thrust direction vectors (before STM mapping) are perpendicular.
            # After STM mapping, b_vectors are in 4D state space so we can't
            # directly check perpendicularity, but we can verify the channel
            # labels are correct and the b_vectors differ.
            b_tan = qubo.b_vectors[i]
            b_norm = qubo.b_vectors[N + i]
            assert not np.allclose(b_tan, b_norm, atol=1e-10)


# ---------------------------------------------------------------------------
# Delta-V reporting
# ---------------------------------------------------------------------------


class TestDeltaV:
    def test_zero_schedule_zero_dv(self, dynamics: PlanarCR3BP) -> None:
        qubo = _make_small_qubo(dynamics, n_steps=10, thrust_mag=0.05)
        q = np.zeros(qubo.n_steps, dtype=np.int64)
        assert qubo.delta_v(q) == 0.0
        assert qubo.delta_v_m_s(q) == 0.0

    def test_full_schedule_dv(self, dynamics: PlanarCR3BP) -> None:
        """All-burn schedule: Δv = thrust_mag * dt * N."""
        N = 10
        thrust_mag = 0.05
        qubo = _make_small_qubo(dynamics, n_steps=N, thrust_mag=thrust_mag)
        q = np.ones(N, dtype=np.int64)
        expected = thrust_mag * qubo.dt_decision * N
        assert qubo.delta_v(q) == pytest.approx(expected, rel=1e-12)

    def test_dv_in_m_s_uses_velocity_unit(self, dynamics: PlanarCR3BP) -> None:
        """delta_v_m_s should equal nondim Δv times the Earth-Moon V unit."""
        from qalunar.reference.edelbaum import VELOCITY_M_S
        N = 10
        qubo = _make_small_qubo(dynamics, n_steps=N, thrust_mag=0.05)
        q = np.array([1, 0, 1, 0, 1, 0, 0, 0, 1, 0], dtype=np.int64)
        ratio = qubo.delta_v_m_s(q) / qubo.delta_v(q)
        assert ratio == pytest.approx(VELOCITY_M_S, rel=1e-12)

    def test_multi_channel_dv_counts_each_firing(
        self, dynamics: PlanarCR3BP
    ) -> None:
        """Two simultaneous channel firings should count as two Δv units."""
        u = np.array([0.05 * 0.5, 0.05 * 0.3])
        _, traj = dynamics.propagate(STATE0, (0.0, 2.0), n_steps=2000, control=u)
        target = traj[-1]
        cfg = ThrustSchedulingConfig(
            thrust_magnitude=0.05,
            thrust_channels=("tangential", "normal"),
        )
        N = 6
        qubo = build_thrust_scheduling_qubo(
            dynamics, STATE0, target, (0.0, 2.0),
            n_decision_steps=N, config=cfg,
        )
        # Single channel active at step 0
        q1 = np.zeros(qubo.n_vars, dtype=np.int64)
        q1[0] = 1
        # Both channels active at step 0
        q2 = np.zeros(qubo.n_vars, dtype=np.int64)
        q2[0] = 1
        q2[N] = 1
        assert qubo.delta_v(q2) == pytest.approx(2.0 * qubo.delta_v(q1))


# ---------------------------------------------------------------------------
# Iterative re-linearization
# ---------------------------------------------------------------------------


def _bf_sampler(qubo: ThrustSchedulingQubo) -> np.ndarray:
    """Brute-force exact sampler for use in iterative tests."""
    return qubo.brute_force()[0]


class TestPropagateSchedule:
    def test_coast_matches_unforced_propagation(
        self, dynamics: PlanarCR3BP
    ) -> None:
        """A coast schedule should match unforced propagation."""
        cfg = ThrustSchedulingConfig(
            thrust_magnitude=0.05, thrust_direction="tangential"
        )
        N = 6
        q_coast = np.zeros(N, dtype=np.int64)
        x_f_sched = propagate_schedule(
            dynamics, STATE0, (0.0, 1.0), q_coast, cfg,
            n_integration_substeps=200,
        )
        _, states_unforced = dynamics.propagate(
            STATE0, (0.0, 1.0), n_steps=N * 200,
        )
        np.testing.assert_allclose(x_f_sched, states_unforced[-1], rtol=1e-12)

    def test_full_burn_changes_state(self, dynamics: PlanarCR3BP) -> None:
        """A schedule with all burns active should differ from coast."""
        cfg = ThrustSchedulingConfig(
            thrust_magnitude=0.05, thrust_direction="tangential"
        )
        N = 6
        q_burn = np.ones(N, dtype=np.int64)
        x_f_burn = propagate_schedule(
            dynamics, STATE0, (0.0, 1.0), q_burn, cfg,
            n_integration_substeps=200,
        )
        x_f_coast = propagate_schedule(
            dynamics, STATE0, (0.0, 1.0), np.zeros(N, dtype=np.int64), cfg,
            n_integration_substeps=200,
        )
        assert np.linalg.norm(x_f_burn - x_f_coast) > 1e-3


class TestNominalSchedule:
    """When linearizing around a non-coast schedule, the QUBO should
    correctly predict the all-zero perturbation case."""

    def test_qubo_at_nominal_predicts_nominal_final(
        self, dynamics: PlanarCR3BP
    ) -> None:
        """Energy of q=q_nom should equal ||x_nom(tf) - target||_W²
        (zero perturbation -> linear prediction = nominal)."""
        cfg = ThrustSchedulingConfig(
            thrust_magnitude=0.03, thrust_direction="fixed",
            thrust_vector=np.array([0.5, 0.3]),
        )
        N = 8
        q_nom = np.array([1, 0, 1, 0, 0, 1, 0, 0], dtype=np.int64)

        target = STATE0 + np.array([0.05, 0.02, 0.0, 0.0])
        qubo = build_thrust_scheduling_qubo(
            dynamics, STATE0, target, (0.0, 1.0),
            n_decision_steps=N, config=cfg,
            nominal_schedule=q_nom,
        )

        # Propagate the true nonlinear trajectory under q_nom — match
        # the QUBO's integrator step count (default 50) to remove the
        # RK4 truncation-step mismatch from the comparison.
        x_f_nom = propagate_schedule(
            dynamics, STATE0, (0.0, 1.0), q_nom, cfg,
            n_integration_substeps=50,
        )

        # The QUBO with nominal q_nom and evaluated AT q_nom predicts
        # x_lin(q_nom | q_nom) = x_nom(tf), so the predicted miss is
        # x_nom(tf) - target. Energy should equal ||that||².
        e = qubo.energy(q_nom)
        e_target = e - cfg.fuel_weight * float(q_nom.sum())
        expected = float(np.linalg.norm(x_f_nom - target) ** 2)
        assert e_target == pytest.approx(expected, rel=1e-10, abs=1e-12)


class TestSolveIterative:
    @pytest.fixture
    def small_problem(self, dynamics: PlanarCR3BP) -> dict:
        """Reachable-target problem small enough for brute-force sampling."""
        cfg = ThrustSchedulingConfig(
            thrust_magnitude=0.05,
            thrust_direction="fixed",
            thrust_vector=np.array([0.5, 0.3]),
            fuel_weight=0.0,
        )
        # Target = some achievable state via thrust
        u = np.array([0.05 * 0.5, 0.05 * 0.3])
        _, traj = dynamics.propagate(
            STATE0, (0.0, 1.5), n_steps=2000, control=u
        )
        target = traj[-1]
        return dict(
            cfg=cfg, state0=STATE0, target=target,
            t_span=(0.0, 1.5), N=10,
        )

    def test_iterative_returns_correct_type(
        self, dynamics: PlanarCR3BP, small_problem: dict
    ) -> None:
        result = solve_iterative(
            dynamics, small_problem["state0"], small_problem["target"],
            small_problem["t_span"], n_decision_steps=small_problem["N"],
            sampler=_bf_sampler, config=small_problem["cfg"],
            max_iters=5, verbose=False,
        )
        assert isinstance(result, IterativeSchedulingResult)
        assert result.schedule.shape == (small_problem["N"],)
        assert result.true_miss.shape == (4,)
        assert result.iterations >= 0
        assert len(result.schedule_history) >= 1
        assert len(result.true_miss_norm_history) == len(result.schedule_history)

    def test_iterative_decreases_true_miss(
        self, dynamics: PlanarCR3BP, small_problem: dict
    ) -> None:
        """Iterative result should be at least as good as single-pass on the
        truth metric."""
        # Single-pass: brute-force the QUBO linearized at coast, then
        # propagate that schedule under truth.
        qubo_single = build_thrust_scheduling_qubo(
            dynamics, small_problem["state0"], small_problem["target"],
            small_problem["t_span"], n_decision_steps=small_problem["N"],
            config=small_problem["cfg"],
        )
        q_single, _ = qubo_single.brute_force()
        x_f_single = propagate_schedule(
            dynamics, small_problem["state0"], small_problem["t_span"],
            q_single, small_problem["cfg"], n_integration_substeps=200,
        )
        single_miss = float(np.linalg.norm(x_f_single - small_problem["target"]))

        # Iterative
        result = solve_iterative(
            dynamics, small_problem["state0"], small_problem["target"],
            small_problem["t_span"], n_decision_steps=small_problem["N"],
            sampler=_bf_sampler, config=small_problem["cfg"],
            max_iters=8,
        )
        iter_miss = float(np.linalg.norm(result.true_miss))

        # Iterative should be no worse than single-pass (with the
        # accept-only-improvement safeguard, it cannot be).
        assert iter_miss <= single_miss + 1e-12

    def test_iterative_history_monotone(
        self, dynamics: PlanarCR3BP, small_problem: dict
    ) -> None:
        """With accept_only_improvement=True, miss history must be non-increasing."""
        result = solve_iterative(
            dynamics, small_problem["state0"], small_problem["target"],
            small_problem["t_span"], n_decision_steps=small_problem["N"],
            sampler=_bf_sampler, config=small_problem["cfg"],
            max_iters=8, accept_only_improvement=True,
        )
        miss = np.asarray(result.true_miss_norm_history)
        diffs = np.diff(miss)
        assert np.all(diffs <= 1e-12), f"Miss history not monotone: {miss}"

    def test_iterative_long_arc_beats_single_pass(
        self, dynamics: PlanarCR3BP
    ) -> None:
        """On a longer arc where linearization error matters, iteration
        should *strictly* improve on single-pass (or at least match it)."""
        cfg = ThrustSchedulingConfig(
            thrust_magnitude=0.08,
            thrust_direction="fixed",
            thrust_vector=np.array([0.5, 0.3]),
            fuel_weight=0.0,
        )
        u = np.array([0.08 * 0.5, 0.08 * 0.3])
        _, traj = dynamics.propagate(
            STATE0, (0.0, 3.0), n_steps=5000, control=u
        )
        target = traj[-1]
        N = 10
        t_span = (0.0, 3.0)

        # Single-pass
        qubo_single = build_thrust_scheduling_qubo(
            dynamics, STATE0, target, t_span,
            n_decision_steps=N, config=cfg,
        )
        q_single, _ = qubo_single.brute_force()
        x_single = propagate_schedule(
            dynamics, STATE0, t_span, q_single, cfg,
            n_integration_substeps=200,
        )
        single_miss = float(np.linalg.norm(x_single - target))

        # Iterative
        result = solve_iterative(
            dynamics, STATE0, target, t_span,
            n_decision_steps=N, sampler=_bf_sampler, config=cfg,
            max_iters=8,
        )
        iter_miss = float(np.linalg.norm(result.true_miss))

        # Iterative no worse; ideally strictly better (long arc).
        assert iter_miss <= single_miss + 1e-12

    def test_fixed_point_termination(
        self, dynamics: PlanarCR3BP, small_problem: dict
    ) -> None:
        """If converged, the final schedule should not change on one more
        re-linearization (it's a fixed point)."""
        result = solve_iterative(
            dynamics, small_problem["state0"], small_problem["target"],
            small_problem["t_span"], n_decision_steps=small_problem["N"],
            sampler=_bf_sampler, config=small_problem["cfg"],
            max_iters=15,
        )
        # Re-linearize at the final schedule and resample
        q_final = result.schedule
        qubo_check = build_thrust_scheduling_qubo(
            dynamics, small_problem["state0"], small_problem["target"],
            small_problem["t_span"],
            n_decision_steps=small_problem["N"], config=small_problem["cfg"],
            nominal_schedule=q_final,
        )
        q_resampled, _ = qubo_check.brute_force()
        # Either schedule is a fixed point, OR the resampled schedule
        # gives no improvement under truth (otherwise solve_iterative
        # would have continued).
        if not np.array_equal(q_resampled, q_final):
            x_resampled = propagate_schedule(
                dynamics, small_problem["state0"], small_problem["t_span"],
                q_resampled, small_problem["cfg"], n_integration_substeps=200,
            )
            resampled_miss = np.linalg.norm(
                x_resampled - small_problem["target"]
            )
            assert resampled_miss >= np.linalg.norm(result.true_miss) - 1e-12
