"""Regression tests for the 2026-09 project audit.

Each test pins a defect the audit found so it cannot silently return:
rotation sense of the synodic frame, left-endpoint quadrature bias in the
impulse-response vectors, hard-coded exclusion penalty, cardinality QUBO
vs. top-K, chain-strength sign bug, SA-vs-brute-force on more than one
instance, monotonicity recomputed independently of the accept gate,
rejection reported as convergence, the capture verdict without a distance
gate, and the untested piecewise STM.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from qalunar.dynamics import PlanarCR3BP
from qalunar.qubo.cardinality import (
    build_cardinality_qubo,
    solve_cardinality_qubo,
    top_k,
)
from qalunar.qubo.lunar_capture import (
    is_captured,
    moon_hill_radius,
    reconstruct_full_trajectory,
    solve_lunar_capture_sliding_window,
)
from qalunar.qubo.receding_horizon import CentralBody
from qalunar.qubo.scheduling_samplers import (
    chain_strength_for,
    sample_brute_force,
    sample_simulated_annealing,
)
from qalunar.qubo.thrust_scheduling import (
    ThrustSchedulingConfig,
    ThrustSchedulingQubo,
    build_thrust_scheduling_qubo,
    propagate_schedule,
    solve_iterative,
)


@pytest.fixture
def dyn() -> PlanarCR3BP:
    return PlanarCR3BP()


STATE0 = np.array([-0.5, 0.3, 0.1, -0.2])
T_SPAN = (0.0, 1.5)


def _moon_state(r: float, speed_factor: float, mu: float) -> np.ndarray:
    """Synodic state on the +x side of the Moon at body-relative radius ``r``
    with inertial speed ``speed_factor * v_circ`` along +y."""
    v_circ = np.sqrt(mu / r)
    px = 1.0 - mu + r
    # v_syn = v_inertial - omega x P,  omega x P = (-P_y, P_x) = (0, px)
    v_inertial_y = speed_factor * v_circ + (1.0 - mu)   # + Moon's own speed
    return np.array([px, 0.0, 0.0, v_inertial_y - px])


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------


class TestCoriolisSign:
    def test_rotation_sense_is_pinned(self, dyn: PlanarCR3BP) -> None:
        """ax = +2 vy + Omega_x, ay = -2 vx + Omega_y. Flipping both signs
        (the mirror-image flow) passes every invariant-based test; this one
        catches it."""
        s = np.array([0.7, -0.3, 0.42, -0.17])
        s_rest = s.copy()
        s_rest[2:] = 0.0
        acc = dyn.rhs(s)[2:]
        omega = dyn.rhs(s_rest)[2:]
        assert acc[0] - omega[0] == pytest.approx(2.0 * s[3])
        assert acc[1] - omega[1] == pytest.approx(-2.0 * s[2])


class TestPiecewiseSTM:
    CONTROLS = [np.array([0.01, 0.0]), None, np.array([0.0, -0.01]), None]
    T = 1.2

    def _flow(self, dyn: PlanarCR3BP, x0: np.ndarray) -> np.ndarray:
        s = x0.copy()
        dt = self.T / len(self.CONTROLS)
        for i, u in enumerate(self.CONTROLS):
            _, seg = dyn.propagate(s, (i * dt, (i + 1) * dt), n_steps=40, control=u)
            s = seg[-1]
        return s

    def test_state_matches_sequential_propagation(self, dyn: PlanarCR3BP) -> None:
        _, states, _ = dyn.propagate_stm_piecewise(
            STATE0, (0.0, self.T), self.CONTROLS, n_substeps=40,
        )
        np.testing.assert_allclose(states[-1], self._flow(dyn, STATE0), atol=1e-12)

    def test_stm_matches_central_differences(self, dyn: PlanarCR3BP) -> None:
        _, _, stms = dyn.propagate_stm_piecewise(
            STATE0, (0.0, self.T), self.CONTROLS, n_substeps=40,
        )
        eps = 1e-6
        J = np.empty((4, 4))
        for k in range(4):
            e = np.zeros(4)
            e[k] = eps
            J[:, k] = (self._flow(dyn, STATE0 + e) - self._flow(dyn, STATE0 - e)) / (2 * eps)
        np.testing.assert_allclose(stms[-1], J, atol=2e-6, rtol=2e-6)


# ---------------------------------------------------------------------------
# QUBO assembly
# ---------------------------------------------------------------------------


class TestImpulseQuadrature:
    @staticmethod
    def _single_flip_errors(dyn: PlanarCR3BP, quadrature: str, N: int = 8) -> np.ndarray:
        # Fixed direction isolates the STM quadrature from the direction-
        # freezing error that both rules share.
        cfg = ThrustSchedulingConfig(
            thrust_magnitude=0.02, thrust_direction="fixed",
            thrust_vector=np.array([0.6, 0.8]), impulse_quadrature=quadrature,
        )
        coast_end = dyn.propagate(STATE0, T_SPAN, n_steps=4000)[1][-1]
        qubo = build_thrust_scheduling_qubo(
            dyn, STATE0, coast_end, T_SPAN, n_decision_steps=N, config=cfg,
            n_integration_substeps=50,
        )
        errs = []
        for j in range(N):
            q = np.zeros(N, dtype=np.int64)
            q[j] = 1
            predicted = qubo.coast_trajectory[-1] + qubo.b_vectors[j]
            true = propagate_schedule(
                dyn, STATE0, T_SPAN, q, cfg, n_integration_substeps=400,
            )
            errs.append(np.linalg.norm(true - predicted))
        return np.asarray(errs)

    def test_midpoint_beats_left_endpoint(self, dyn: PlanarCR3BP) -> None:
        left = self._single_flip_errors(dyn, "left")
        mid = self._single_flip_errors(dyn, "midpoint")
        assert mid.mean() < 0.5 * left.mean()

    def test_default_is_midpoint(self) -> None:
        assert ThrustSchedulingConfig().impulse_quadrature == "midpoint"

    def test_invalid_rule_rejected(self, dyn: PlanarCR3BP) -> None:
        cfg = ThrustSchedulingConfig(impulse_quadrature="right")
        with pytest.raises(ValueError):
            build_thrust_scheduling_qubo(
                dyn, STATE0, STATE0, T_SPAN, n_decision_steps=4, config=cfg,
            )


class TestExclusionPenalty:
    def _target(self, dyn: PlanarCR3BP) -> np.ndarray:
        return dyn.propagate(
            STATE0, T_SPAN, n_steps=2000, control=np.array([0.01, 0.004]),
        )[1][-1]

    def test_auto_scaled_to_physical_swing_and_enforced(self, dyn: PlanarCR3BP) -> None:
        N = 6
        base = dict(thrust_magnitude=0.02, thrust_channels=("tangential", "normal"))
        q_phys = build_thrust_scheduling_qubo(
            dyn, STATE0, self._target(dyn), T_SPAN, n_decision_steps=N,
            config=ThrustSchedulingConfig(**base, exclusive=False),
        )
        q_auto = build_thrust_scheduling_qubo(
            dyn, STATE0, self._target(dyn), T_SPAN, n_decision_steps=N,
            config=ThrustSchedulingConfig(**base, exclusive=True),
        )
        q_fixed = build_thrust_scheduling_qubo(
            dyn, STATE0, self._target(dyn), T_SPAN, n_decision_steps=N,
            config=ThrustSchedulingConfig(**base, exclusive=True, exclusion_penalty=10.0),
        )
        swing = (np.abs(q_phys.linear) + np.abs(q_phys.Q).sum(axis=1)).max()
        assert q_phys.exclusion_penalty == 0.0
        assert q_auto.exclusion_penalty == pytest.approx(2.0 * swing)
        assert q_fixed.exclusion_penalty == 10.0
        # The auto value is orders of magnitude tighter than the old default...
        assert q_auto.coefficient_range() < q_fixed.coefficient_range() / 100
        # ...and still enforces the constraint at the ground state.
        q_best, _ = q_auto.brute_force()
        for i in range(N):
            assert q_best[i] + q_best[N + i] <= 1


class TestCardinalityQubo:
    @pytest.mark.parametrize("seed,k", [(0, 3), (1, 4), (2, 5)])
    def test_sa_ground_state_is_top_k(self, seed: int, k: int) -> None:
        g = np.random.default_rng(seed).uniform(5.0, 160.0, 24)
        q, info = solve_cardinality_qubo(g, k, num_reads=500, seed=seed)
        assert info["budget_met"] and info["matches_top_k"]
        np.testing.assert_array_equal(q, top_k(g, k))

    def test_penalty_must_exceed_max_gain(self) -> None:
        g = np.arange(1.0, 25.0)
        qubo = build_cardinality_qubo(g, 4)
        assert qubo.penalty > g.max()
        with pytest.raises(ValueError):
            build_cardinality_qubo(g, 4, penalty=10.0)

    def test_symmetric_q_counts_each_pair_once(self) -> None:
        g = np.array([3.0, 9.0, 1.0, 7.0, 5.0])
        qubo = build_cardinality_qubo(g, 2)
        q = top_k(g, 2)
        assert qubo.energy(q) == pytest.approx(-g[q == 1].sum())
        q3 = q.copy()
        q3[np.argmin(g)] = 1                      # one burn over budget
        assert qubo.energy(q3) == pytest.approx(-g[q3 == 1].sum() + qubo.penalty)


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------


def test_chain_strength_uses_absolute_maximum() -> None:
    Q = np.array([[0.1, -5.0], [-5.0, 0.2]])
    qubo = ThrustSchedulingQubo(
        Q=Q, linear=np.array([0.3, 0.1]), constant=0.0, n_steps=2, n_channels=1,
        channel_labels=("t",), b_vectors=np.zeros((2, 4)), target_gap=np.zeros(4),
        fuel_weight=0.0, t_grid=np.zeros(2), coast_trajectory=np.zeros((1, 4)),
    )
    assert chain_strength_for(qubo, 2.0) == pytest.approx(10.0)


class TestSARecoversBruteForce:
    """The manuscript claimed SA recovers the brute-force optimum "for N <= 20
    on every instance"; it had been tested on one instance at N = 10. A sweep
    over 18 random instances (N in {15, 18, 20}, seeds 0-5) at the code's
    default 1000 reads recovered the optimum in 17; at N = 20 seed 2 the SA
    energy was 3.5x the optimum at 1000 reads and still 2.05x at 3000. The
    claim therefore holds statistically at N <= 15 with 1000 reads, and that
    is what is pinned here; the N = 20 counterexample is left in the
    manuscript's solver-benchmark section, not hidden by a weaker test."""

    @pytest.mark.parametrize("N", [10, 12, 15])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_sa_matches_brute_force(self, dyn: PlanarCR3BP, N: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        state0 = STATE0 + rng.normal(0.0, 0.05, 4)
        u = rng.uniform(-0.03, 0.03, 2)
        target = dyn.propagate(state0, T_SPAN, n_steps=2000, control=u)[1][-1]
        cfg = ThrustSchedulingConfig(
            thrust_magnitude=0.05, thrust_direction="fixed",
            thrust_vector=u / np.linalg.norm(u),
        )
        qubo = build_thrust_scheduling_qubo(
            dyn, state0, target, T_SPAN, n_decision_steps=N, config=cfg,
        )
        bf = sample_brute_force(qubo)
        sa = sample_simulated_annealing(qubo, num_reads=1000, seed=seed)
        assert sa.energy <= bf.energy + 1e-12 * max(1.0, abs(bf.energy))


# ---------------------------------------------------------------------------
# Trust-region loop
# ---------------------------------------------------------------------------


class TestTrustRegion:
    def _setup(self, dyn: PlanarCR3BP):
        cfg = ThrustSchedulingConfig(
            thrust_magnitude=0.05, thrust_direction="fixed",
            thrust_vector=np.array([0.5, 0.3]),
        )
        target = dyn.propagate(
            STATE0, T_SPAN, n_steps=2000, control=np.array([0.025, 0.015]),
        )[1][-1]
        return cfg, target

    def test_history_monotone_when_recomputed_independently(self, dyn: PlanarCR3BP) -> None:
        """The recorded history is appended only after the accept gate, so it
        cannot violate monotonicity by construction. Re-propagate every
        schedule independently and check the sequence *that* produces."""
        cfg, target = self._setup(dyn)
        res = solve_iterative(
            dyn, STATE0, target, T_SPAN, n_decision_steps=10,
            sampler=lambda q: q.brute_force()[0], config=cfg,
            max_iters=6, n_truth_substeps=200,
        )
        norms = [
            float(np.linalg.norm(
                propagate_schedule(dyn, STATE0, T_SPAN, s, cfg, n_integration_substeps=200)
                - target))
            for s in res.schedule_history
        ]
        np.testing.assert_allclose(norms, res.true_miss_norm_history, rtol=1e-10)
        assert all(b <= a + 1e-12 for a, b in zip(norms, norms[1:]))
        assert norms[-1] < norms[0]

    def test_rejection_is_reported_as_stall_not_convergence(self, dyn: PlanarCR3BP) -> None:
        cfg, target = self._setup(dyn)
        cfg_bad = replace(cfg, thrust_vector=-cfg.thrust_vector)   # any burn hurts
        res = solve_iterative(
            dyn, STATE0, target, T_SPAN, n_decision_steps=8,
            sampler=lambda q: np.ones(q.n_vars, dtype=np.int64), config=cfg_bad,
            max_iters=3,
        )
        assert res.iterations == 0
        assert res.converged is False
        assert "non-improving" in res.converged_reason
        np.testing.assert_array_equal(res.schedule, np.zeros(8, dtype=np.int64))


# ---------------------------------------------------------------------------
# Capture verdict and flown timeline
# ---------------------------------------------------------------------------


class TestCaptureGate:
    def test_hill_radius_value(self, dyn: PlanarCR3BP) -> None:
        assert moon_hill_radius(dyn.mu) == pytest.approx(0.1594, abs=2e-3)

    def test_bound_near_moon_is_captured(self, dyn: PlanarCR3BP) -> None:
        s = _moon_state(0.02, 1.0, dyn.mu)
        assert CentralBody.moon(dyn.mu).two_body_energy(s) < 0.0
        assert is_captured(s, dyn.mu)

    def test_negative_energy_outside_hill_sphere_is_not_captured(self, dyn: PlanarCR3BP) -> None:
        s = _moon_state(0.5, 1.0, dyn.mu)            # 3x the Hill radius out
        assert CentralBody.moon(dyn.mu).two_body_energy(s) < 0.0   # bare sign test says bound
        assert not is_captured(s, dyn.mu)


class TestFlownTimeline:
    def test_segments_cover_total_tof_and_reconstruct_is_continuous(self, dyn: PlanarCR3BP) -> None:
        s0 = _moon_state(0.03, 1.15, dyn.mu)          # mildly hyperbolic flyby
        res = solve_lunar_capture_sliding_window(
            dyn, s0, sampler=lambda q: q.brute_force()[0],
            n_windows=3, window_revs=0.25, n_decision_steps=6,
            thrust_magnitude=0.02, moon_action_radius=None,
            n_integration_substeps=30, n_truth_substeps=60, max_inner_iters=2,
        )
        assert res.segments, "segments must be recorded"
        assert res.total_tof == pytest.approx(sum(seg.duration for seg in res.segments))
        assert res.total_tof > 0.0
        for a, b in zip(res.segments, res.segments[1:]):
            assert b.t_start == pytest.approx(a.t_start + a.duration)
            np.testing.assert_allclose(b.state_initial, a.state_final)
        t, x = reconstruct_full_trajectory(dyn, res, thrust_magnitude=0.02,
                                           n_integration_substeps=40)
        assert np.all(np.diff(t) > 0.0)
        assert t[-1] == pytest.approx(res.total_tof, rel=1e-9)
        np.testing.assert_allclose(x[-1], res.final_state, atol=1e-6)

    def test_drift_outside_action_radius_is_recorded(self, dyn: PlanarCR3BP) -> None:
        s0 = _moon_state(0.06, 1.0, dyn.mu)
        res = solve_lunar_capture_sliding_window(
            dyn, s0, sampler=lambda q: q.brute_force()[0],
            n_windows=2, n_decision_steps=4, thrust_magnitude=0.02,
            moon_action_radius=0.01, drift_t_max=0.2,
            n_integration_substeps=20, n_truth_substeps=40, max_inner_iters=1,
        )
        assert res.windows == []
        assert [seg.kind for seg in res.segments] == ["drift", "drift"]
        assert res.total_tof == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Quantum-inspired tier (simulated bifurcation)
# ---------------------------------------------------------------------------


class TestSimulatedBifurcation:
    """dSB is a heuristic; the benchmark accounts for p < 1 via TTS. What is
    pinned here is that it is a *valid* sampler (bitstrings, consistent energy,
    deterministic under a seed) and that it recovers the exact optimum on the
    small instance where SA does."""

    def _qubo(self, dyn: PlanarCR3BP, N: int = 10):
        rng = np.random.default_rng(0)
        u = rng.uniform(-0.03, 0.03, 2)
        target = dyn.propagate(STATE0, T_SPAN, n_steps=2000, control=u)[1][-1]
        cfg = ThrustSchedulingConfig(thrust_magnitude=0.05, thrust_direction="fixed",
                                     thrust_vector=u / np.linalg.norm(u))
        return build_thrust_scheduling_qubo(dyn, STATE0, target, T_SPAN,
                                            n_decision_steps=N, config=cfg)

    def test_valid_and_deterministic(self, dyn: PlanarCR3BP) -> None:
        from qalunar.qubo.scheduling_samplers import sample_simulated_bifurcation
        q = self._qubo(dyn)
        a = sample_simulated_bifurcation(q, num_reads=50, seed=3)
        b = sample_simulated_bifurcation(q, num_reads=50, seed=3)
        assert set(np.unique(a.schedule)) <= {0, 1}
        assert a.energy == pytest.approx(q.energy(a.schedule))
        np.testing.assert_array_equal(a.schedule, b.schedule)
        assert a.backend == "sb" and a.metadata["discrete"] is True

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_recovers_optimum_at_n10(self, dyn: PlanarCR3BP, seed: int) -> None:
        from qalunar.qubo.scheduling_samplers import sample_simulated_bifurcation
        q = self._qubo(dyn, N=10)
        bf = sample_brute_force(q)
        sb = sample_simulated_bifurcation(q, num_reads=100, seed=seed)
        assert sb.energy <= bf.energy * (1 + 1e-9) + 1e-15
