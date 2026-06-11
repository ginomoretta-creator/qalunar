"""Tests for the unified scheduling sampler interface."""

from __future__ import annotations

import numpy as np
import pytest

from qalunar.dynamics import PlanarCR3BP
from qalunar.qubo.scheduling_samplers import (
    SchedulingSampleResult,
    adapt_for_iterative,
    sample_brute_force,
    sample_simulated_annealing,
)
from qalunar.qubo.thrust_scheduling import (
    ThrustSchedulingConfig,
    build_thrust_scheduling_qubo,
    solve_iterative,
)


@pytest.fixture
def dynamics() -> PlanarCR3BP:
    return PlanarCR3BP()


STATE0 = np.array([-0.5, 0.3, 0.1, -0.2])


def _build(dynamics: PlanarCR3BP, N: int = 10):
    u = np.array([0.025, 0.015])
    _, traj = dynamics.propagate(STATE0, (0.0, 1.5), n_steps=2000, control=u)
    target = traj[-1]
    cfg = ThrustSchedulingConfig(
        thrust_magnitude=0.05, thrust_direction="fixed",
        thrust_vector=np.array([0.5, 0.3]), fuel_weight=0.0,
    )
    return build_thrust_scheduling_qubo(
        dynamics, STATE0, target, (0.0, 1.5),
        n_decision_steps=N, config=cfg,
    ), target


class TestBruteForceSampler:
    def test_returns_correct_type(self, dynamics: PlanarCR3BP) -> None:
        qubo, _ = _build(dynamics, N=8)
        result = sample_brute_force(qubo)
        assert isinstance(result, SchedulingSampleResult)
        assert result.backend == "brute_force"
        assert result.schedule.shape == (qubo.n_vars,)

    def test_finds_global_minimum(self, dynamics: PlanarCR3BP) -> None:
        qubo, _ = _build(dynamics, N=10)
        result = sample_brute_force(qubo)
        # Verify against direct call
        bf_q, bf_e = qubo.brute_force()
        assert result.energy == pytest.approx(bf_e, rel=1e-12)
        np.testing.assert_array_equal(result.schedule, bf_q)


class TestSASampler:
    def test_returns_correct_type(self, dynamics: PlanarCR3BP) -> None:
        qubo, _ = _build(dynamics, N=10)
        result = sample_simulated_annealing(qubo, num_reads=200)
        assert isinstance(result, SchedulingSampleResult)
        assert result.backend == "sa"
        assert result.metadata["num_reads"] == 200

    def test_matches_brute_force_on_small_problem(
        self, dynamics: PlanarCR3BP
    ) -> None:
        qubo, _ = _build(dynamics, N=10)
        bf = sample_brute_force(qubo)
        sa = sample_simulated_annealing(qubo, num_reads=500, seed=0)
        # SA should hit the global minimum on a small problem
        assert sa.energy <= bf.energy + 1e-9


class TestAdapter:
    def test_adapter_strips_metadata(self, dynamics: PlanarCR3BP) -> None:
        """adapt_for_iterative should turn a sampler-with-metadata into
        one returning just a bitstring."""
        qubo, _ = _build(dynamics, N=8)
        wrapped = adapt_for_iterative(sample_brute_force)
        bitstring = wrapped(qubo)
        assert isinstance(bitstring, np.ndarray)
        assert bitstring.shape == (qubo.n_vars,)
        # Must match the underlying call
        direct = sample_brute_force(qubo).schedule
        np.testing.assert_array_equal(bitstring, direct)

    def test_adapter_works_with_solve_iterative(
        self, dynamics: PlanarCR3BP
    ) -> None:
        """Sampler from this module should be usable as solve_iterative
        sampler argument."""
        qubo, target = _build(dynamics, N=10)
        cfg = ThrustSchedulingConfig(
            thrust_magnitude=0.05, thrust_direction="fixed",
            thrust_vector=np.array([0.5, 0.3]), fuel_weight=0.0,
        )

        result = solve_iterative(
            dynamics, STATE0, target, (0.0, 1.5),
            n_decision_steps=10,
            sampler=adapt_for_iterative(sample_brute_force),
            config=cfg, max_iters=5,
        )
        assert result.schedule.shape == (10,)
