"""Tests for the MILP baseline solver."""

from __future__ import annotations

import numpy as np
import pytest

from qalunar.dynamics import PlanarCR3BP
from qalunar.qubo.milp_baseline import MilpResult, solve_scheduling_milp
from qalunar.qubo.thrust_scheduling import (
    ThrustSchedulingConfig,
    build_thrust_scheduling_qubo,
)


@pytest.fixture
def dynamics() -> PlanarCR3BP:
    return PlanarCR3BP()


STATE0 = np.array([-0.5, 0.3, 0.1, -0.2])


def _build_problem(
    dynamics: PlanarCR3BP, N: int = 10, fuel_weight: float = 0.0,
    thrust_mag: float = 0.05,
) -> tuple:
    u = np.array([thrust_mag * 0.5, thrust_mag * 0.3])
    _, traj = dynamics.propagate(STATE0, (0.0, 2.0), n_steps=2000, control=u)
    target = traj[-1]
    cfg = ThrustSchedulingConfig(
        thrust_magnitude=thrust_mag,
        thrust_direction="fixed",
        thrust_vector=np.array([0.5, 0.3]),
        fuel_weight=fuel_weight,
    )
    qubo = build_thrust_scheduling_qubo(
        dynamics, STATE0, target, (0.0, 2.0),
        n_decision_steps=N, config=cfg,
    )
    return qubo, target


class TestMilpAgreesWithBruteForce:
    def test_small_n_matches_brute_force(self, dynamics: PlanarCR3BP) -> None:
        """MILP energy should equal brute-force energy on a small problem."""
        qubo, _ = _build_problem(dynamics, N=10)
        bf_q, bf_e = qubo.brute_force()
        result = solve_scheduling_milp(qubo, time_limit=30.0)
        assert isinstance(result, MilpResult)
        assert result.is_optimal
        # Energies must match to high precision
        assert result.energy == pytest.approx(bf_e, rel=1e-9, abs=1e-12)

    def test_with_fuel_penalty(self, dynamics: PlanarCR3BP) -> None:
        """MILP optimum should match brute force when fuel penalty is active."""
        qubo, _ = _build_problem(dynamics, N=12, fuel_weight=0.01)
        bf_q, bf_e = qubo.brute_force()
        result = solve_scheduling_milp(qubo, time_limit=30.0)
        assert result.is_optimal
        assert result.energy == pytest.approx(bf_e, rel=1e-9, abs=1e-12)

    def test_multi_channel_with_exclusion(
        self, dynamics: PlanarCR3BP
    ) -> None:
        """MILP should respect the exclusion penalty."""
        cfg = ThrustSchedulingConfig(
            thrust_magnitude=0.05,
            thrust_channels=("tangential", "normal"),
            fuel_weight=0.0,
            exclusive=True, exclusion_penalty=10.0,
        )
        u = np.array([0.05 * 0.5, 0.05 * 0.3])
        _, traj = dynamics.propagate(STATE0, (0.0, 1.0), n_steps=2000, control=u)
        target = traj[-1]
        N = 6  # 2 channels x 6 = 12 vars
        qubo = build_thrust_scheduling_qubo(
            dynamics, STATE0, target, (0.0, 1.0),
            n_decision_steps=N, config=cfg,
        )
        bf_q, bf_e = qubo.brute_force()
        result = solve_scheduling_milp(qubo, time_limit=30.0)
        assert result.is_optimal
        assert result.energy == pytest.approx(bf_e, rel=1e-9, abs=1e-12)


class TestMilpProperties:
    def test_returns_binary_schedule(self, dynamics: PlanarCR3BP) -> None:
        qubo, _ = _build_problem(dynamics, N=8)
        result = solve_scheduling_milp(qubo, time_limit=10.0)
        # All entries should be 0 or 1
        assert set(result.schedule.tolist()).issubset({0, 1})
        assert result.schedule.shape == (qubo.n_vars,)

    def test_energy_is_consistent(self, dynamics: PlanarCR3BP) -> None:
        """The reported energy should equal qubo.energy(schedule)."""
        qubo, _ = _build_problem(dynamics, N=10)
        result = solve_scheduling_milp(qubo, time_limit=10.0)
        e_recomputed = qubo.energy(result.schedule)
        assert result.energy == pytest.approx(e_recomputed, rel=1e-12)

    def test_solve_time_recorded(self, dynamics: PlanarCR3BP) -> None:
        qubo, _ = _build_problem(dynamics, N=8)
        result = solve_scheduling_milp(qubo, time_limit=10.0)
        assert result.solve_time > 0.0
        assert result.solve_time < 10.0


class TestMilpScales:
    def test_n_15_solves(self, dynamics: PlanarCR3BP) -> None:
        """N=15 (15 binary, ~105 aux) should solve in well under the
        time limit."""
        qubo, _ = _build_problem(dynamics, N=15)
        result = solve_scheduling_milp(qubo, time_limit=30.0)
        assert result.is_optimal
        # Verify against brute force
        bf_q, bf_e = qubo.brute_force()
        assert result.energy == pytest.approx(bf_e, rel=1e-9, abs=1e-10)
