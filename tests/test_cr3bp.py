"""Tests for the planar CR3BP dynamics module."""

from __future__ import annotations

import numpy as np
import pytest

from qalunar.dynamics import EARTH_MOON_MU, PlanarCR3BP


@pytest.fixture
def cr3bp() -> PlanarCR3BP:
    return PlanarCR3BP()


# ---------------------------------------------------------------------------
# Construction and bookkeeping
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_mu_is_earth_moon(self, cr3bp: PlanarCR3BP) -> None:
        assert cr3bp.mu == EARTH_MOON_MU

    def test_dimensions(self, cr3bp: PlanarCR3BP) -> None:
        assert cr3bp.n_state == 4
        assert cr3bp.n_control == 2

    def test_primary_positions(self, cr3bp: PlanarCR3BP) -> None:
        np.testing.assert_allclose(cr3bp.earth_position(), [-EARTH_MOON_MU, 0.0])
        np.testing.assert_allclose(cr3bp.moon_position(), [1.0 - EARTH_MOON_MU, 0.0])

    def test_custom_mu(self) -> None:
        sun_earth = PlanarCR3BP(mu=3.0404233e-6)
        assert sun_earth.mu == pytest.approx(3.0404233e-6)
        np.testing.assert_allclose(sun_earth.earth_position(), [-3.0404233e-6, 0.0])


# ---------------------------------------------------------------------------
# Right-hand side
# ---------------------------------------------------------------------------


class TestRHS:
    def test_equilibrium_l4_is_fixed_point(self, cr3bp: PlanarCR3BP) -> None:
        l4, _ = cr3bp.lagrange_triangular_points()
        state = np.array([l4[0], l4[1], 0.0, 0.0])
        np.testing.assert_allclose(cr3bp.rhs(state), np.zeros(4), atol=1e-14)

    def test_equilibrium_l5_is_fixed_point(self, cr3bp: PlanarCR3BP) -> None:
        _, l5 = cr3bp.lagrange_triangular_points()
        state = np.array([l5[0], l5[1], 0.0, 0.0])
        np.testing.assert_allclose(cr3bp.rhs(state), np.zeros(4), atol=1e-14)

    def test_velocity_block_is_identity(self, cr3bp: PlanarCR3BP) -> None:
        """The first two rows of the RHS should just copy the velocities."""
        state = np.array([0.7, -0.3, 0.42, -0.17])
        f = cr3bp.rhs(state)
        assert f[0] == state[2]
        assert f[1] == state[3]

    def test_control_enters_linearly(self, cr3bp: PlanarCR3BP) -> None:
        state = np.array([0.3, 0.2, 0.1, -0.05])
        u1 = np.array([0.01, -0.02])
        u2 = np.array([0.04, 0.03])

        f0 = cr3bp.rhs(state)
        f1 = cr3bp.rhs(state, u1)
        f2 = cr3bp.rhs(state, u2)
        f_sum = cr3bp.rhs(state, u1 + u2)

        # Only the acceleration rows can change, and they change linearly.
        np.testing.assert_allclose(f1[2:] - f0[2:], u1, atol=1e-15)
        np.testing.assert_allclose(f2[2:] - f0[2:], u2, atol=1e-15)
        np.testing.assert_allclose(f_sum[2:] - f0[2:], u1 + u2, atol=1e-15)
        # Velocity rows are unaffected by control.
        np.testing.assert_allclose(f1[:2], f0[:2])
        np.testing.assert_allclose(f2[:2], f0[:2])


# ---------------------------------------------------------------------------
# Jacobians
# ---------------------------------------------------------------------------


class TestJacobians:
    @pytest.mark.parametrize("seed", [0, 1, 7, 42, 99, 2024])
    def test_state_jacobian_matches_central_differences(
        self, cr3bp: PlanarCR3BP, seed: int
    ) -> None:
        """Analytical df/dx should match central finite differences.

        We reject samples that fall too close to either primary because the
        1/r**5 terms amplify rounding noise there.
        """
        local_rng = np.random.default_rng(seed)
        for _ in range(100):
            state = local_rng.uniform(-1.5, 1.5, size=4)
            r1 = np.hypot(state[0] + cr3bp.mu, state[1])
            r2 = np.hypot(state[0] - 1.0 + cr3bp.mu, state[1])
            if r1 > 0.15 and r2 > 0.15:
                break
        else:
            pytest.skip("could not find a state sufficiently far from both primaries")

        analytic = cr3bp.jacobian_state(state)
        eps = 1e-6
        numerical = np.zeros_like(analytic)
        for j in range(4):
            dx = np.zeros(4)
            dx[j] = eps
            f_plus = cr3bp.rhs(state + dx)
            f_minus = cr3bp.rhs(state - dx)
            numerical[:, j] = (f_plus - f_minus) / (2.0 * eps)

        np.testing.assert_allclose(analytic, numerical, atol=1e-7, rtol=1e-6)

    def test_state_jacobian_is_control_independent(self, cr3bp: PlanarCR3BP) -> None:
        """Adding control should not change df/dx."""
        state = np.array([0.3, 0.2, 0.1, -0.05])
        # Control-free Jacobian
        j_nocontrol = cr3bp.jacobian_state(state)
        # Dynamics ARE affected by control, but df/dx shouldn't be. Verify
        # by finite-differencing f(x, u) at the same u and checking the
        # result matches j_nocontrol.
        u = np.array([0.03, -0.01])
        eps = 1e-6
        numerical = np.zeros((4, 4))
        for j in range(4):
            dx = np.zeros(4)
            dx[j] = eps
            f_plus = cr3bp.rhs(state + dx, u)
            f_minus = cr3bp.rhs(state - dx, u)
            numerical[:, j] = (f_plus - f_minus) / (2.0 * eps)
        np.testing.assert_allclose(j_nocontrol, numerical, atol=1e-7, rtol=1e-6)

    def test_control_jacobian_is_constant(self, cr3bp: PlanarCR3BP) -> None:
        expected = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        np.testing.assert_array_equal(cr3bp.jacobian_control(), expected)
        # Pass a state: interface accepts it but output is identical.
        state = np.array([0.5, 0.3, -0.1, 0.2])
        np.testing.assert_array_equal(cr3bp.jacobian_control(state), expected)


# ---------------------------------------------------------------------------
# Jacobi integral
# ---------------------------------------------------------------------------


class TestJacobiIntegral:
    def test_l4_value(self, cr3bp: PlanarCR3BP) -> None:
        """At L4 with zero velocity, C_J equals exactly 3, independent of mu.

        With our convention
          two_omega = x**2 + y**2 + 2*(1-mu)/r1 + 2*mu/r2 + mu*(1-mu),
        evaluated at L4 = (1/2 - mu, sqrt(3)/2) with r1 = r2 = 1 and v = 0,

          C_J = (1 - mu + mu**2) + 2*(1-mu) + 2*mu + mu*(1-mu)
              = 1 - mu + mu**2 + 2 - 2*mu + 2*mu + mu - mu**2
              = 3.

        The mu*(1-mu) bookkeeping constant exactly cancels the
        (-mu + mu**2) contribution from ``x**2 + y**2``.
        """
        l4, _ = cr3bp.lagrange_triangular_points()
        state = np.array([l4[0], l4[1], 0.0, 0.0])
        cj = cr3bp.jacobi_integral(state)
        assert cj == pytest.approx(3.0, abs=1e-14)

    def test_conserved_under_rk4_drift(self, cr3bp: PlanarCR3BP) -> None:
        """Pure-drift RK4 should keep C_J constant to global RK4 accuracy.

        Starting from a benign state well away from both primaries. Over
        1000 steps at dt=1e-3 the RK4 global error should push C_J by less
        than ~1e-8 for a moderately nonlinear trajectory.
        """
        state = np.array([-1.0, 0.0, 0.0, 0.5])
        dt = 1.0e-3
        n_steps = 1000

        cj0 = cr3bp.jacobi_integral(state)
        x = state.copy()
        max_drift = 0.0
        for _ in range(n_steps):
            k1 = cr3bp.rhs(x)
            k2 = cr3bp.rhs(x + 0.5 * dt * k1)
            k3 = cr3bp.rhs(x + 0.5 * dt * k2)
            k4 = cr3bp.rhs(x + dt * k3)
            x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            max_drift = max(max_drift, abs(cr3bp.jacobi_integral(x) - cj0))

        assert max_drift < 1e-8, f"Jacobi drift {max_drift:.3e} exceeds 1e-8"
