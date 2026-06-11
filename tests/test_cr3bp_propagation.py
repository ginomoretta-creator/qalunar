"""Tests for CR3BP propagation and state transition matrix."""

from __future__ import annotations

import numpy as np
import pytest

from qalunar.dynamics import PlanarCR3BP


@pytest.fixture
def cr3bp() -> PlanarCR3BP:
    return PlanarCR3BP()


# A benign state away from both primaries
BENIGN_STATE = np.array([-0.5, 0.3, 0.1, -0.2])


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------


class TestPropagate:
    def test_initial_state_preserved(self, cr3bp: PlanarCR3BP) -> None:
        t, states = cr3bp.propagate(BENIGN_STATE, (0.0, 1.0), n_steps=100)
        np.testing.assert_array_equal(states[0], BENIGN_STATE)

    def test_output_shapes(self, cr3bp: PlanarCR3BP) -> None:
        n = 200
        t, states = cr3bp.propagate(BENIGN_STATE, (0.0, 2.0), n_steps=n)
        assert t.shape == (n + 1,)
        assert states.shape == (n + 1, 4)

    def test_jacobi_integral_conserved(self, cr3bp: PlanarCR3BP) -> None:
        """Unforced propagation should conserve the Jacobi integral."""
        _, states = cr3bp.propagate(BENIGN_STATE, (0.0, 3.0), n_steps=10000)
        cj0 = cr3bp.jacobi_integral(states[0])
        for i in range(len(states)):
            cj = cr3bp.jacobi_integral(states[i])
            assert abs(cj - cj0) < 1e-8, f"step {i}: Jacobi drift {abs(cj - cj0):.3e}"

    def test_control_changes_trajectory(self, cr3bp: PlanarCR3BP) -> None:
        """Applying thrust should produce a different final state."""
        _, states_coast = cr3bp.propagate(BENIGN_STATE, (0.0, 1.0), n_steps=500)
        u = np.array([0.01, 0.0])
        _, states_thrust = cr3bp.propagate(BENIGN_STATE, (0.0, 1.0), n_steps=500, control=u)
        # States should diverge
        diff = np.max(np.abs(states_coast[-1] - states_thrust[-1]))
        assert diff > 1e-3

    def test_l4_stays_at_rest(self, cr3bp: PlanarCR3BP) -> None:
        """Propagation from L4 at rest should stay put."""
        l4, _ = cr3bp.lagrange_triangular_points()
        state0 = np.array([l4[0], l4[1], 0.0, 0.0])
        _, states = cr3bp.propagate(state0, (0.0, 5.0), n_steps=5000)
        np.testing.assert_allclose(states[-1], state0, atol=1e-10)


# ---------------------------------------------------------------------------
# State Transition Matrix
# ---------------------------------------------------------------------------


class TestPropagateSTM:
    def test_initial_stm_is_identity(self, cr3bp: PlanarCR3BP) -> None:
        _, _, stms = cr3bp.propagate_stm(BENIGN_STATE, (0.0, 1.0), n_steps=100)
        np.testing.assert_allclose(stms[0], np.eye(4), atol=1e-15)

    def test_output_shapes(self, cr3bp: PlanarCR3BP) -> None:
        n = 200
        t, states, stms = cr3bp.propagate_stm(BENIGN_STATE, (0.0, 2.0), n_steps=n)
        assert t.shape == (n + 1,)
        assert states.shape == (n + 1, 4)
        assert stms.shape == (n + 1, 4, 4)

    def test_states_match_propagate(self, cr3bp: PlanarCR3BP) -> None:
        """State trajectory from propagate_stm should match propagate."""
        _, states1 = cr3bp.propagate(BENIGN_STATE, (0.0, 2.0), n_steps=500)
        _, states2, _ = cr3bp.propagate_stm(BENIGN_STATE, (0.0, 2.0), n_steps=500)
        np.testing.assert_allclose(states1, states2, atol=1e-13)

    def test_stm_matches_finite_differences(self, cr3bp: PlanarCR3BP) -> None:
        """STM Phi(tf, t0) should map perturbations: dx(tf) ≈ Phi * dx(t0).

        This is the gold-standard test: perturb the initial state in each
        direction, propagate, and verify the final-state deviation matches
        Phi @ perturbation.
        """
        tf = 1.5
        n_steps = 1500
        eps = 1e-7

        _, states_ref, stms = cr3bp.propagate_stm(BENIGN_STATE, (0.0, tf), n_steps=n_steps)
        phi_tf = stms[-1]  # Phi(tf, t0)

        for j in range(4):
            dx0 = np.zeros(4)
            dx0[j] = eps
            _, states_plus = cr3bp.propagate(BENIGN_STATE + dx0, (0.0, tf), n_steps=n_steps)
            _, states_minus = cr3bp.propagate(BENIGN_STATE - dx0, (0.0, tf), n_steps=n_steps)

            # Central difference: d(state_f)/d(state_0[j])
            fd_col = (states_plus[-1] - states_minus[-1]) / (2 * eps)
            np.testing.assert_allclose(
                phi_tf[:, j], fd_col, atol=1e-4, rtol=1e-4,
                err_msg=f"STM column {j} doesn't match finite differences"
            )

    def test_stm_determinant_one(self, cr3bp: PlanarCR3BP) -> None:
        """The CR3BP is Hamiltonian, so det(Phi) = 1 (symplecticity)."""
        _, _, stms = cr3bp.propagate_stm(BENIGN_STATE, (0.0, 3.0), n_steps=3000)
        for i in [0, 500, 1000, 2000, 3000]:
            det = np.linalg.det(stms[i])
            assert abs(det - 1.0) < 1e-6, f"step {i}: det(Phi) = {det:.8f}"

    def test_stm_composition(self, cr3bp: PlanarCR3BP) -> None:
        """Phi(t2, t0) = Phi(t2, t1) @ Phi(t1, t0) for any intermediate t1.

        We compute two separate propagations and verify the composition rule.
        """
        t_mid = 1.0
        t_end = 2.0
        n = 1000

        # Full propagation t0 -> t_end
        _, _, stms_full = cr3bp.propagate_stm(BENIGN_STATE, (0.0, t_end), n_steps=2 * n)
        phi_full = stms_full[-1]

        # First leg: t0 -> t_mid
        _, states1, stms1 = cr3bp.propagate_stm(BENIGN_STATE, (0.0, t_mid), n_steps=n)
        phi_01 = stms1[-1]

        # Second leg: t_mid -> t_end (starting from the state at t_mid)
        _, _, stms2 = cr3bp.propagate_stm(states1[-1], (t_mid, t_end), n_steps=n)
        phi_12 = stms2[-1]

        # Composition
        phi_composed = phi_12 @ phi_01
        np.testing.assert_allclose(phi_full, phi_composed, atol=1e-8)
