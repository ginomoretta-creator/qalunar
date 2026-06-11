"""Tests for the indirect TFC+ELM transcription of the planar CR3BP."""

from __future__ import annotations

import numpy as np
import pytest

from qalunar.dynamics import PlanarCR3BP
from qalunar.transcription import (
    IndirectTfcElmConfig,
    IndirectTfcElmTranscription,
    OuterLoopConfig,
    SequentialSolution,
)
from qalunar.transcription.indirect_tfc_elm import (
    _hermite_switching,
    _omega_grad_hess,
    _tanh_basis,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_bcs() -> dict[str, np.ndarray]:
    """A modest transfer near the Earth-Moon barycenter.

    Chosen so the trajectory stays away from either primary and the
    nonlinear gravity terms remain mild, so the linearized system is
    well-behaved and tests can pin concrete residual thresholds.
    """
    return dict(
        r0=np.array([-0.3, 0.0]),
        v0=np.array([0.0, 0.6]),
        rf=np.array([0.4, 0.2]),
        vf=np.array([-0.1, 0.0]),
        time_of_flight=2.5,
    )


@pytest.fixture
def transcription(simple_bcs) -> IndirectTfcElmTranscription:
    # n=20 training points, L=80 basis functions matches Gino's MATLAB
    # default. With L >> n the linear system has full row rank, so the
    # classical LSQ reference residual is driven to essentially zero.
    cfg = IndirectTfcElmConfig(n_training=20, n_basis=80, seed=7)
    return IndirectTfcElmTranscription(
        dynamics=PlanarCR3BP(),
        config=cfg,
        **simple_bcs,
    )


# ---------------------------------------------------------------------------
# Hermite switching functions
# ---------------------------------------------------------------------------


class TestHermiteSwitching:
    def test_endpoint_values(self) -> None:
        z = np.array([-1.0, 1.0])
        (om1, om2, om3, om4), _, _ = _hermite_switching(z)
        np.testing.assert_allclose(om1, [1.0, 0.0], atol=1e-14)
        np.testing.assert_allclose(om2, [0.0, 1.0], atol=1e-14)
        np.testing.assert_allclose(om3, [0.0, 0.0], atol=1e-14)
        np.testing.assert_allclose(om4, [0.0, 0.0], atol=1e-14)

    def test_endpoint_first_derivatives(self) -> None:
        z = np.array([-1.0, 1.0])
        _, (om1d, om2d, om3d, om4d), _ = _hermite_switching(z)
        np.testing.assert_allclose(om1d, [0.0, 0.0], atol=1e-14)
        np.testing.assert_allclose(om2d, [0.0, 0.0], atol=1e-14)
        np.testing.assert_allclose(om3d, [1.0, 0.0], atol=1e-14)
        np.testing.assert_allclose(om4d, [0.0, 1.0], atol=1e-14)

    def test_derivatives_match_numerical(self) -> None:
        """Analytic derivatives agree with central differences."""
        z = np.linspace(-0.9, 0.9, 50)
        eps = 1e-5
        (om1, om2, om3, om4), (om1d, om2d, om3d, om4d), (om1dd, om2dd, om3dd, om4dd) = (
            _hermite_switching(z)
        )
        (om1_p, om2_p, om3_p, om4_p), _, _ = _hermite_switching(z + eps)
        (om1_m, om2_m, om3_m, om4_m), _, _ = _hermite_switching(z - eps)
        for analytic, plus, minus in (
            (om1d, om1_p, om1_m),
            (om2d, om2_p, om2_m),
            (om3d, om3_p, om3_m),
            (om4d, om4_p, om4_m),
        ):
            num_d = (plus - minus) / (2 * eps)
            np.testing.assert_allclose(analytic, num_d, atol=1e-7)


# ---------------------------------------------------------------------------
# Tanh ELM basis
# ---------------------------------------------------------------------------


class TestTanhBasis:
    def test_shapes(self) -> None:
        z = np.linspace(-1.0, 1.0, 5)
        w = np.array([0.5, -1.0, 2.0])
        b = np.array([0.1, -0.2, 0.0])
        h, hd, hdd = _tanh_basis(z, w, b)
        assert h.shape == (5, 3)
        assert hd.shape == (5, 3)
        assert hdd.shape == (5, 3)

    def test_values_against_definition(self) -> None:
        z = np.array([-0.3, 0.1, 0.7])
        w = np.array([1.5])
        b = np.array([-0.4])
        h, hd, hdd = _tanh_basis(z, w, b)
        expected_h = np.tanh(1.5 * z - 0.4).reshape(-1, 1)
        expected_hd = 1.5 * (1.0 - expected_h ** 2)
        expected_hdd = -2.0 * 1.5 ** 2 * expected_h * (1.0 - expected_h ** 2)
        np.testing.assert_allclose(h, expected_h, atol=1e-14)
        np.testing.assert_allclose(hd, expected_hd, atol=1e-14)
        np.testing.assert_allclose(hdd, expected_hdd, atol=1e-14)

    def test_derivatives_match_numerical(self) -> None:
        z = np.linspace(-0.9, 0.9, 40)
        w = np.array([0.7, -1.3])
        b = np.array([0.2, -0.1])
        eps = 1e-6
        h, hd, hdd = _tanh_basis(z, w, b)
        h_p, _, _ = _tanh_basis(z + eps, w, b)
        h_m, _, _ = _tanh_basis(z - eps, w, b)
        num_hd = (h_p - h_m) / (2 * eps)
        np.testing.assert_allclose(hd, num_hd, atol=1e-7)

        _, hd_p, _ = _tanh_basis(z + eps, w, b)
        _, hd_m, _ = _tanh_basis(z - eps, w, b)
        num_hdd = (hd_p - hd_m) / (2 * eps)
        np.testing.assert_allclose(hdd, num_hdd, atol=1e-7)


# ---------------------------------------------------------------------------
# Omega gradient/Hessian helper
# ---------------------------------------------------------------------------


class TestOmegaGradHess:
    def test_matches_cr3bp_jacobian_state(self) -> None:
        """Hessian of Omega (rows 2-3 of df/dx) agrees with jacobian_state."""
        cr3bp = PlanarCR3BP()
        rng = np.random.default_rng(0)
        for _ in range(10):
            x = float(rng.uniform(-1.0, 1.0))
            y = float(rng.uniform(-1.0, 1.0))
            # Skip points too close to either primary to avoid 1/r^5 blowup.
            r1 = np.hypot(x + cr3bp.mu, y)
            r2 = np.hypot(x - (1.0 - cr3bp.mu), y)
            if r1 < 0.2 or r2 < 0.2:
                continue
            _, _, omxx, omxy, omyy = _omega_grad_hess(
                cr3bp.mu, np.array([x]), np.array([y])
            )
            J = cr3bp.jacobian_state(np.array([x, y, 0.0, 0.0]))
            assert omxx[0] == pytest.approx(J[2, 0], rel=1e-12)
            assert omxy[0] == pytest.approx(J[2, 1], rel=1e-12)
            assert omyy[0] == pytest.approx(J[3, 1], rel=1e-12)

    def test_gradient_matches_rhs(self) -> None:
        """Omega gradient matches the (ax, ay) from rhs() at zero velocity."""
        cr3bp = PlanarCR3BP()
        x, y = 0.3, -0.1
        omx, omy, *_ = _omega_grad_hess(cr3bp.mu, np.array([x]), np.array([y]))
        # rhs at zero velocity: ax = 2*0 + Omega_x, ay = -2*0 + Omega_y.
        f = cr3bp.rhs(np.array([x, y, 0.0, 0.0]))
        assert f[2] == pytest.approx(omx[0], rel=1e-12)
        assert f[3] == pytest.approx(omy[0], rel=1e-12)


# ---------------------------------------------------------------------------
# Transcription construction and shape sanity
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_sizes(self, transcription) -> None:
        n = transcription.n_training
        L = transcription.n_basis
        assert transcription.F.shape == (n, L)
        assert transcription.Fd.shape == (n, L)
        assert transcription.Fdd.shape == (n, L)
        assert transcription.H.shape == (n, L)
        assert transcription.Hd.shape == (n, L)
        assert transcription.Cx.shape == (n,)
        assert transcription.Cxd.shape == (n,)
        assert transcription.Cxdd.shape == (n,)
        assert transcription.n_equations == 6 * n
        assert transcription.n_unknowns == 6 * L

    def test_time_grid_spans_flight(self, transcription, simple_bcs) -> None:
        assert transcription.t[0] == pytest.approx(0.0)
        assert transcription.t[-1] == pytest.approx(simple_bcs["time_of_flight"])

    def test_c_scaling(self, transcription, simple_bcs) -> None:
        expected = 2.0 / simple_bcs["time_of_flight"]
        assert transcription.c == pytest.approx(expected)

    def test_rejects_invalid_config(self) -> None:
        dyn = PlanarCR3BP()
        r0 = np.zeros(2)
        v0 = np.zeros(2)
        rf = np.ones(2)
        vf = np.zeros(2)
        with pytest.raises(ValueError, match="n_training"):
            IndirectTfcElmTranscription(
                dynamics=dyn,
                r0=r0,
                v0=v0,
                rf=rf,
                vf=vf,
                time_of_flight=1.0,
                config=IndirectTfcElmConfig(n_training=1),
            )
        with pytest.raises(ValueError, match="n_basis"):
            IndirectTfcElmTranscription(
                dynamics=dyn,
                r0=r0,
                v0=v0,
                rf=rf,
                vf=vf,
                time_of_flight=1.0,
                config=IndirectTfcElmConfig(n_basis=0),
            )
        with pytest.raises(ValueError, match="time_of_flight"):
            IndirectTfcElmTranscription(
                dynamics=dyn,
                r0=r0,
                v0=v0,
                rf=rf,
                vf=vf,
                time_of_flight=0.0,
            )
        with pytest.raises(ValueError, match="length-2"):
            IndirectTfcElmTranscription(
                dynamics=dyn,
                r0=np.zeros(3),
                v0=v0,
                rf=rf,
                vf=vf,
                time_of_flight=1.0,
            )


# ---------------------------------------------------------------------------
# Constrained expression: BCs are enforced *by construction*
# ---------------------------------------------------------------------------


class TestBoundaryConditions:
    def test_F_is_zero_at_endpoints(self, transcription) -> None:
        np.testing.assert_allclose(transcription.F[0, :], 0.0, atol=1e-13)
        np.testing.assert_allclose(transcription.F[-1, :], 0.0, atol=1e-13)

    def test_Fd_is_zero_at_endpoints(self, transcription) -> None:
        np.testing.assert_allclose(transcription.Fd[0, :], 0.0, atol=1e-12)
        np.testing.assert_allclose(transcription.Fd[-1, :], 0.0, atol=1e-12)

    def test_C_endpoint_values_are_BCs(self, transcription, simple_bcs) -> None:
        assert transcription.Cx[0] == pytest.approx(simple_bcs["r0"][0])
        assert transcription.Cx[-1] == pytest.approx(simple_bcs["rf"][0])
        assert transcription.Cy[0] == pytest.approx(simple_bcs["r0"][1])
        assert transcription.Cy[-1] == pytest.approx(simple_bcs["rf"][1])

    def test_C_endpoint_derivatives_are_velocities(
        self, transcription, simple_bcs
    ) -> None:
        assert transcription.Cxd[0] == pytest.approx(simple_bcs["v0"][0], abs=1e-12)
        assert transcription.Cxd[-1] == pytest.approx(simple_bcs["vf"][0], abs=1e-12)
        assert transcription.Cyd[0] == pytest.approx(simple_bcs["v0"][1], abs=1e-12)
        assert transcription.Cyd[-1] == pytest.approx(simple_bcs["vf"][1], abs=1e-12)

    def test_decoded_trajectory_hits_BCs_for_random_xi(
        self, transcription, simple_bcs
    ) -> None:
        """Decoding ANY xi must give BCs exactly -- the defining property of TFC."""
        rng = np.random.default_rng(11)
        xi = rng.standard_normal(transcription.n_unknowns)
        traj = transcription.decode_trajectory(xi)
        assert traj["x"][0] == pytest.approx(simple_bcs["r0"][0], abs=1e-11)
        assert traj["x"][-1] == pytest.approx(simple_bcs["rf"][0], abs=1e-11)
        assert traj["y"][0] == pytest.approx(simple_bcs["r0"][1], abs=1e-11)
        assert traj["y"][-1] == pytest.approx(simple_bcs["rf"][1], abs=1e-11)
        assert traj["vx"][0] == pytest.approx(simple_bcs["v0"][0], abs=1e-11)
        assert traj["vx"][-1] == pytest.approx(simple_bcs["vf"][0], abs=1e-11)
        assert traj["vy"][0] == pytest.approx(simple_bcs["v0"][1], abs=1e-11)
        assert traj["vy"][-1] == pytest.approx(simple_bcs["vf"][1], abs=1e-11)


# ---------------------------------------------------------------------------
# Linear system shape and solvability
# ---------------------------------------------------------------------------


class TestLinearSystem:
    def test_shapes(self, transcription) -> None:
        nom_x, nom_y = transcription.initial_nominal()
        A, B = transcription.build_linear_system(nom_x, nom_y)
        assert A.shape == (transcription.n_equations, transcription.n_unknowns)
        assert B.shape == (transcription.n_equations,)

    def test_rejects_wrong_nominal_shape(self, transcription) -> None:
        with pytest.raises(ValueError, match="length n"):
            transcription.build_linear_system(
                np.zeros(transcription.n_training - 1),
                np.zeros(transcription.n_training),
            )

    def test_costate_rows_have_zero_rhs(self, transcription) -> None:
        nom_x, nom_y = transcription.initial_nominal()
        _, B = transcription.build_linear_system(nom_x, nom_y)
        n = transcription.n_training
        np.testing.assert_array_equal(B[2 * n :], 0.0)

    def test_lsq_solution_has_small_residual(self, transcription) -> None:
        """A full-row-rank lsq solve should drive ||A xi - B|| to ~ 0.

        With L >> n the system is underdetermined and full row rank, so
        the lsq residual is bounded only by the condition number of A
        (which is large for tanh ELMs with weights in [-3, 3]). We test
        against a relative threshold to ||B||.
        """
        nom_x, nom_y = transcription.initial_nominal()
        A, B = transcription.build_linear_system(nom_x, nom_y)
        xi, *_ = np.linalg.lstsq(A, B, rcond=None)
        residual = np.linalg.norm(A @ xi - B)
        assert residual < 1e-8 * np.linalg.norm(B)

    def test_lsq_solution_satisfies_BCs(self, transcription, simple_bcs) -> None:
        """TFC guarantees BCs hold regardless of the solver's choice of xi."""
        nom_x, nom_y = transcription.initial_nominal()
        A, B = transcription.build_linear_system(nom_x, nom_y)
        xi, *_ = np.linalg.lstsq(A, B, rcond=None)
        traj = transcription.decode_trajectory(xi)
        assert traj["x"][0] == pytest.approx(simple_bcs["r0"][0], abs=1e-10)
        assert traj["x"][-1] == pytest.approx(simple_bcs["rf"][0], abs=1e-10)
        assert traj["vx"][-1] == pytest.approx(simple_bcs["vf"][0], abs=1e-10)
        assert traj["vy"][0] == pytest.approx(simple_bcs["v0"][1], abs=1e-10)


# ---------------------------------------------------------------------------
# Physical correctness: linearized dynamics residuals at the nominal
# ---------------------------------------------------------------------------


class TestPhysicalResidual:
    def test_lsq_trajectory_satisfies_linearized_dynamics(
        self, transcription
    ) -> None:
        """After one lsq solve, the linearized dynamics residual is ~ 0."""
        nom_x, nom_y = transcription.initial_nominal()
        A, B = transcription.build_linear_system(nom_x, nom_y)
        xi, *_ = np.linalg.lstsq(A, B, rcond=None)
        traj = transcription.decode_trajectory(xi)

        # Recompute the Hessian/gradient at the same nominal we built with.
        omx, omy, omxx, omxy, omyy = _omega_grad_hess(
            transcription.dynamics.mu, nom_x, nom_y
        )
        # Second-derivative of x and y at the training points.
        xi_x = xi[: transcription.n_basis]
        xi_y = xi[transcription.n_basis : 2 * transcription.n_basis]
        ax = transcription.Fdd @ xi_x + transcription.Cxdd
        ay = transcription.Fdd @ xi_y + transcription.Cydd

        # Linearized equations: ddx - 2 vy - (Omega_x_lin) + lambda_vx = 0
        # where Omega_x_lin = Omega_x(nom) + Omega_xx (x - nom_x) + Omega_xy (y - nom_y).
        omx_lin = omx + omxx * (traj["x"] - nom_x) + omxy * (traj["y"] - nom_y)
        omy_lin = omy + omxy * (traj["x"] - nom_x) + omyy * (traj["y"] - nom_y)

        res_x = ax - 2.0 * traj["vy"] - omx_lin + traj["lambda_vx"]
        res_y = ay + 2.0 * traj["vx"] - omy_lin + traj["lambda_vy"]

        assert np.max(np.abs(res_x)) < 1e-8
        assert np.max(np.abs(res_y)) < 1e-8

    def test_costate_linearized_dynamics_satisfied(self, transcription) -> None:
        """Same for the four costate ODEs."""
        nom_x, nom_y = transcription.initial_nominal()
        A, B = transcription.build_linear_system(nom_x, nom_y)
        xi, *_ = np.linalg.lstsq(A, B, rcond=None)
        traj = transcription.decode_trajectory(xi)

        _, _, omxx, omxy, omyy = _omega_grad_hess(
            transcription.dynamics.mu, nom_x, nom_y
        )
        # d lambda_rx/dt needs Hd @ xi_lrx.
        L = transcription.n_basis
        xi_lrx = xi[2 * L : 3 * L]
        xi_lry = xi[3 * L : 4 * L]
        xi_lvx = xi[4 * L : 5 * L]
        xi_lvy = xi[5 * L : 6 * L]
        dlrx = transcription.Hd @ xi_lrx
        dlry = transcription.Hd @ xi_lry
        dlvx = transcription.Hd @ xi_lvx
        dlvy = transcription.Hd @ xi_lvy

        res_lrx = dlrx + omxx * traj["lambda_vx"] + omxy * traj["lambda_vy"]
        res_lry = dlry + omxy * traj["lambda_vx"] + omyy * traj["lambda_vy"]
        res_lvx = dlvx + traj["lambda_rx"] - 2.0 * traj["lambda_vy"]
        res_lvy = dlvy + traj["lambda_ry"] + 2.0 * traj["lambda_vx"]

        for r in (res_lrx, res_lry, res_lvx, res_lvy):
            assert np.max(np.abs(r)) < 1e-8


# ---------------------------------------------------------------------------
# Integration with build_linear_lsq_qubo
# ---------------------------------------------------------------------------


class TestQuboIntegration:
    def test_qubo_energy_identity_on_warm_start(self, transcription) -> None:
        """Wiring into build_linear_lsq_qubo preserves the core energy identity."""
        from qalunar.qubo import LinearLsqToQuboConfig, build_linear_lsq_qubo

        nom_x, nom_y = transcription.initial_nominal()
        A, B = transcription.build_linear_system(nom_x, nom_y)

        cfg = LinearLsqToQuboConfig(bits_per_variable=6)
        result = build_linear_lsq_qubo(A, B, cfg)

        A_aug = np.vstack(
            [A, np.sqrt(result.lambda_tikhonov) * np.eye(A.shape[1])]
        )
        B_aug = np.concatenate([B, np.zeros(A.shape[1])])
        q = result.warm_start_bits()
        xi = result.decode_xi(q)
        expected = float(np.sum((A_aug @ xi - B_aug) ** 2))
        assert result.energy(q) == pytest.approx(expected, rel=1e-9, abs=1e-9)


# ---------------------------------------------------------------------------
# Nonlinear residual
# ---------------------------------------------------------------------------


class TestNonlinearResidual:
    def test_keys_and_shapes(self, transcription) -> None:
        xi = np.zeros(transcription.n_unknowns)
        res = transcription.nonlinear_residual(xi)
        assert set(res.keys()) == {"ddx", "ddy", "dlrx", "dlry", "dlvx", "dlvy"}
        for v in res.values():
            assert v.shape == (transcription.n_training,)

    def test_costate_v_rows_are_linear_in_xi(self, transcription) -> None:
        """dlvx and dlvy don't involve Omega, so they are linear in xi.

        Setting xi = 0 gives dlvx = dlvy = 0 identically since H, Hd,
        lambda_r, and lambda_v all vanish.
        """
        xi = np.zeros(transcription.n_unknowns)
        res = transcription.nonlinear_residual(xi)
        np.testing.assert_allclose(res["dlvx"], 0.0, atol=1e-12)
        np.testing.assert_allclose(res["dlvy"], 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Sequential linearization outer loop
# ---------------------------------------------------------------------------


@pytest.fixture
def gentle_transcription() -> IndirectTfcElmTranscription:
    """A short, gentle transfer near the Earth-Moon barycenter.

    The nominal moves through a region with moderate Omega curvature,
    so the sequential linearization outer loop converges without
    needing an adaptive line search.
    """
    cfg = IndirectTfcElmConfig(n_training=20, n_basis=80, seed=7)
    return IndirectTfcElmTranscription(
        dynamics=PlanarCR3BP(),
        r0=np.array([-0.1, 0.2]),
        v0=np.array([0.0, 0.0]),
        rf=np.array([0.1, 0.25]),
        vf=np.array([0.0, 0.0]),
        time_of_flight=0.5,
        config=cfg,
    )


class TestSequentialSolve:
    def test_returns_sequential_solution(self, gentle_transcription) -> None:
        sol = gentle_transcription.solve_sequential()
        assert isinstance(sol, SequentialSolution)
        assert sol.xi.shape == (gentle_transcription.n_unknowns,)

    def test_converges_on_gentle_problem(self, gentle_transcription) -> None:
        sol = gentle_transcription.solve_sequential(
            OuterLoopConfig(max_iter=40, tol=1e-7, damping=0.5)
        )
        assert sol.converged is True
        assert sol.n_iterations < 40
        assert sol.history["trajectory_change_inf"][-1] < 1e-7

    def test_converges_undamped_on_very_gentle_problem(
        self, gentle_transcription
    ) -> None:
        """With alpha = 1 (no damping) the gentle problem still converges."""
        sol = gentle_transcription.solve_sequential(
            OuterLoopConfig(max_iter=40, tol=1e-7, damping=1.0)
        )
        assert sol.converged is True
        # Undamped should be faster than damped when convergence holds.
        assert sol.n_iterations <= 20

    def test_nonlinear_residual_shrinks(self, gentle_transcription) -> None:
        """The full nonlinear residual should be small at the final iterate."""
        sol = gentle_transcription.solve_sequential(
            OuterLoopConfig(max_iter=40, tol=1e-7, damping=0.5)
        )
        final_nl = sol.history["nonlinear_residual"][-1]
        first_nl = sol.history["nonlinear_residual"][0]
        assert final_nl < 1e-3
        # And it should actually decrease across the loop.
        assert final_nl < first_nl

    def test_history_lengths_match_iterations(self, gentle_transcription) -> None:
        sol = gentle_transcription.solve_sequential(
            OuterLoopConfig(max_iter=40, tol=1e-7, damping=0.5)
        )
        for key in ("trajectory_change_inf", "linearized_residual", "nonlinear_residual"):
            assert len(sol.history[key]) == sol.n_iterations

    def test_trajectory_hits_bcs(self, gentle_transcription) -> None:
        sol = gentle_transcription.solve_sequential(
            OuterLoopConfig(max_iter=40, tol=1e-7, damping=0.5)
        )
        traj = sol.trajectory
        assert traj["x"][0] == pytest.approx(gentle_transcription.r0[0], abs=1e-10)
        assert traj["x"][-1] == pytest.approx(gentle_transcription.rf[0], abs=1e-10)
        assert traj["y"][0] == pytest.approx(gentle_transcription.r0[1], abs=1e-10)
        assert traj["y"][-1] == pytest.approx(gentle_transcription.rf[1], abs=1e-10)
        assert traj["vx"][0] == pytest.approx(gentle_transcription.v0[0], abs=1e-10)
        assert traj["vy"][-1] == pytest.approx(gentle_transcription.vf[1], abs=1e-10)

    def test_diverging_loop_returns_not_converged(self, transcription) -> None:
        """The aggressive fixture diverges under pure fixed-damping Picard;
        confirm the loop reports that cleanly instead of crashing or claiming
        success. ``line_search=False`` disables the Pass-2 backtracking so the
        iteration really does diverge (Pass 2 partially tames this problem).
        """
        sol = transcription.solve_sequential(
            OuterLoopConfig(
                max_iter=5, tol=1e-9, damping=1.0, line_search=False
            )
        )
        assert sol.converged is False
        assert sol.n_iterations == 5
        assert len(sol.history["trajectory_change_inf"]) == 5

    def test_accepts_explicit_initial_nominal(self, gentle_transcription) -> None:
        """Passing a custom initial nominal should run and still converge."""
        n = gentle_transcription.n_training
        # Slightly perturb the BC-only Hermite seed.
        x0, y0 = gentle_transcription.initial_nominal()
        rng = np.random.default_rng(3)
        seed = (x0 + 0.02 * rng.standard_normal(n), y0 + 0.02 * rng.standard_normal(n))
        sol = gentle_transcription.solve_sequential(
            OuterLoopConfig(max_iter=40, tol=1e-7, damping=0.5),
            initial_nominal=seed,
        )
        assert sol.converged is True

    def test_rejects_bad_damping(self, gentle_transcription) -> None:
        with pytest.raises(ValueError, match="damping"):
            gentle_transcription.solve_sequential(
                OuterLoopConfig(damping=0.0)
            )
        with pytest.raises(ValueError, match="damping"):
            gentle_transcription.solve_sequential(
                OuterLoopConfig(damping=1.5)
            )

    def test_rejects_bad_max_iter(self, gentle_transcription) -> None:
        with pytest.raises(ValueError, match="max_iter"):
            gentle_transcription.solve_sequential(
                OuterLoopConfig(max_iter=0)
            )

    def test_rejects_bad_tol(self, gentle_transcription) -> None:
        with pytest.raises(ValueError, match="tol"):
            gentle_transcription.solve_sequential(
                OuterLoopConfig(tol=0.0)
            )

    def test_rejects_wrong_initial_nominal_shape(self, gentle_transcription) -> None:
        n = gentle_transcription.n_training
        with pytest.raises(ValueError, match="initial_nominal"):
            gentle_transcription.solve_sequential(
                initial_nominal=(np.zeros(n - 1), np.zeros(n)),
            )


# ---------------------------------------------------------------------------
# Pass 2: backtracking line search on the nominal update
# ---------------------------------------------------------------------------


class TestLineSearch:
    """Tests for the Armijo-style backtracking line search on ``alpha``.

    The merit function is the fixed-point residual
    ``||traj(xi(n_trial)) - n_trial||_inf`` so a successful line search
    strictly decreases ``history["trajectory_change_inf"]`` between
    iterations. On the gentle fixture ``alpha = 1`` (full Newton-like
    step) is always accepted; on the aggressive fixture the line
    search backtracks to smaller steps and reaches a much lower final
    fixed-point residual than the Pass-1 fixed-damping loop.
    """

    def test_default_config_has_line_search_enabled(self) -> None:
        assert OuterLoopConfig().line_search is True

    def test_default_damping_is_full_step(self) -> None:
        """Default ``damping`` is 1.0 so the line search starts at a full Newton step."""
        assert OuterLoopConfig().damping == 1.0

    def test_history_exposes_step_size(self, gentle_transcription) -> None:
        sol = gentle_transcription.solve_sequential(
            OuterLoopConfig(max_iter=40, tol=1e-7)
        )
        assert "step_size" in sol.history
        assert len(sol.history["step_size"]) == sol.n_iterations

    def test_gentle_problem_takes_full_steps(self, gentle_transcription) -> None:
        """On the gentle fixture every non-terminal step should be alpha = 1."""
        sol = gentle_transcription.solve_sequential(
            OuterLoopConfig(max_iter=40, tol=1e-7, damping=1.0, line_search=True)
        )
        assert sol.converged is True
        # All iterations except the final (converged-break) one.
        stepped_alphas = sol.history["step_size"][:-1]
        assert len(stepped_alphas) >= 1
        assert all(alpha == pytest.approx(1.0) for alpha in stepped_alphas)

    def test_step_size_stays_within_bounds(self, gentle_transcription) -> None:
        """Every recorded alpha is either 0 (converged break) or in [min_damping, damping]."""
        cfg = OuterLoopConfig(
            max_iter=40, tol=1e-7, damping=0.8, min_damping=1e-4
        )
        sol = gentle_transcription.solve_sequential(cfg)
        for alpha in sol.history["step_size"]:
            assert alpha == 0.0 or (cfg.min_damping - 1e-12 <= alpha <= cfg.damping + 1e-12)

    def test_step_norm_is_monotone_under_line_search(
        self, gentle_transcription
    ) -> None:
        """Armijo merit is the fixed-point residual, so it must not grow.

        On the gentle fixture the iteration converges quadratically and
        the merit strictly decreases at every step (relative slack of
        1e-10 absorbs floating-point wiggles at the convergence
        boundary).
        """
        sol = gentle_transcription.solve_sequential(
            OuterLoopConfig(max_iter=40, tol=1e-9, damping=1.0, line_search=True)
        )
        hist = sol.history["trajectory_change_inf"]
        assert len(hist) >= 3
        for i in range(1, len(hist)):
            slack = 1e-10 * max(1.0, hist[i - 1])
            assert hist[i] <= hist[i - 1] + slack, (
                f"step_norm grew from {hist[i-1]:.3e} to {hist[i]:.3e} at iter {i}"
            )

    def test_line_search_bounds_step_norm_on_aggressive_problem(
        self, transcription
    ) -> None:
        """The Armijo guarantee: step_norm never grows past its initial value.

        Pure fixed-damping Picard on the aggressive fixture oscillates
        wildly (peak step_norm ~2 compared to an initial value ~0.7),
        while the line search version keeps the fixed-point residual
        bounded by its starting value for all iterations. This is the
        concrete stability benefit of Pass 2 over Pass 1.
        """
        budget_args = dict(max_iter=30, tol=1e-9, damping=1.0)
        sol_fixed = transcription.solve_sequential(
            OuterLoopConfig(**budget_args, line_search=False)
        )
        sol_ls = transcription.solve_sequential(
            OuterLoopConfig(**budget_args, line_search=True)
        )
        hist_fixed = sol_fixed.history["trajectory_change_inf"]
        hist_ls = sol_ls.history["trajectory_change_inf"]

        # Line search must keep step_norm bounded by its initial value.
        initial_ls = hist_ls[0]
        slack = 1e-10 * max(1.0, initial_ls)
        assert all(h <= initial_ls + slack for h in hist_ls), (
            f"line search let step_norm grow beyond initial {initial_ls:.3e}"
        )

        # Fixed-damping, in contrast, oscillates well above its
        # starting value on this fixture — that's the divergence Pass
        # 2 is meant to suppress.
        assert max(hist_fixed) > 2.0 * hist_fixed[0]
        # And the worst step under line search is comfortably below
        # the worst step under fixed damping.
        assert max(hist_ls) < 0.5 * max(hist_fixed)

    def test_line_search_off_matches_pass1_behavior(
        self, gentle_transcription
    ) -> None:
        """With ``line_search=False`` the loop should take fixed ``damping`` steps."""
        cfg = OuterLoopConfig(
            max_iter=40, tol=1e-7, damping=0.5, line_search=False
        )
        sol = gentle_transcription.solve_sequential(cfg)
        assert sol.converged is True
        # All non-convergence iterations must record exactly 0.5.
        stepped = sol.history["step_size"][:-1]
        assert all(alpha == pytest.approx(0.5) for alpha in stepped)

    def test_rejects_bad_min_damping(self, gentle_transcription) -> None:
        with pytest.raises(ValueError, match="min_damping"):
            gentle_transcription.solve_sequential(
                OuterLoopConfig(damping=0.5, min_damping=0.0)
            )
        with pytest.raises(ValueError, match="min_damping"):
            gentle_transcription.solve_sequential(
                # min_damping must be <= damping
                OuterLoopConfig(damping=0.2, min_damping=0.5)
            )

    def test_rejects_bad_backtrack_factor(self, gentle_transcription) -> None:
        with pytest.raises(ValueError, match="backtrack_factor"):
            gentle_transcription.solve_sequential(
                OuterLoopConfig(backtrack_factor=0.0)
            )
        with pytest.raises(ValueError, match="backtrack_factor"):
            gentle_transcription.solve_sequential(
                OuterLoopConfig(backtrack_factor=1.0)
            )

    def test_rejects_bad_armijo_c(self, gentle_transcription) -> None:
        with pytest.raises(ValueError, match="armijo_c"):
            gentle_transcription.solve_sequential(
                OuterLoopConfig(armijo_c=-1e-4)
            )
        with pytest.raises(ValueError, match="armijo_c"):
            gentle_transcription.solve_sequential(
                OuterLoopConfig(armijo_c=1.0)
            )

    def test_line_search_validation_skipped_when_disabled(
        self, gentle_transcription
    ) -> None:
        """Bad line-search fields must not error when ``line_search=False``."""
        cfg = OuterLoopConfig(
            max_iter=5,
            tol=1e-4,
            damping=0.5,
            line_search=False,
            min_damping=0.0,       # would be rejected if line search were on
            backtrack_factor=2.0,  # same
            armijo_c=-1.0,         # same
        )
        sol = gentle_transcription.solve_sequential(cfg)
        assert isinstance(sol, SequentialSolution)
