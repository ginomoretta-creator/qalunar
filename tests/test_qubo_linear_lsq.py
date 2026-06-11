"""Tests for the linear-LSQ-to-QUBO transcription."""

from __future__ import annotations

import numpy as np
import pytest

from qalunar.qubo import (
    LinearLsqToQuboConfig,
    LinearLsqToQuboResult,
    build_linear_lsq_qubo,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def well_conditioned_system() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Overdetermined, full-rank system with a known ground truth.

    Singular spectrum is linspace(5, 0.5, 10) so condition number is
    10 and Tikhonov regularization is negligible. ``B = A @ xi_true``
    makes the system consistent; ``xi_true`` lies well inside the
    default encoding range so the warm start can recover it to
    quantization precision.
    """
    rng = np.random.default_rng(0)
    m, n = 40, 10
    u_raw = rng.standard_normal((m, n))
    v_raw = rng.standard_normal((n, n))
    U, _ = np.linalg.qr(u_raw)        # (m, n) with orthonormal columns
    V, _ = np.linalg.qr(v_raw)        # (n, n) orthogonal
    s = np.linspace(5.0, 0.5, n)
    A = (U * s) @ V.T
    xi_true = rng.uniform(-0.3, 0.3, size=n)
    B = A @ xi_true
    return A, B, xi_true


# ---------------------------------------------------------------------------
# Construction / shapes
# ---------------------------------------------------------------------------


class TestBuild:
    def test_returns_result_instance(self, well_conditioned_system) -> None:
        A, B, _ = well_conditioned_system
        result = build_linear_lsq_qubo(A, B)
        assert isinstance(result, LinearLsqToQuboResult)

    def test_shapes(self, well_conditioned_system) -> None:
        A, B, _ = well_conditioned_system
        cfg = LinearLsqToQuboConfig(bits_per_variable=6)
        result = build_linear_lsq_qubo(A, B, cfg)

        assert result.n_variables == A.shape[1]
        assert 1 <= result.n_reduced <= A.shape[1]
        assert result.n_bits == result.n_reduced * 6
        assert result.Q.shape == (result.n_bits, result.n_bits)
        assert result.linear.shape == (result.n_bits,)
        assert result.V_r.shape == (A.shape[1], result.n_reduced)
        assert result.alpha_offset.shape == (result.n_reduced,)
        assert result.step_sizes.shape == (result.n_reduced,)
        assert result.bits_per_variable == 6

    def test_q_is_symmetric(self, well_conditioned_system) -> None:
        A, B, _ = well_conditioned_system
        result = build_linear_lsq_qubo(A, B)
        np.testing.assert_allclose(result.Q, result.Q.T, atol=1e-12)

    def test_full_rank_for_well_conditioned(self, well_conditioned_system) -> None:
        A, B, _ = well_conditioned_system
        result = build_linear_lsq_qubo(A, B)
        assert result.effective_rank == A.shape[1]

    def test_default_config_used_when_none(self, well_conditioned_system) -> None:
        A, B, _ = well_conditioned_system
        result = build_linear_lsq_qubo(A, B, config=None)
        assert result.config.bits_per_variable == 8


class TestRankReduction:
    def test_rank_deficient_matrix_is_truncated(self) -> None:
        """A matrix with an exact duplicated column should drop in rank."""
        rng = np.random.default_rng(1)
        base = rng.standard_normal((30, 3))
        A = np.hstack([base, base[:, [0]]])  # col 3 == col 0 -> rank 3
        xi_true = np.array([1.0, 2.0, -0.5, 0.0])
        B = A @ xi_true
        # Drop the Tikhonov floor below the default svd_threshold so the
        # rank truncation can actually drop the smallest singular value.
        cfg = LinearLsqToQuboConfig(
            tikhonov_relative=1e-8, svd_threshold=1e-3
        )
        result = build_linear_lsq_qubo(A, B, cfg)
        assert result.effective_rank == 3


# ---------------------------------------------------------------------------
# Tikhonov reference
# ---------------------------------------------------------------------------


class TestTikhonov:
    def test_xi_tik_matches_closed_form(self, well_conditioned_system) -> None:
        """xi_tik should equal (A^T A + lambda I)^-1 A^T B to machine precision."""
        A, B, _ = well_conditioned_system
        result = build_linear_lsq_qubo(A, B)
        lam = result.lambda_tikhonov
        n = A.shape[1]
        closed_form = np.linalg.solve(A.T @ A + lam * np.eye(n), A.T @ B)
        np.testing.assert_allclose(result.xi_tik, closed_form, atol=1e-10, rtol=1e-10)

    def test_recovers_ground_truth_when_regularization_tiny(
        self, well_conditioned_system
    ) -> None:
        """Consistent full-rank system with tiny lambda => xi_tik ~ xi_true."""
        A, B, xi_true = well_conditioned_system
        result = build_linear_lsq_qubo(A, B)
        np.testing.assert_allclose(result.xi_tik, xi_true, atol=1e-4)

    def test_lambda_scales_with_sigma_max(self, well_conditioned_system) -> None:
        A, B, _ = well_conditioned_system
        cfg = LinearLsqToQuboConfig(tikhonov_relative=2.5e-7)
        result = build_linear_lsq_qubo(A, B, cfg)
        sigma_max = np.linalg.svd(A, compute_uv=False)[0]
        assert result.lambda_tikhonov == pytest.approx(2.5e-7 * sigma_max ** 2)


# ---------------------------------------------------------------------------
# Core identity: QUBO energy == augmented residual norm squared
# ---------------------------------------------------------------------------


class TestEnergyIdentity:
    """E(q) must equal ||A_aug @ xi(q) - B_aug||^2 by construction.

    This is the central correctness test — if it passes, the QUBO is
    a faithful encoding of the augmented least-squares objective.
    """

    def test_energy_equals_augmented_residual_random_bits(
        self, well_conditioned_system
    ) -> None:
        A, B, _ = well_conditioned_system
        result = build_linear_lsq_qubo(A, B)

        A_aug = np.vstack(
            [A, np.sqrt(result.lambda_tikhonov) * np.eye(A.shape[1])]
        )
        B_aug = np.concatenate([B, np.zeros(A.shape[1])])

        rng = np.random.default_rng(2024)
        for _ in range(20):
            q = rng.integers(0, 2, size=result.n_bits)
            xi = result.decode_xi(q)
            augmented_res_sq = float(np.sum((A_aug @ xi - B_aug) ** 2))
            np.testing.assert_allclose(
                result.energy(q), augmented_res_sq, rtol=1e-9, atol=1e-9
            )

    def test_energy_identity_on_warm_start(self, well_conditioned_system) -> None:
        A, B, _ = well_conditioned_system
        result = build_linear_lsq_qubo(A, B)
        A_aug = np.vstack(
            [A, np.sqrt(result.lambda_tikhonov) * np.eye(A.shape[1])]
        )
        B_aug = np.concatenate([B, np.zeros(A.shape[1])])
        q = result.warm_start_bits()
        xi = result.decode_xi(q)
        expected = float(np.sum((A_aug @ xi - B_aug) ** 2))
        np.testing.assert_allclose(result.energy(q), expected, rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


class TestDecoding:
    def test_zero_bits_gives_alpha_offset(self, well_conditioned_system) -> None:
        A, B, _ = well_conditioned_system
        result = build_linear_lsq_qubo(A, B)
        zero_bits = np.zeros(result.n_bits, dtype=int)
        np.testing.assert_allclose(
            result.decode_alpha(zero_bits), result.alpha_offset
        )

    def test_all_ones_gives_upper_corner(self, well_conditioned_system) -> None:
        A, B, _ = well_conditioned_system
        result = build_linear_lsq_qubo(A, B)
        ones = np.ones(result.n_bits, dtype=int)
        max_int = (1 << result.bits_per_variable) - 1
        expected = result.alpha_offset + result.step_sizes * max_int
        np.testing.assert_allclose(result.decode_alpha(ones), expected)

    def test_decoded_xi_is_V_r_times_alpha(self, well_conditioned_system) -> None:
        A, B, _ = well_conditioned_system
        result = build_linear_lsq_qubo(A, B)
        rng = np.random.default_rng(5)
        q = rng.integers(0, 2, size=result.n_bits)
        np.testing.assert_allclose(
            result.decode_xi(q), result.V_r @ result.decode_alpha(q)
        )

    def test_wrong_bitstring_length_raises(self, well_conditioned_system) -> None:
        A, B, _ = well_conditioned_system
        result = build_linear_lsq_qubo(A, B)
        with pytest.raises(ValueError, match="length"):
            result.decode_alpha(np.zeros(result.n_bits + 1, dtype=int))

    def test_energy_wrong_length_raises(self, well_conditioned_system) -> None:
        A, B, _ = well_conditioned_system
        result = build_linear_lsq_qubo(A, B)
        with pytest.raises(ValueError, match="length"):
            result.energy(np.zeros(result.n_bits - 1, dtype=int))


# ---------------------------------------------------------------------------
# Warm start
# ---------------------------------------------------------------------------


class TestWarmStart:
    def test_warm_start_within_half_step_of_alpha_tik(
        self, well_conditioned_system
    ) -> None:
        """Decoded warm-start alpha is within half a step of alpha_tik per coord."""
        A, B, _ = well_conditioned_system
        cfg = LinearLsqToQuboConfig(bits_per_variable=10)
        result = build_linear_lsq_qubo(A, B, cfg)
        q_warm = result.warm_start_bits()
        alpha_warm = result.decode_alpha(q_warm)
        err = np.abs(alpha_warm - result.alpha_tik)
        assert np.all(err <= 0.5 * result.step_sizes + 1e-12)

    def test_warm_start_beats_random_by_wide_margin(
        self, well_conditioned_system
    ) -> None:
        """Warm start energy should be orders of magnitude below random bits."""
        A, B, _ = well_conditioned_system
        result = build_linear_lsq_qubo(A, B)
        E_warm = result.energy(result.warm_start_bits())

        rng = np.random.default_rng(42)
        random_energies = [
            result.energy(rng.integers(0, 2, size=result.n_bits))
            for _ in range(20)
        ]
        assert E_warm < 0.01 * float(np.mean(random_energies))

    def test_warm_start_residual_small_for_consistent_system(
        self, well_conditioned_system
    ) -> None:
        """A 12-bit warm start on a well-conditioned consistent system is tight.

        The consistent RHS has ||B|| ~ O(1), and a 12-bit warm start on
        a cond(A) ~ 10 system should put the original-system residual
        well below 1% of that — 5e-3 is comfortably above the observed
        ~1.2e-3 on the fixture but still a strong correctness claim.
        """
        A, B, _ = well_conditioned_system
        cfg = LinearLsqToQuboConfig(bits_per_variable=12)
        result = build_linear_lsq_qubo(A, B, cfg)
        xi_warm = result.decode_xi(result.warm_start_bits())
        assert np.linalg.norm(A @ xi_warm - B) < 5e-3

    def test_warm_start_bits_are_binary(self, well_conditioned_system) -> None:
        A, B, _ = well_conditioned_system
        result = build_linear_lsq_qubo(A, B)
        bits = result.warm_start_bits()
        assert bits.shape == (result.n_bits,)
        assert set(np.unique(bits).tolist()).issubset({0, 1})


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_requires_2d_A(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            build_linear_lsq_qubo(np.zeros(3), np.zeros(3))

    def test_requires_matching_B_length(self) -> None:
        A = np.eye(3)
        with pytest.raises(ValueError, match="length"):
            build_linear_lsq_qubo(A, np.zeros(4))

    def test_requires_positive_bits(self, well_conditioned_system) -> None:
        A, B, _ = well_conditioned_system
        with pytest.raises(ValueError, match="bits_per_variable"):
            build_linear_lsq_qubo(
                A, B, LinearLsqToQuboConfig(bits_per_variable=0)
            )

    def test_requires_nonzero_A(self) -> None:
        with pytest.raises(ValueError, match="singular"):
            build_linear_lsq_qubo(np.zeros((4, 3)), np.zeros(4))
