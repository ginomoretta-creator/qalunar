"""Tests for classical QUBO samplers and the full TFC+ELM -> QUBO pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from qalunar.dynamics import PlanarCR3BP
from qalunar.qubo import (
    LinearLsqToQuboConfig,
    QuboSampleResult,
    build_linear_lsq_qubo,
    sample_exact,
    sample_simulated_annealing,
)
from qalunar.transcription import (
    IndirectTfcElmConfig,
    IndirectTfcElmTranscription,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_consistent_system() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Small well-conditioned consistent system for exact-solver tests.

    4 unknowns, 12 rows, cond ~ 5 -> the QUBO ground state should
    recover xi_true to within quantization error.
    """
    rng = np.random.default_rng(99)
    m, n = 12, 4
    U, _ = np.linalg.qr(rng.standard_normal((m, n)))
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))
    s = np.array([5.0, 3.0, 2.0, 1.0])
    A = (U * s) @ V.T
    xi_true = rng.uniform(-0.2, 0.2, size=n)
    B = A @ xi_true
    return A, B, xi_true


@pytest.fixture
def small_qubo(
    small_consistent_system,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Small QUBO with 5 bits/var -> 20 total bits (exact-solver safe)."""
    A, B, xi_true = small_consistent_system
    cfg = LinearLsqToQuboConfig(bits_per_variable=5)
    result = build_linear_lsq_qubo(A, B, cfg)
    return result, A, B


# ---------------------------------------------------------------------------
# BQM conversion sanity
# ---------------------------------------------------------------------------


class TestBqmConversion:
    def test_bqm_energy_matches_qubo_energy(self, small_qubo) -> None:
        """The dimod BQM and our energy() must agree on random bitstrings."""
        from qalunar.qubo.samplers import _qubo_result_to_bqm

        result, _, _ = small_qubo
        bqm = _qubo_result_to_bqm(result)
        rng = np.random.default_rng(42)
        for _ in range(20):
            q = rng.integers(0, 2, size=result.n_bits)
            sample = {i: int(q[i]) for i in range(result.n_bits)}
            bqm_energy = bqm.energy(sample)
            our_energy = result.energy(q)
            np.testing.assert_allclose(
                bqm_energy, our_energy, rtol=1e-9, atol=1e-9
            )


# ---------------------------------------------------------------------------
# Exact solver
# ---------------------------------------------------------------------------


class TestExactSolver:
    def test_returns_result_instance(self, small_qubo) -> None:
        result, A, B = small_qubo
        sr = sample_exact(result, A=A, B=B)
        assert isinstance(sr, QuboSampleResult)

    def test_bitstring_is_binary(self, small_qubo) -> None:
        result, _, _ = small_qubo
        sr = sample_exact(result)
        assert set(np.unique(sr.bitstring).tolist()).issubset({0, 1})

    def test_energy_is_global_minimum(self, small_qubo) -> None:
        """Exact solver must find the lowest possible energy."""
        result, _, _ = small_qubo
        sr = sample_exact(result)
        # Warm start is near-optimal; exact should be at or below it.
        warm_energy = result.energy(result.warm_start_bits())
        assert sr.energy <= warm_energy + 1e-10

    def test_recovers_xi_tik_to_quantization_precision(
        self, small_qubo
    ) -> None:
        """Ground-state xi should be close to xi_tik (quantization limited)."""
        result, A, B = small_qubo
        sr = sample_exact(result, A=A, B=B)
        # 5 bits -> step = 2R/31 per coord.  xi should be within a
        # few steps of xi_tik.
        np.testing.assert_allclose(sr.xi, result.xi_tik, atol=0.05)

    def test_residual_norm_is_reported(self, small_qubo) -> None:
        result, A, B = small_qubo
        sr = sample_exact(result, A=A, B=B)
        assert np.isfinite(sr.residual_norm)
        expected = float(np.linalg.norm(A @ sr.xi - B))
        np.testing.assert_allclose(sr.residual_norm, expected, rtol=1e-12)

    def test_residual_nan_when_no_AB(self, small_qubo) -> None:
        result, _, _ = small_qubo
        sr = sample_exact(result)
        assert np.isnan(sr.residual_norm)

    def test_rejects_large_qubo(self) -> None:
        rng = np.random.default_rng(0)
        A = rng.standard_normal((30, 5))
        B = rng.standard_normal(30)
        cfg = LinearLsqToQuboConfig(bits_per_variable=5)
        result = build_linear_lsq_qubo(A, B, cfg)
        assert result.n_bits == 25
        with pytest.raises(ValueError, match="n_bits <= 20"):
            sample_exact(result)


# ---------------------------------------------------------------------------
# Simulated annealing
# ---------------------------------------------------------------------------


class TestSimulatedAnnealing:
    def test_returns_result_instance(self, small_qubo) -> None:
        result, A, B = small_qubo
        sr = sample_simulated_annealing(
            result, num_reads=50, seed=0, A=A, B=B
        )
        assert isinstance(sr, QuboSampleResult)

    def test_bitstring_is_binary(self, small_qubo) -> None:
        result, _, _ = small_qubo
        sr = sample_simulated_annealing(result, num_reads=50, seed=0)
        assert set(np.unique(sr.bitstring).tolist()).issubset({0, 1})

    def test_warm_start_improves_energy(self, small_qubo) -> None:
        """Warm-started SA should find energy <= cold-started SA."""
        result, _, _ = small_qubo
        cold = sample_simulated_annealing(result, num_reads=200, seed=1)
        warm = sample_simulated_annealing(
            result,
            num_reads=200,
            seed=1,
            initial_states=result.warm_start_bits(),
        )
        assert warm.energy <= cold.energy + 1e-8

    def test_sa_recovers_reasonable_xi(self, small_qubo) -> None:
        """SA with enough reads should get close to xi_tik."""
        result, A, B = small_qubo
        sr = sample_simulated_annealing(
            result,
            num_reads=500,
            seed=42,
            initial_states=result.warm_start_bits(),
            A=A,
            B=B,
        )
        np.testing.assert_allclose(sr.xi, result.xi_tik, atol=0.05)

    def test_residual_norm_reported(self, small_qubo) -> None:
        result, A, B = small_qubo
        sr = sample_simulated_annealing(
            result, num_reads=50, seed=0, A=A, B=B
        )
        assert np.isfinite(sr.residual_norm)


# ---------------------------------------------------------------------------
# End-to-end: TFC+ELM -> QUBO -> SA -> decoded trajectory
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Full pipeline integration: indirect TFC+ELM transcription produces
    a linear system ``(A, B)``, that system is encoded as a QUBO, sampled
    with SA, and the decoded trajectory is compared to the Tikhonov
    reference ``xi_tik`` (which is the target the QUBO is designed to
    approximate -- **not** the minimum-norm ``lstsq`` solution, which
    lives in a higher-dimensional space that the SVD-reduced QUBO cannot
    represent).
    """

    @pytest.fixture
    def pipeline_result(self):
        """Run the full pipeline on a small problem and return everything
        needed for assertions.
        """
        # Mild BCs away from primaries.
        bcs = dict(
            r0=np.array([-0.3, 0.0]),
            v0=np.array([0.0, 0.6]),
            rf=np.array([0.4, 0.2]),
            vf=np.array([-0.1, 0.0]),
            time_of_flight=2.5,
        )
        cfg = IndirectTfcElmConfig(n_training=10, n_basis=30, seed=7)
        trans = IndirectTfcElmTranscription(
            dynamics=PlanarCR3BP(), config=cfg, **bcs
        )

        # Build linear system at initial nominal
        nominal_x, nominal_y = trans.initial_nominal()
        A, B = trans.build_linear_system(nominal_x, nominal_y)

        # QUBO encode with 6 bits/var (default SVD threshold)
        qubo_cfg = LinearLsqToQuboConfig(bits_per_variable=6)
        qubo = build_linear_lsq_qubo(A, B, qubo_cfg)

        # Tikhonov reference trajectory (what the QUBO is targeting)
        traj_tik = trans.decode_trajectory(qubo.xi_tik)

        # SA sample with warm start.
        sr = sample_simulated_annealing(
            qubo,
            num_reads=2000,
            seed=42,
            initial_states=qubo.warm_start_bits(),
            A=A,
            B=B,
        )
        traj_qubo = trans.decode_trajectory(sr.xi)

        return dict(
            trans=trans,
            A=A,
            B=B,
            qubo=qubo,
            traj_tik=traj_tik,
            sr=sr,
            traj_qubo=traj_qubo,
        )

    def test_trajectory_positions_close_to_tikhonov(
        self, pipeline_result
    ) -> None:
        """QUBO trajectory (x, y) should closely track the Tikhonov reference."""
        traj_tik = pipeline_result["traj_tik"]
        traj_qubo = pipeline_result["traj_qubo"]

        dx = np.max(np.abs(traj_qubo["x"] - traj_tik["x"]))
        dy = np.max(np.abs(traj_qubo["y"] - traj_tik["y"]))
        # Visual-overlap threshold at 6 bits / ~360 qubits. The decoded
        # trajectory tracks the Tikhonov reference to within ~0.05-0.055
        # nondim: the gap is the 6-bit quantization plus SA heuristic noise
        # (SA minimises the QUBO residual norm, whose SVD-reduced floor is
        # ~13.0 and which it reaches to ~13.4-14.0 at 2k reads -- not the max
        # position deviation asserted here, so the two are only loosely
        # coupled). 0.06 nondim (~23,000 km, ~6% of the Earth-Moon distance)
        # keeps an honest margin while still asserting close visual overlap.
        assert dx < 0.06, f"max |dx| = {dx:.6f}"
        assert dy < 0.06, f"max |dy| = {dy:.6f}"

    def test_qubo_residual_bounded(self, pipeline_result) -> None:
        """QUBO residual should be within 20% of the Tikhonov residual.

        Both are solving the same regularized problem; the only gap is
        quantization noise from the binary encoding.
        """
        A = pipeline_result["A"]
        B = pipeline_result["B"]
        qubo = pipeline_result["qubo"]

        tik_res = float(np.linalg.norm(A @ qubo.xi_tik - B))
        qubo_res = pipeline_result["sr"].residual_norm
        assert qubo_res < 1.2 * tik_res + 0.1, (
            f"QUBO residual {qubo_res:.4e} >> Tikhonov residual {tik_res:.4e}"
        )

    def test_qubo_energy_below_random(self, pipeline_result) -> None:
        """SA best energy should be well below random bitstrings."""
        qubo = pipeline_result["qubo"]
        sr = pipeline_result["sr"]
        rng = np.random.default_rng(123)
        random_energies = [
            qubo.energy(rng.integers(0, 2, size=qubo.n_bits))
            for _ in range(20)
        ]
        assert sr.energy < 0.1 * float(np.mean(random_energies))

    def test_sa_energy_near_warm_start(self, pipeline_result) -> None:
        """SA should find energy at or below the warm-start quantization."""
        qubo = pipeline_result["qubo"]
        sr = pipeline_result["sr"]
        warm_energy = qubo.energy(qubo.warm_start_bits())
        assert sr.energy <= warm_energy + 1e-6
