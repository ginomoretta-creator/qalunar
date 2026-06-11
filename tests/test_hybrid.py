"""Tests for the hybrid QUBO sequential solver with nonlinear reranking."""

from __future__ import annotations

import numpy as np
import pytest

from qalunar.dynamics import PlanarCR3BP
from qalunar.hybrid import (
    HybridConfig,
    HybridSolution,
    rerank_samples,
    solve_hybrid,
)
from qalunar.qubo import LinearLsqToQuboConfig, build_linear_lsq_qubo
from qalunar.transcription import (
    IndirectTfcElmConfig,
    IndirectTfcElmTranscription,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_transcription() -> IndirectTfcElmTranscription:
    """Small problem for fast tests."""
    bcs = dict(
        r0=np.array([-0.3, 0.0]),
        v0=np.array([0.0, 0.6]),
        rf=np.array([0.4, 0.2]),
        vf=np.array([-0.1, 0.0]),
        time_of_flight=2.5,
    )
    cfg = IndirectTfcElmConfig(n_training=8, n_basis=20, seed=7)
    return IndirectTfcElmTranscription(dynamics=PlanarCR3BP(), config=cfg, **bcs)


# ---------------------------------------------------------------------------
# Reranking unit tests
# ---------------------------------------------------------------------------


class TestReranking:
    def test_rerank_returns_valid_rank(self, small_transcription) -> None:
        """Reranking should return a rank between 1 and k."""
        trans = small_transcription
        nominal_x, nominal_y = trans.initial_nominal()
        A, B = trans.build_linear_system(nominal_x, nominal_y)

        qubo_cfg = LinearLsqToQuboConfig(bits_per_variable=4)
        qubo = build_linear_lsq_qubo(A, B, qubo_cfg)

        # Generate some random bitstrings (pretending they're top-k)
        rng = np.random.default_rng(0)
        k = 10
        fake_top_k = rng.integers(0, 2, size=(k, qubo.n_bits)).astype(np.int64)
        # Put warm start as first row (ground state)
        fake_top_k[0] = qubo.warm_start_bits()

        rank, xi, best_nl, ground_nl = rerank_samples(
            fake_top_k, qubo, trans
        )

        assert 1 <= rank <= k
        assert xi.shape == (trans.n_unknowns,)
        assert np.isfinite(best_nl)
        assert np.isfinite(ground_nl)
        assert best_nl <= ground_nl  # reranked should be <= ground state

    def test_single_sample_gives_rank_1(self, small_transcription) -> None:
        """With one sample, reranking must return rank 1."""
        trans = small_transcription
        nominal_x, nominal_y = trans.initial_nominal()
        A, B = trans.build_linear_system(nominal_x, nominal_y)

        qubo_cfg = LinearLsqToQuboConfig(bits_per_variable=4)
        qubo = build_linear_lsq_qubo(A, B, qubo_cfg)

        single = qubo.warm_start_bits().reshape(1, -1)
        rank, _, _, _ = rerank_samples(single, qubo, trans)
        assert rank == 1


# ---------------------------------------------------------------------------
# Hybrid solver
# ---------------------------------------------------------------------------


class TestSolveHybrid:
    def test_returns_hybrid_solution(self, small_transcription) -> None:
        cfg = HybridConfig(
            qubo=LinearLsqToQuboConfig(bits_per_variable=4),
            num_reads=50,
            rerank_top_k=10,
            max_iter=3,
            seed=0,
        )
        sol = solve_hybrid(small_transcription, config=cfg)
        assert isinstance(sol, HybridSolution)

    def test_history_keys_present(self, small_transcription) -> None:
        cfg = HybridConfig(
            qubo=LinearLsqToQuboConfig(bits_per_variable=4),
            num_reads=50,
            rerank_top_k=10,
            max_iter=3,
            seed=0,
        )
        sol = solve_hybrid(small_transcription, config=cfg)

        expected_keys = {
            "trajectory_change_inf",
            "nonlinear_residual",
            "linearized_residual",
            "step_size",
            "ground_state_nl_res",
            "reranked_nl_res",
            "best_qubo_rank",
            "n_unique_samples",
        }
        assert set(sol.history.keys()) == expected_keys

    def test_history_lengths_consistent(self, small_transcription) -> None:
        cfg = HybridConfig(
            qubo=LinearLsqToQuboConfig(bits_per_variable=4),
            num_reads=50,
            rerank_top_k=10,
            max_iter=5,
            seed=0,
        )
        sol = solve_hybrid(small_transcription, config=cfg)

        n = sol.n_iterations
        for key, vals in sol.history.items():
            assert len(vals) == n, f"history[{key!r}] has {len(vals)} entries, expected {n}"

    def test_reranked_nl_leq_ground_nl(self, small_transcription) -> None:
        """Reranked NL residual should never exceed ground-state NL residual."""
        cfg = HybridConfig(
            qubo=LinearLsqToQuboConfig(bits_per_variable=4),
            num_reads=100,
            rerank_top_k=20,
            max_iter=5,
            seed=0,
        )
        sol = solve_hybrid(small_transcription, config=cfg)

        for it in range(sol.n_iterations):
            assert sol.history["reranked_nl_res"][it] <= (
                sol.history["ground_state_nl_res"][it] + 1e-12
            ), f"iteration {it}: reranked > ground state"

    def test_no_rerank_uses_ground_state(self, small_transcription) -> None:
        """With rerank=False, best_qubo_rank should always be 1."""
        cfg = HybridConfig(
            qubo=LinearLsqToQuboConfig(bits_per_variable=4),
            num_reads=50,
            rerank_top_k=10,
            rerank=False,
            max_iter=3,
            seed=0,
        )
        sol = solve_hybrid(small_transcription, config=cfg)

        for rank in sol.history["best_qubo_rank"]:
            assert rank == 1.0

    def test_nonlinear_residual_bounded(self, small_transcription) -> None:
        """NL residual should stay finite and not blow up."""
        cfg = HybridConfig(
            qubo=LinearLsqToQuboConfig(bits_per_variable=6),
            num_reads=200,
            rerank_top_k=30,
            max_iter=10,
            seed=42,
        )
        sol = solve_hybrid(small_transcription, config=cfg)

        for nl in sol.history["nonlinear_residual"]:
            assert np.isfinite(nl)
            assert nl < 1e6  # shouldn't diverge

    def test_custom_initial_nominal(self, small_transcription) -> None:
        """Solver accepts a custom initial nominal."""
        trans = small_transcription
        n = trans.n_training
        # Start from slightly perturbed Hermite interpolant
        nx, ny = trans.initial_nominal()
        nx_pert = nx + 0.01
        ny_pert = ny + 0.01

        cfg = HybridConfig(
            qubo=LinearLsqToQuboConfig(bits_per_variable=4),
            num_reads=50,
            rerank_top_k=10,
            max_iter=3,
            seed=0,
        )
        sol = solve_hybrid(
            trans, config=cfg, initial_nominal=(nx_pert, ny_pert)
        )
        assert isinstance(sol, HybridSolution)
