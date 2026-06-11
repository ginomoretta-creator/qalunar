"""Hybrid classical-quantum sequential solver with nonlinear reranking.

At each outer iteration of the sequential linearization the linear
system ``A @ xi = B`` is encoded as a QUBO and sampled (via simulated
annealing now, QPU later). Instead of using only the ground-state
sample, we decode the *top-k* lowest-energy bitstrings and evaluate
the *full nonlinear* TPBVP residual for each. The sample with the
lowest nonlinear residual becomes the candidate for the nominal update.

Why this helps
--------------

The QUBO energy equals the *linearized* least-squares residual, which
is a first-order approximation of the true dynamics residual. A sample
that is slightly worse in QUBO energy may decode to a trajectory that
is *better* in the nonlinear sense, because the linearization is stale.
Quantum annealers naturally produce diverse low-energy samples (via
quantum tunneling through energy barriers), making them effective
proposal generators for this reranking strategy.

The key diagnostic is **best_qubo_rank**: the position (by QUBO
energy) of the sample that won the nonlinear reranking. If this is
frequently > 1, it proves the strategy adds value beyond ground-state
extraction. A classical SA sampler can approximate this diversity, but
a QPU is expected to produce more structurally diverse samples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qalunar.qubo import (
    LinearLsqToQuboConfig,
    build_linear_lsq_qubo,
)
from qalunar.qubo.linear_lsq import LinearLsqToQuboResult
from qalunar.qubo.samplers import (
    QuboSampleResult,
    _best_sample_to_array,
    _qubo_result_to_bqm,
)
from qalunar.transcription.indirect_tfc_elm import IndirectTfcElmTranscription

import dimod
import neal


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------


@dataclass(frozen=True)
class HybridConfig:
    """Configuration for the hybrid QUBO sequential solver.

    Parameters
    ----------
    qubo : LinearLsqToQuboConfig
        QUBO encoding hyperparameters (bits, Tikhonov, SVD threshold).
    num_reads : int, default 1000
        SA / QPU samples per outer iteration.
    rerank_top_k : int, default 50
        How many lowest-energy samples to evaluate nonlinear residual
        for. Higher values explore more of the landscape but cost more
        nonlinear-residual evaluations (each is cheap: one matrix-vector
        product per equation).
    rerank : bool, default True
        When ``False``, always use the QUBO ground state (lowest-energy
        sample) without reranking. Useful as a control to isolate the
        effect of reranking.
    seed : int or None, default 42
        Base RNG seed. Each outer iteration uses ``seed + iteration``
        so samples vary across iterations.
    max_iter : int, default 50
        Maximum outer iterations.
    tol : float, default 1e-6
        Convergence tolerance on the undamped fixed-point step norm
        ``max(max|dx|, max|dy|)``.
    damping : float, default 0.5
        Fixed damping factor for the nominal update. Values < 1.0
        stabilize convergence at the cost of slower progress.
    """

    qubo: LinearLsqToQuboConfig = field(default_factory=LinearLsqToQuboConfig)
    num_reads: int = 1000
    rerank_top_k: int = 50
    rerank: bool = True
    seed: int | None = 42
    max_iter: int = 50
    tol: float = 1e-6
    damping: float = 0.5


# ------------------------------------------------------------------
# Result container
# ------------------------------------------------------------------


@dataclass
class HybridSolution:
    """Result of :func:`solve_hybrid`.

    Attributes
    ----------
    xi : (6 L,) ndarray
        Final coefficient vector.
    trajectory : dict of (n,) ndarrays
        Decoded state / costate / control at the training points.
    converged : bool
        Whether the fixed-point step norm dropped below ``tol``.
    n_iterations : int
        Number of outer iterations executed.
    history : dict of lists
        Per-iteration diagnostics:

        * ``trajectory_change_inf``: undamped step norm (convergence metric)
        * ``nonlinear_residual``: infinity-norm of the full nonlinear TPBVP
          residual at the decoded trajectory
        * ``linearized_residual``: ``||A xi - B||_2`` of the selected sample
        * ``step_size``: damping factor actually used
        * ``ground_state_nl_res``: nonlinear residual of the QUBO ground
          state (lowest-energy sample)
        * ``reranked_nl_res``: nonlinear residual of the reranking winner
        * ``best_qubo_rank``: 1-based rank (by QUBO energy) of the sample
          that won the reranking. ``1`` means ground state was best;
          ``> 1`` means an excited state was better in the nonlinear sense.
        * ``n_unique_samples``: number of distinct samples in the top-k pool
    """

    xi: NDArray[np.float64]
    trajectory: dict[str, NDArray[np.float64]]
    converged: bool
    n_iterations: int
    history: dict[str, list[float]]


# ------------------------------------------------------------------
# Core reranking logic
# ------------------------------------------------------------------


def _extract_top_k_bitstrings(
    sample_set: dimod.SampleSet,
    n_bits: int,
    k: int,
) -> NDArray[np.int64]:
    """Extract top-k lowest-energy bitstrings as a ``(k, n_bits)`` array.

    The record is sorted by energy explicitly — dimod does not guarantee
    record order, and a sampler that returns reads in submission order
    (e.g. a QPU) would otherwise silently break the reranking.
    """
    variables = list(sample_set.variables)
    var_to_col = {v: i for i, v in enumerate(variables)}
    col_order = [var_to_col[i] for i in range(n_bits)]

    record = sample_set.record
    order = np.argsort(record.energy, kind="stable")
    actual_k = min(k, len(record))
    samples = record.sample[order[:actual_k]][:, col_order]
    return samples.astype(np.int64)


def rerank_samples(
    top_k_bits: NDArray[np.int64],
    qubo_result: LinearLsqToQuboResult,
    transcription: IndirectTfcElmTranscription,
) -> tuple[int, NDArray[np.float64], float, float]:
    """Rerank bitstrings by full nonlinear TPBVP residual.

    Parameters
    ----------
    top_k_bits : (k, n_bits) ndarray
        Bitstrings to evaluate, ordered by QUBO energy (lowest first).
    qubo_result : LinearLsqToQuboResult
        For decoding bitstrings to xi.
    transcription : IndirectTfcElmTranscription
        For evaluating the nonlinear residual.

    Returns
    -------
    best_rank : int
        1-based rank (by QUBO energy) of the winner.
    best_xi : (6 L,) ndarray
        Coefficient vector of the winner.
    best_nl_res : float
        Nonlinear residual of the winner.
    ground_nl_res : float
        Nonlinear residual of the rank-1 (ground state) sample.
    """
    k = top_k_bits.shape[0]
    best_rank = 1
    best_xi: NDArray[np.float64] | None = None
    best_nl_res = float("inf")
    ground_nl_res = float("inf")

    for i in range(k):
        bits = top_k_bits[i]
        xi = qubo_result.decode_xi(bits)
        nl_dict = transcription.nonlinear_residual(xi)
        nl_res = float(max(np.max(np.abs(r)) for r in nl_dict.values()))

        if i == 0:
            ground_nl_res = nl_res

        if nl_res < best_nl_res:
            best_nl_res = nl_res
            best_xi = xi
            best_rank = i + 1

    assert best_xi is not None
    return best_rank, best_xi, best_nl_res, ground_nl_res


# ------------------------------------------------------------------
# Hybrid sequential solver
# ------------------------------------------------------------------


def solve_hybrid(
    transcription: IndirectTfcElmTranscription,
    config: HybridConfig | None = None,
    initial_nominal: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
) -> HybridSolution:
    """Sequential linearization with QUBO sampling and nonlinear reranking.

    Drop-in replacement for
    :meth:`IndirectTfcElmTranscription.solve_sequential` that uses QUBO
    sampling instead of ``lstsq`` at each inner solve.

    Parameters
    ----------
    transcription : IndirectTfcElmTranscription
        Fully constructed transcription (BCs, basis, etc.).
    config : HybridConfig, optional
        Hybrid solver hyperparameters. Uses sensible defaults.
    initial_nominal : (x_bar, y_bar) tuple, optional
        Starting nominal. Defaults to the BC-only Hermite interpolant.

    Returns
    -------
    HybridSolution
        Final solution with reranking diagnostics.
    """
    cfg = config if config is not None else HybridConfig()
    n = transcription.n_training

    if initial_nominal is None:
        nominal_x, nominal_y = transcription.initial_nominal()
    else:
        nominal_x = np.asarray(initial_nominal[0], dtype=np.float64).copy()
        nominal_y = np.asarray(initial_nominal[1], dtype=np.float64).copy()

    history: dict[str, list[float]] = {
        "trajectory_change_inf": [],
        "nonlinear_residual": [],
        "linearized_residual": [],
        "step_size": [],
        "ground_state_nl_res": [],
        "reranked_nl_res": [],
        "best_qubo_rank": [],
        "n_unique_samples": [],
    }

    # ------------------------------------------------------------------
    # Initial QUBO solve at the starting nominal
    # ------------------------------------------------------------------
    xi, traj, lin_res, nl_res, rank, ground_nl, reranked_nl, n_unique, prev_bits = (
        _hybrid_solve_step(transcription, nominal_x, nominal_y, cfg, iteration=0)
    )

    converged = False
    for it in range(cfg.max_iter):
        dx = traj["x"] - nominal_x
        dy = traj["y"] - nominal_y
        step_norm = float(max(np.max(np.abs(dx)), np.max(np.abs(dy))))

        history["trajectory_change_inf"].append(step_norm)
        history["nonlinear_residual"].append(nl_res)
        history["linearized_residual"].append(lin_res)
        history["ground_state_nl_res"].append(ground_nl)
        history["reranked_nl_res"].append(reranked_nl)
        history["best_qubo_rank"].append(float(rank))
        history["n_unique_samples"].append(float(n_unique))

        if step_norm < cfg.tol:
            converged = True
            history["step_size"].append(0.0)
            break

        alpha = cfg.damping
        nominal_x = nominal_x + alpha * dx
        nominal_y = nominal_y + alpha * dy
        history["step_size"].append(alpha)

        xi, traj, lin_res, nl_res, rank, ground_nl, reranked_nl, n_unique, prev_bits = (
            _hybrid_solve_step(
                transcription, nominal_x, nominal_y, cfg, iteration=it + 1,
                prev_bits=prev_bits,
            )
        )

    return HybridSolution(
        xi=xi,
        trajectory=traj,
        converged=converged,
        n_iterations=len(history["trajectory_change_inf"]),
        history=history,
    )


def _hybrid_solve_step(
    transcription: IndirectTfcElmTranscription,
    nominal_x: NDArray[np.float64],
    nominal_y: NDArray[np.float64],
    cfg: HybridConfig,
    iteration: int,
    prev_bits: NDArray[np.int64] | None = None,
) -> tuple[
    NDArray[np.float64],  # xi
    dict[str, NDArray[np.float64]],  # trajectory
    float,  # linearized_residual
    float,  # nonlinear_residual (of selected sample)
    int,  # best_qubo_rank
    float,  # ground_state_nl_res
    float,  # reranked_nl_res
    int,  # n_unique_samples
    NDArray[np.int64],  # winner_bits (for carry-forward)
]:
    """One hybrid solve step: build QUBO, sample, rerank, return best."""

    A, B = transcription.build_linear_system(nominal_x, nominal_y)
    qubo = build_linear_lsq_qubo(A, B, cfg.qubo)

    # Sample
    bqm = _qubo_result_to_bqm(qubo)
    sampler = neal.SimulatedAnnealingSampler()
    sa_kwargs: dict[str, Any] = dict(num_reads=cfg.num_reads)
    if cfg.seed is not None:
        sa_kwargs["seed"] = cfg.seed + iteration

    # Warm start: use previous winner if available, else Tikhonov quantization
    warm = qubo.warm_start_bits()
    init_samples = [{i: int(warm[i]) for i in range(qubo.n_bits)}]
    if prev_bits is not None and len(prev_bits) == qubo.n_bits:
        init_samples.append({i: int(prev_bits[i]) for i in range(qubo.n_bits)})
    init_set = dimod.SampleSet.from_samples(
        init_samples, vartype=dimod.BINARY, energy=[0.0] * len(init_samples)
    )
    sa_kwargs["initial_states"] = init_set

    sample_set = sampler.sample(bqm, **sa_kwargs)

    # Extract top-k
    top_k_bits = _extract_top_k_bitstrings(
        sample_set, qubo.n_bits, cfg.rerank_top_k
    )
    n_unique = len(np.unique(top_k_bits, axis=0))

    if cfg.rerank and top_k_bits.shape[0] > 1:
        rank, xi, reranked_nl, ground_nl = rerank_samples(
            top_k_bits, qubo, transcription
        )
        # The winning bitstring is at position rank-1 in top_k_bits
        winner_bits = top_k_bits[rank - 1].copy()
    else:
        # Ground state only
        bits = _best_sample_to_array(sample_set, qubo.n_bits)
        xi = qubo.decode_xi(bits)
        nl_dict = transcription.nonlinear_residual(xi)
        nl_res_val = float(max(np.max(np.abs(r)) for r in nl_dict.values()))
        rank = 1
        ground_nl = nl_res_val
        reranked_nl = nl_res_val
        winner_bits = bits.copy()

    traj = transcription.decode_trajectory(xi)
    lin_res = float(np.linalg.norm(A @ xi - B))
    nl_dict = transcription.nonlinear_residual(xi)
    nl_res = float(max(np.max(np.abs(r)) for r in nl_dict.values()))

    return xi, traj, lin_res, nl_res, rank, ground_nl, reranked_nl, n_unique, winner_bits
