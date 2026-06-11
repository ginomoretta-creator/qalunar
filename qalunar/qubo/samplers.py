"""Classical QUBO samplers for validating the pipeline without D-Wave hardware.

Wraps Ocean SDK classical samplers -- simulated annealing via ``dwave-neal``
and brute-force exact via ``dimod.ExactSolver`` -- so they accept a
:class:`LinearLsqToQuboResult` and return decoded trajectory coefficients
ready for comparison against the LSQ reference.

These samplers provide the classical baseline.  When D-Wave access becomes
available the same QUBO is submitted to the QPU and the results are
compared against the SA / exact solutions produced here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import dimod
import neal
import numpy as np
from numpy.typing import NDArray

from qalunar.qubo.linear_lsq import LinearLsqToQuboResult


# ------------------------------------------------------------------
# Result container
# ------------------------------------------------------------------


@dataclass(frozen=True)
class QuboSampleResult:
    """Result of sampling a QUBO classically.

    Attributes
    ----------
    bitstring : (n_bits,) ndarray of int64
        Best sample (lowest energy) found by the sampler.
    xi : (n_vars,) ndarray
        Decoded original unknowns ``V_r @ alpha(bitstring)``.
    energy : float
        QUBO energy of the best sample.
    residual_norm : float
        ``||A @ xi - B||_2`` where ``(A, B)`` are the original
        (un-augmented) system matrices.  Only meaningful when the
        caller passes ``A`` and ``B`` via :func:`sample_simulated_annealing`
        or :func:`sample_exact`; otherwise ``nan``.
    num_reads : int
        Total number of samples drawn.
    sample_set : dimod.SampleSet
        Full Ocean SDK sample set for further analysis (energy
        histogram, excited states, timing, ...).
    """

    bitstring: NDArray[np.int64]
    xi: NDArray[np.float64]
    energy: float
    residual_norm: float
    num_reads: int
    sample_set: dimod.SampleSet


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _qubo_result_to_bqm(
    result: LinearLsqToQuboResult,
) -> dimod.BinaryQuadraticModel:
    """Convert our ``(Q, linear, const)`` triple to a dimod BQM.

    For binary variables ``q_i^2 = q_i``, so the quadratic form
    ``q^T Q q + linear^T q + const`` decomposes into:

    * linear biases  ``h_i = Q_ii + linear_i``
    * quadratic biases  ``J_{ij} = 2 Q_{ij}``  (``Q`` is symmetric)
    * offset  ``const``
    """
    n = result.n_bits
    Q = result.Q
    lin = result.linear

    h: dict[int, float] = {}
    J: dict[tuple[int, int], float] = {}

    for i in range(n):
        h[i] = float(Q[i, i] + lin[i])
        for j in range(i + 1, n):
            val = float(2.0 * Q[i, j])
            if abs(val) > 1e-15:
                J[(i, j)] = val

    return dimod.BinaryQuadraticModel(h, J, result.const, dimod.BINARY)


def _best_sample_to_array(
    sample_set: dimod.SampleSet, n_bits: int
) -> NDArray[np.int64]:
    """Extract the lowest-energy sample as a numpy bitstring array."""
    best = sample_set.first.sample
    bits = np.zeros(n_bits, dtype=np.int64)
    for i in range(n_bits):
        bits[i] = int(best[i])
    return bits


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def sample_simulated_annealing(
    qubo_result: LinearLsqToQuboResult,
    num_reads: int = 1000,
    seed: int | None = None,
    initial_states: NDArray[np.int64] | None = None,
    A: NDArray[np.float64] | None = None,
    B: NDArray[np.float64] | None = None,
    **sa_kwargs: Any,
) -> QuboSampleResult:
    """Sample a QUBO with simulated annealing (``dwave-neal``).

    Parameters
    ----------
    qubo_result : LinearLsqToQuboResult
        QUBO from :func:`build_linear_lsq_qubo`.
    num_reads : int, default 1000
        Number of independent annealing runs.
    seed : int, optional
        RNG seed for reproducibility.
    initial_states : (n_bits,) ndarray, optional
        Seeds every SA run from this bitstring (warm start).
        Typical usage: ``qubo_result.warm_start_bits()``.
    A, B : ndarray, optional
        Original (un-augmented) system matrices.  When provided the
        result includes ``residual_norm = ||A @ xi - B||_2``.
    **sa_kwargs
        Extra keyword arguments forwarded to
        ``neal.SimulatedAnnealingSampler.sample``.

    Returns
    -------
    QuboSampleResult
        Best sample found, decoded into the original unknowns.
    """
    bqm = _qubo_result_to_bqm(qubo_result)
    sampler = neal.SimulatedAnnealingSampler()

    kwargs: dict[str, Any] = dict(num_reads=num_reads, **sa_kwargs)
    if seed is not None:
        kwargs["seed"] = seed
    if initial_states is not None:
        init = {i: int(initial_states[i]) for i in range(qubo_result.n_bits)}
        init_set = dimod.SampleSet.from_samples(
            [init], vartype=dimod.BINARY, energy=[0.0]
        )
        kwargs["initial_states"] = init_set

    sample_set = sampler.sample(bqm, **kwargs)
    bits = _best_sample_to_array(sample_set, qubo_result.n_bits)
    xi = qubo_result.decode_xi(bits)
    energy = qubo_result.energy(bits)

    res_norm = float("nan")
    if A is not None and B is not None:
        res_norm = float(np.linalg.norm(A @ xi - B))

    return QuboSampleResult(
        bitstring=bits,
        xi=xi,
        energy=energy,
        residual_norm=res_norm,
        num_reads=num_reads,
        sample_set=sample_set,
    )


def sample_exact(
    qubo_result: LinearLsqToQuboResult,
    A: NDArray[np.float64] | None = None,
    B: NDArray[np.float64] | None = None,
) -> QuboSampleResult:
    """Find the exact QUBO ground state by brute-force enumeration.

    Only feasible for small QUBOs (``n_bits <= 20``).  For larger
    problems use :func:`sample_simulated_annealing`.

    Parameters
    ----------
    qubo_result : LinearLsqToQuboResult
        QUBO from :func:`build_linear_lsq_qubo`.
    A, B : ndarray, optional
        Original (un-augmented) system matrices for residual reporting.

    Returns
    -------
    QuboSampleResult
        True ground state of the QUBO.

    Raises
    ------
    ValueError
        If ``n_bits > 20``.
    """
    if qubo_result.n_bits > 20:
        raise ValueError(
            f"ExactSolver is only feasible for n_bits <= 20, "
            f"got {qubo_result.n_bits}. Use sample_simulated_annealing."
        )

    bqm = _qubo_result_to_bqm(qubo_result)
    sampler = dimod.ExactSolver()
    sample_set = sampler.sample(bqm)
    bits = _best_sample_to_array(sample_set, qubo_result.n_bits)
    xi = qubo_result.decode_xi(bits)
    energy = qubo_result.energy(bits)

    res_norm = float("nan")
    if A is not None and B is not None:
        res_norm = float(np.linalg.norm(A @ xi - B))

    return QuboSampleResult(
        bitstring=bits,
        xi=xi,
        energy=energy,
        residual_norm=res_norm,
        num_reads=2 ** qubo_result.n_bits,
        sample_set=sample_set,
    )
