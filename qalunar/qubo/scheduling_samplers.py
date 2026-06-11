"""Unified sampler interface for the scheduling QUBO.

Provides a small set of sampler factories — brute force, simulated
annealing, Leap Hybrid BQM, and the D-Wave QPU — that all share the
signature ``sampler(qubo) -> NDArray[int64]`` expected by
:func:`qalunar.qubo.thrust_scheduling.solve_iterative`.

Sampler factories return a callable plus a metadata dict, so the
caller can swap solvers in benchmark scripts without per-solver
plumbing. Cloud samplers gracefully fail when no token is configured;
the absence of a token is reported via ``available()``.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from qalunar.qubo.thrust_scheduling import ThrustSchedulingQubo


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class SchedulingSampleResult:
    """Standard output of any sampler in this module.

    Attributes
    ----------
    schedule : (M,) ndarray
        Best bitstring found.
    energy : float
        QUBO energy of ``schedule``.
    solve_time : float
        Wall-clock seconds.
    backend : str
        Human-readable backend name (``'brute_force'``, ``'sa'``,
        ``'leap_hybrid'``, ``'dwave_advantage'``, ...).
    metadata : dict
        Extra fields specific to the backend (``num_reads``, ``qpu_time``,
        ``timing``, ``problem_id``, ...).
    """

    schedule: NDArray[np.int64]
    energy: float
    solve_time: float
    backend: str
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# QUBO -> BQM helper
# ---------------------------------------------------------------------------


def _qubo_to_bqm(qubo: ThrustSchedulingQubo):
    """Convert ThrustSchedulingQubo to a dimod BQM.

    Note ``Q`` is symmetric, so ``J_{ij} = 2 * Q_{ij}`` for ``i<j``,
    and ``h_i = Q_{ii} + linear_i``. The constant offset is preserved.
    """
    import dimod

    M = qubo.n_vars
    h: dict[int, float] = {}
    J: dict[tuple[int, int], float] = {}
    for i in range(M):
        h[i] = float(qubo.Q[i, i] + qubo.linear[i])
        for j in range(i + 1, M):
            val = float(2.0 * qubo.Q[i, j])
            if abs(val) > 1e-15:
                J[(i, j)] = val
    return dimod.BinaryQuadraticModel(h, J, qubo.constant, dimod.BINARY)


def _bitstring_from_sample(sample: dict, M: int) -> NDArray[np.int64]:
    bits = np.zeros(M, dtype=np.int64)
    for i in range(M):
        bits[i] = int(sample[i])
    return bits


# ---------------------------------------------------------------------------
# Local samplers
# ---------------------------------------------------------------------------


def sample_brute_force(qubo: ThrustSchedulingQubo) -> SchedulingSampleResult:
    """Exact ground state via enumeration (n_vars ≤ 20)."""
    t0 = time.perf_counter()
    schedule, energy = qubo.brute_force()
    elapsed = time.perf_counter() - t0
    return SchedulingSampleResult(
        schedule=schedule, energy=float(energy),
        solve_time=elapsed, backend="brute_force",
        metadata={"n_evaluations": 2 ** qubo.n_vars},
    )


def sample_kerberos(
    qubo: ThrustSchedulingQubo,
    max_iter: int = 10,
    convergence: int = 3,
    qpu_reads: int = 100,
    qpu_sampler: Any | None = None,
) -> SchedulingSampleResult:
    """Sample via the dwave-hybrid Kerberos reference workflow.

    Kerberos runs three branches in parallel: classical tabu, classical
    simulated annealing, and (optionally) a QPU branch. When
    ``qpu_sampler`` is ``None`` the QPU branch is dropped and the entire
    workflow runs locally with no Leap account required. This makes
    Kerberos a useful upstream proxy for hybrid behaviour at sizes
    where raw simulated annealing gets noisy, without consuming cloud
    minutes.

    Parameters
    ----------
    max_iter : int
        Outer iterations of the Kerberos workflow.
    convergence : int
        Stop after this many iterations without improvement.
    qpu_reads : int
        Reads per QPU call (only used when ``qpu_sampler`` is given).
    qpu_sampler : optional
        QPU sampler object. ``None`` runs Kerberos fully classically.
    """
    import hybrid as h

    bqm = _qubo_to_bqm(qubo)

    t0 = time.perf_counter()
    if qpu_sampler is None:
        # Local-only workflow: tabu + simulated annealing branches.
        iteration = h.RacingBranches(
            h.InterruptableTabuSampler(),
            h.EnergyImpactDecomposer(size=min(50, qubo.n_vars))
                | h.SimulatedAnnealingSubproblemSampler()
                | h.SplatComposer(),
        ) | h.ArgMin()
        workflow = h.Loop(iteration, max_iter=max_iter, convergence=convergence)
        initial = h.State.from_problem(bqm)
        final = workflow.run(initial).result()
    else:
        from hybrid.reference.kerberos import KerberosSampler
        sampler = KerberosSampler()
        sample_set = sampler.sample(
            bqm, max_iter=max_iter, convergence=convergence,
            qpu_reads=qpu_reads, qpu_sampler=qpu_sampler,
        )
        elapsed = time.perf_counter() - t0
        best = sample_set.first.sample
        bits = _bitstring_from_sample(best, qubo.n_vars)
        return SchedulingSampleResult(
            schedule=bits, energy=qubo.energy(bits),
            solve_time=elapsed, backend="kerberos+qpu",
            metadata={
                "max_iter": max_iter, "convergence": convergence,
                "qpu_reads": qpu_reads,
            },
        )
    elapsed = time.perf_counter() - t0

    best = final.samples.first.sample
    bits = _bitstring_from_sample(best, qubo.n_vars)
    return SchedulingSampleResult(
        schedule=bits, energy=qubo.energy(bits),
        solve_time=elapsed, backend="kerberos_local",
        metadata={"max_iter": max_iter, "convergence": convergence},
    )


def sample_simulated_annealing(
    qubo: ThrustSchedulingQubo,
    num_reads: int = 1000,
    seed: int | None = 42,
    **sa_kwargs: Any,
) -> SchedulingSampleResult:
    """Simulated annealing via dwave-neal."""
    import neal

    bqm = _qubo_to_bqm(qubo)
    sampler = neal.SimulatedAnnealingSampler()
    kwargs: dict[str, Any] = dict(num_reads=num_reads, **sa_kwargs)
    if seed is not None:
        kwargs["seed"] = seed

    t0 = time.perf_counter()
    sample_set = sampler.sample(bqm, **kwargs)
    elapsed = time.perf_counter() - t0

    best = sample_set.first.sample
    bits = _bitstring_from_sample(best, qubo.n_vars)
    return SchedulingSampleResult(
        schedule=bits, energy=qubo.energy(bits),
        solve_time=elapsed, backend="sa",
        metadata={"num_reads": num_reads, "seed": seed},
    )


# ---------------------------------------------------------------------------
# Cloud samplers — gracefully fail without a token
# ---------------------------------------------------------------------------


def leap_token_available() -> bool:
    """True if a Leap API token appears to be configured.

    Checks in order: ``DWAVE_API_TOKEN`` env var, then the dwave-cloud
    config file (``~/.config/dwave/dwave.conf`` or platform equivalent).
    """
    if os.environ.get("DWAVE_API_TOKEN"):
        return True
    try:
        from dwave.cloud import Client
        with Client.from_config() as client:
            return bool(client.config.get("token") or client.token)
    except Exception:
        return False


def sample_leap_hybrid(
    qubo: ThrustSchedulingQubo,
    time_limit: float | None = None,
    label: str = "qalunar-scheduling",
) -> SchedulingSampleResult:
    """Submit to D-Wave Leap Hybrid BQM solver.

    Parameters
    ----------
    time_limit : float, optional
        Hint passed to LeapHybridSampler. ``None`` lets the cloud pick
        a default (proportional to problem size).
    label : str
        Solver job label (visible in the Leap dashboard).

    Raises
    ------
    RuntimeError
        If no Leap token is configured.
    """
    if not leap_token_available():
        raise RuntimeError(
            "No Leap API token found. Configure via the DWAVE_API_TOKEN "
            "environment variable or ``dwave config create``."
        )

    from dwave.system import LeapHybridSampler

    bqm = _qubo_to_bqm(qubo)
    sampler = LeapHybridSampler()
    kwargs: dict[str, Any] = dict(label=label)
    if time_limit is not None:
        kwargs["time_limit"] = time_limit

    t0 = time.perf_counter()
    sample_set = sampler.sample(bqm, **kwargs)
    elapsed = time.perf_counter() - t0

    best = sample_set.first.sample
    bits = _bitstring_from_sample(best, qubo.n_vars)
    timing = dict(sample_set.info.get("run_time", {})) if "run_time" in sample_set.info else {}
    return SchedulingSampleResult(
        schedule=bits, energy=qubo.energy(bits),
        solve_time=elapsed, backend="leap_hybrid",
        metadata={
            "qpu_access_time": sample_set.info.get("qpu_access_time"),
            "run_time": sample_set.info.get("run_time"),
            "problem_id": sample_set.info.get("problem_id"),
            "timing": timing,
        },
    )


def sample_dwave_qpu(
    qubo: ThrustSchedulingQubo,
    num_reads: int = 1000,
    chain_strength_multiplier: float = 2.0,
    label: str = "qalunar-scheduling-qpu",
) -> SchedulingSampleResult:
    """Submit to a D-Wave QPU via EmbeddingComposite + DWaveSampler.

    For Advantage / Advantage 2. Embedding is computed automatically.

    Raises
    ------
    RuntimeError
        If no Leap token is configured.
    """
    if not leap_token_available():
        raise RuntimeError(
            "No Leap API token found. Configure via the DWAVE_API_TOKEN "
            "environment variable or ``dwave config create``."
        )

    from dwave.system import DWaveSampler, EmbeddingComposite

    bqm = _qubo_to_bqm(qubo)
    sampler = EmbeddingComposite(DWaveSampler())
    chain_strength = chain_strength_multiplier * float(
        max(abs(qubo.Q.max()), abs(qubo.linear).max(), 1e-9)
    )

    t0 = time.perf_counter()
    sample_set = sampler.sample(
        bqm, num_reads=num_reads,
        chain_strength=chain_strength, label=label,
    )
    elapsed = time.perf_counter() - t0

    best = sample_set.first.sample
    bits = _bitstring_from_sample(best, qubo.n_vars)
    return SchedulingSampleResult(
        schedule=bits, energy=qubo.energy(bits),
        solve_time=elapsed, backend="dwave_qpu",
        metadata={
            "num_reads": num_reads,
            "chain_strength": chain_strength,
            "qpu_timing": sample_set.info.get("timing"),
            "problem_id": sample_set.info.get("problem_id"),
        },
    )


# ---------------------------------------------------------------------------
# Adapter: convert a SchedulingSampleResult-producing function into the
# ``sampler(qubo) -> bitstring`` callable expected by solve_iterative.
# ---------------------------------------------------------------------------


def adapt_for_iterative(
    full_sampler: Callable[[ThrustSchedulingQubo], SchedulingSampleResult],
) -> Callable[[ThrustSchedulingQubo], NDArray[np.int64]]:
    """Wrap a full-result sampler so it returns just the bitstring."""
    def _inner(qubo: ThrustSchedulingQubo) -> NDArray[np.int64]:
        return full_sampler(qubo).schedule
    return _inner
