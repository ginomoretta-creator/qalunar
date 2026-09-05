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
    seed: int | None = 42,
) -> SchedulingSampleResult:
    """Sample via the dwave-hybrid Kerberos reference workflow.

    Kerberos is a *classical* local workflow (tabu + simulated annealing on
    decomposed subproblems) unless a ``qpu_sampler`` is supplied; it is a
    hybrid-solver proxy, not a quantum result, and is labelled as such in
    ``backend``.

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
    seed : int or None
        Seeds the global Python/NumPy generators the workflow draws from
        (dwave-hybrid exposes no per-sampler seed).
    """
    import hybrid as h

    bqm = _qubo_to_bqm(qubo)

    t0 = time.perf_counter()
    if qpu_sampler is None:
        # Local-only workflow: tabu + simulated annealing branches.
        # dwave-hybrid's samplers expose no seed argument; the workflow draws
        # from Python's and NumPy's global generators, which are seeded here
        # so a run is reproducible for a given ``seed``.
        if seed is not None:
            import random
            random.seed(seed)
            np.random.seed(seed % (2 ** 32))
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
        metadata={"max_iter": max_iter, "convergence": convergence, "seed": seed},
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
# Quantum-inspired classical tier: ballistic simulated bifurcation
# ---------------------------------------------------------------------------


def sample_simulated_bifurcation(
    qubo: ThrustSchedulingQubo,
    num_reads: int = 100,
    n_steps: int = 3000,
    dt: float = 1.0,
    seed: int | None = 42,
    discrete: bool = True,
) -> SchedulingSampleResult:
    """Simulated bifurcation (discrete SB by default) in NumPy.

    SB integrates a set of Kerr-parametric-oscillator-inspired classical
    equations whose fixed points at the end of a bifurcation sweep encode Ising
    ground states (Goto, Tatsumura & Dixon, Sci. Adv. 5, eaav2372, 2019;
    ballistic and discrete variants: Goto et al., Sci. Adv. 7, eabe7953,
    2021). Benchmarks
    place SB and its relatives at the front of the quantum-inspired heuristics
    (Zeng et al., Commun. Phys. 7, 249, 2024), which makes it the honest
    classical proxy for an annealer when no QPU is available.

    The QUBO ``q^T Q q + l^T q`` is mapped to Ising spins ``q = (1 + s)/2``:
    ``H(s) = s^T J s + h^T s`` with ``J = Q/4`` (diagonal dropped, it is a
    constant for spins) and ``h = (Q 1 + l)/2``. ``num_reads`` trajectories
    with independent random initial positions are integrated in parallel with
    a symplectic Euler scheme; positions are clipped to ``[-1, 1]`` with the
    velocity reset (the wall), and the sign of the final position is the
    spin. With ``discrete=True`` (default) the force uses ``sign(x)`` instead
    of ``x`` (dSB), which on these low-rank, linear-dominated instances
    recovered the brute-force optimum in 60-80 % of 100-read runs at N = 15-20
    where bSB recovered none; ``dt = 1`` and 3000 steps were the best of the
    sweep in ``tests``/notes. Success probability below one is handled by the
    time-to-solution accounting of the benchmark, not hidden.
    """
    rng = np.random.default_rng(seed)
    M = qubo.n_vars
    Q = np.asarray(qubo.Q, dtype=np.float64)
    lin = np.asarray(qubo.linear, dtype=np.float64)
    J = 0.25 * Q
    np.fill_diagonal(J, 0.0)
    h = 0.5 * (Q.sum(axis=1) + lin)
    # Coupling scale for the c0 normalisation (Goto's 0.5 / (sigma sqrt(N)));
    # h is included so a linear-dominated instance is not under-driven.
    offdiag = 2.0 * J[~np.eye(M, dtype=bool)]
    sigma = float(np.sqrt(np.mean(offdiag ** 2))) or 1e-12
    c0 = 0.5 / (sigma * np.sqrt(M))
    a0 = 1.0

    t0 = time.perf_counter()
    x = rng.uniform(-0.1, 0.1, size=(M, num_reads))
    y = np.zeros_like(x)
    for k in range(n_steps):
        a = a0 * (k + 1) / n_steps
        xs = np.sign(x) if discrete else x
        grad = 2.0 * (J @ xs) + h[:, None]         # dH/dx (dSB: on sign(x))
        y += dt * (-(a0 - a) * x - c0 * grad)
        x += dt * a0 * y
        wall = np.abs(x) > 1.0
        x[wall] = np.sign(x[wall])
        y[wall] = 0.0
    s_final = np.where(x >= 0.0, 1.0, -1.0)
    q_all = (0.5 * (1.0 + s_final)).astype(np.int64)      # (M, R)
    energies = (np.einsum("ir,ij,jr->r", q_all, Q, q_all)
                + lin @ q_all + qubo.constant)
    best = int(np.argmin(energies))
    elapsed = time.perf_counter() - t0
    bits = q_all[:, best].copy()
    return SchedulingSampleResult(
        schedule=bits, energy=qubo.energy(bits),
        solve_time=elapsed, backend="sb",
        metadata={"num_reads": num_reads, "n_steps": n_steps, "dt": dt,
                  "seed": seed, "c0": c0, "discrete": discrete,
                  "all_energies": np.sort(energies)[:10].tolist()},
    )


def sample_simulated_bifurcation_torch(
    qubo: ThrustSchedulingQubo,
    num_reads: int = 128,
    max_steps: int = 10_000,
    seed: int | None = 42,
    ballistic: bool = False,
    heated: bool = False,
    **kwargs: Any,
) -> SchedulingSampleResult:
    """Simulated bifurcation via the published ``simulated-bifurcation``
    package (Toshiba-SB reference implementation in PyTorch; Goto et al. 2019,
    2021). This is the quantum-inspired tier used in the benchmark: an
    independently maintained implementation removes the "poorly tuned in-house
    SB" objection that the NumPy version (:func:`sample_simulated_bifurcation`)
    would invite. ``num_reads`` maps to the package's ``agents``.

    Raises ``ImportError`` if the package (and torch) are not installed; it is
    an optional extra (``pip install qalunar[benchmarks]``).
    """
    import inspect

    import simulated_bifurcation as sbpkg
    import torch

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32))
    Q = torch.tensor(np.asarray(qubo.Q, dtype=np.float64))
    lin = torch.tensor(np.asarray(qubo.linear, dtype=np.float64))
    params = inspect.signature(sbpkg.minimize).parameters
    call: dict[str, Any] = {}
    for k, v in {
        "vector": lin, "constant": float(qubo.constant), "input_type": "binary",
        "domain": "binary", "agents": num_reads, "max_steps": max_steps,
        "ballistic": ballistic, "heated": heated, "best_only": True,
        "verbose": False, "dtype": torch.float64,
    }.items():
        if k in params:
            call[k] = v
    call.update({k: v for k, v in kwargs.items() if k in params})
    t0 = time.perf_counter()
    out = sbpkg.minimize(Q, **call)
    elapsed = time.perf_counter() - t0
    vec = out[0] if isinstance(out, (tuple, list)) else out
    vec = np.asarray(vec.detach().cpu().numpy() if hasattr(vec, "detach") else vec)
    if vec.ndim > 1:
        vec = vec[0]
    bits = np.asarray(np.rint(vec), dtype=np.int64).ravel()
    if bits.min() < 0:          # spin output: map -1/+1 -> 0/1
        bits = ((bits + 1) // 2).astype(np.int64)
    return SchedulingSampleResult(
        schedule=bits, energy=qubo.energy(bits),
        solve_time=elapsed, backend="sb_torch",
        metadata={"num_reads": num_reads, "max_steps": max_steps, "seed": seed,
                  "ballistic": ballistic, "heated": heated,
                  "package_version": getattr(sbpkg, "__version__", "?")},
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
    # LeapHybridSampler reports ``run_time`` as a scalar (microseconds),
    # not a mapping; coercing it with dict() raised TypeError after the
    # cloud job had already been charged.
    timing = sample_set.info.get("run_time")
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


def chain_strength_for(qubo: ThrustSchedulingQubo, multiplier: float = 2.0) -> float:
    """Chain strength as a multiple of the largest |coefficient| in the QUBO.

    Uses ``max|Q|`` (not ``|max Q|``): a negative coupling of large magnitude
    would otherwise be ignored and chains could break on it.
    """
    scale = max(float(np.abs(qubo.Q).max()), float(np.abs(qubo.linear).max()), 1e-9)
    return multiplier * scale


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
    chain_strength = chain_strength_for(qubo, chain_strength_multiplier)

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
