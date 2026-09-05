"""Cardinality-constrained slot selection as a QUBO.

The duty-cycle-limited scheduling problem of the CubeSat and SMART-1 case
studies is: given per-slot gains ``g_j`` measured in the truth model (e.g.
the perigee-radius gain from firing only slot ``j``), choose at most ``k``
slots that maximise the summed gain,

    max  sum_j g_j q_j    s.t.  sum_j q_j = k,   q_j in {0, 1}.

As a QUBO this is

    min  -sum_j g_j q_j + lam * (sum_j q_j - k)^2,

whose quadratic term couples every pair of slots. For a *linear* objective the
ground state is provably the top-``k`` set (sort and take the largest ``k``
gains), so the sampler is not doing anything a sort cannot do; the value of
casting it as a QUBO is (a) that the same form accepts pairwise interaction
terms when the slot effects do not superpose, and (b) that it is the object a
quantum annealer ingests. Both facts are stated plainly here so callers do not
report a sort as an annealing result.

Penalty weight
--------------
Changing the burn count by one costs at least ``lam`` in the penalty term, and
gains at most ``max |g_j|`` in the objective, so any ``lam > max |g_j|``
makes every constraint-violating state energetically unfavourable. The default
uses ``lam = 2 max |g_j|`` -- the smallest safe value with a factor-two margin,
so the coefficient dynamic range stays as small as possible (matters on analog
hardware).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "CardinalityQubo",
    "build_cardinality_qubo",
    "solve_cardinality_qubo",
    "top_k",
]


@dataclass(frozen=True)
class CardinalityQubo:
    """QUBO ``min q^T Q q + l^T q + c`` for cardinality-``k`` selection."""

    Q: NDArray[np.float64]
    linear: NDArray[np.float64]
    constant: float
    k: int
    penalty: float
    gains: NDArray[np.float64]

    @property
    def n_vars(self) -> int:
        return int(self.gains.size)

    def energy(self, q: NDArray[np.int64]) -> float:
        qf = np.asarray(q, dtype=np.float64)
        return float(qf @ self.Q @ qf + self.linear @ qf + self.constant)

    def coefficient_range(self) -> float:
        """max |coefficient| / min nonzero |coefficient| -- the dynamic range an
        analog annealer must resolve."""
        coeffs = np.concatenate([np.abs(self.Q[np.triu_indices(self.n_vars)]),
                                 np.abs(self.linear)])
        nz = coeffs[coeffs > 0]
        return float(nz.max() / nz.min()) if nz.size else 1.0


def top_k(gains: NDArray[np.float64], k: int) -> NDArray[np.int64]:
    """Closed-form ground state of the linear cardinality problem."""
    g = np.asarray(gains, dtype=np.float64)
    q = np.zeros(g.size, dtype=np.int64)
    if k > 0:
        q[np.argsort(g, kind="stable")[::-1][:k]] = 1
    return q


def build_cardinality_qubo(
    gains: NDArray[np.float64],
    k: int,
    penalty: float | None = None,
) -> CardinalityQubo:
    """Assemble the symmetric-``Q`` form of the cardinality QUBO.

    ``q^T Q q`` with symmetric ``Q`` counts each off-diagonal pair twice, so
    the pairwise penalty ``2 lam q_i q_j`` is stored as ``lam`` on both sides.
    """
    g = np.asarray(gains, dtype=np.float64)
    n = g.size
    if not (0 <= k <= n):
        raise ValueError(f"k={k} outside [0, {n}]")
    lam = float(penalty) if penalty is not None else 2.0 * float(np.max(np.abs(g)))
    if lam <= float(np.max(np.abs(g))):
        raise ValueError("penalty must exceed max |gain| to enforce the budget")
    # (sum q - k)^2 = sum_i q_i (1 - 2k) + 2 sum_{i<j} q_i q_j + k^2   (q^2 = q)
    Q = np.full((n, n), lam, dtype=np.float64)
    np.fill_diagonal(Q, 0.0)
    linear = -g + lam * (1.0 - 2.0 * k)
    constant = lam * float(k) ** 2
    return CardinalityQubo(Q=Q, linear=linear, constant=constant, k=k,
                           penalty=lam, gains=g)


def solve_cardinality_qubo(
    gains: NDArray[np.float64],
    k: int,
    *,
    num_reads: int = 2000,
    seed: int | None = 1,
    penalty: float | None = None,
) -> tuple[NDArray[np.int64], dict[str, Any]]:
    """Sample the cardinality QUBO with simulated annealing (dwave-neal).

    Returns the best bitstring and a diagnostics dict containing the penalty
    used, the coefficient dynamic range, whether the budget was met, and
    whether the sample equals the closed-form top-``k`` ground state.
    """
    import dimod
    import neal

    qubo = build_cardinality_qubo(gains, k, penalty)
    n = qubo.n_vars
    h = {i: float(qubo.Q[i, i] + qubo.linear[i]) for i in range(n)}
    J = {(i, j): float(2.0 * qubo.Q[i, j]) for i in range(n) for j in range(i + 1, n)}
    bqm = dimod.BinaryQuadraticModel(h, J, qubo.constant, dimod.BINARY)
    kwargs: dict[str, Any] = {"num_reads": num_reads}
    if seed is not None:
        kwargs["seed"] = seed
    ss = neal.SimulatedAnnealingSampler().sample(bqm, **kwargs)
    best = ss.first.sample
    q = np.array([int(best[i]) for i in range(n)], dtype=np.int64)
    ref = top_k(gains, k)
    info = {
        "penalty": qubo.penalty,
        "coefficient_range": qubo.coefficient_range(),
        "budget_met": int(q.sum()) == k,
        "matches_top_k": bool(np.array_equal(q, ref)),
        "energy": qubo.energy(q),
        "energy_top_k": qubo.energy(ref),
        "num_reads": num_reads,
        "seed": seed,
    }
    return q, info
