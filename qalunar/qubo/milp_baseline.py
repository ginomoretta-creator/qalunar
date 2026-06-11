"""Mixed-integer linear programming baseline for the scheduling QUBO.

Provides a *classical, provably-optimal* baseline against which simulated
annealing and quantum annealing can be compared at scales beyond
brute-force enumeration. Uses ``scipy.optimize.milp`` (HiGHS backend),
which is available without external dependencies.

The QUBO energy

    E(q) = q^T Q q + linear^T q + const

with q ∈ {0,1}^M is converted to a pure MILP via the standard McCormick
linearization. For binary q_i, q_i^2 = q_i, so

    q^T Q q  =  Σ_i Q_ii q_i  +  2 Σ_{i<j} Q_ij q_i q_j .

Introduce auxiliary binary y_{ij} = q_i q_j and the three constraints

    y_{ij} ≤ q_i ,   y_{ij} ≤ q_j ,   y_{ij} ≥ q_i + q_j − 1 .

This yields exactly the same global minimum as the QUBO. The solver
returns a certificate of optimality (HiGHS proves the LP-relaxation
lower bound matches the integer-feasible upper bound).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import LinearConstraint, milp, Bounds
from scipy.sparse import lil_matrix

from qalunar.qubo.thrust_scheduling import ThrustSchedulingQubo


@dataclass(frozen=True)
class MilpResult:
    """Result of :func:`solve_scheduling_milp`.

    Attributes
    ----------
    schedule : (M,) ndarray
        Optimal binary schedule.
    energy : float
        ``qubo.energy(schedule)`` — provably the global minimum (up to
        the solver's tolerance).
    is_optimal : bool
        ``True`` if HiGHS proved optimality. ``False`` if the time
        limit was hit before optimality was certified.
    solve_time : float
        Wall-clock seconds.
    n_aux_vars : int
        Number of auxiliary y_{ij} variables introduced.
    """

    schedule: NDArray[np.int64]
    energy: float
    is_optimal: bool
    solve_time: float
    n_aux_vars: int


def solve_scheduling_milp(
    qubo: ThrustSchedulingQubo,
    time_limit: float = 60.0,
    mip_rel_gap: float = 1e-9,
    presolve: bool = True,
    only_nonzero_couplings: bool = True,
) -> MilpResult:
    """Solve the QUBO exactly as a MILP using HiGHS via SciPy.

    Parameters
    ----------
    qubo : ThrustSchedulingQubo
        The scheduling QUBO. Q is assumed symmetric.
    time_limit : float
        HiGHS time limit (seconds).
    mip_rel_gap : float
        Relative MIP optimality gap. Default ``1e-9`` for near-exact.
    presolve : bool
        Enable HiGHS presolve. Usually faster.
    only_nonzero_couplings : bool
        If ``True`` (default), aux variables ``y_{ij}`` are only created
        for pairs with ``Q_ij != 0``. The scheduling QUBO has dense Q in
        general, but exclusion-only or short-arc problems can be sparse.
    """
    import time

    Q = qubo.Q
    lin = qubo.linear
    M = qubo.n_vars

    # Decision: x = [q_0, ..., q_{M-1},  y_{i,j} for each coupled pair]
    # Linear objective coefficients
    c_q = np.array([Q[i, i] + lin[i] for i in range(M)], dtype=np.float64)

    pairs: list[tuple[int, int]] = []
    pair_coeffs: list[float] = []
    for i in range(M):
        for j in range(i + 1, M):
            qij = Q[i, j]
            if only_nonzero_couplings and abs(qij) < 1e-15:
                continue
            pairs.append((i, j))
            pair_coeffs.append(2.0 * qij)
    Y = len(pairs)
    c = np.concatenate([c_q, np.asarray(pair_coeffs, dtype=np.float64)])
    n_total = M + Y

    # Build linearization constraints:  y_{ij} ≤ q_i,  y_{ij} ≤ q_j,
    # y_{ij} ≥ q_i + q_j − 1.
    #
    # For scipy LinearConstraint with lb ≤ A x ≤ ub:
    #   y - q_i ≤ 0     -->  −∞ ≤ -q_i + y ≤ 0
    #   y - q_j ≤ 0     -->  −∞ ≤ -q_j + y ≤ 0
    #   y - q_i - q_j ≥ -1  -->  -1 ≤ -q_i - q_j + y ≤ +∞
    n_rows = 3 * Y
    A = lil_matrix((n_rows, n_total))
    lb = np.empty(n_rows)
    ub = np.empty(n_rows)
    for k, (i, j) in enumerate(pairs):
        col_y = M + k
        # y - q_i ≤ 0
        A[3 * k, i] = -1.0
        A[3 * k, col_y] = 1.0
        lb[3 * k] = -np.inf
        ub[3 * k] = 0.0
        # y - q_j ≤ 0
        A[3 * k + 1, j] = -1.0
        A[3 * k + 1, col_y] = 1.0
        lb[3 * k + 1] = -np.inf
        ub[3 * k + 1] = 0.0
        # -q_i - q_j + y ≥ -1
        A[3 * k + 2, i] = -1.0
        A[3 * k + 2, j] = -1.0
        A[3 * k + 2, col_y] = 1.0
        lb[3 * k + 2] = -1.0
        ub[3 * k + 2] = np.inf

    constraints = LinearConstraint(A.tocsr(), lb, ub)

    integrality = np.ones(n_total, dtype=np.int64)
    bounds = Bounds(lb=np.zeros(n_total), ub=np.ones(n_total))

    options = {
        "time_limit": time_limit,
        "mip_rel_gap": mip_rel_gap,
        "presolve": presolve,
        "disp": False,
    }

    t0 = time.perf_counter()
    res = milp(
        c=c, constraints=constraints, integrality=integrality,
        bounds=bounds, options=options,
    )
    elapsed = time.perf_counter() - t0

    if res.x is None:
        raise RuntimeError(
            f"MILP returned no solution (status={res.status}, "
            f"message={res.message!r})"
        )

    q_opt = np.round(res.x[:M]).astype(np.int64)
    energy = qubo.energy(q_opt)
    # status == 0 means optimal (per scipy docs / HiGHS); time-limit hit
    # gives a non-optimal-but-feasible result.
    is_optimal = bool(res.status == 0)

    return MilpResult(
        schedule=q_opt,
        energy=energy,
        is_optimal=is_optimal,
        solve_time=elapsed,
        n_aux_vars=Y,
    )
