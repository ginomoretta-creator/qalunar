"""Hermite-Simpson direct collocation for the energy-optimal planar CR3BP.

Classical direct-transcription reference solver that plays two roles in
the project:

1. **Warm start for the indirect TFC+ELM iteration.** The sequential
   linearization outer loop in
   :meth:`qalunar.transcription.IndirectTfcElmTranscription.solve_sequential`
   only contracts reliably from initial nominals already in the basin
   of attraction of the target TPBVP. The BC-only Hermite seed
   (``xi = 0``) is fine for short/gentle transfers but stalls on
   realistic LEO->LLO problems. A classical direct-NLP solution, even
   at modest node count, is a much stronger seed.

2. **Classical baseline for the D-Wave benchmark sweep.** The same
   solver produces a CPU reference objective value that the QUBO
   samples are compared against in the benchmark study (project
   Task #9).

Formulation
-----------

Decision variables (per node, ``k = 0 .. N``)::

    x_k = [ x(t_k), y(t_k), vx(t_k), vy(t_k) ]   (4 vars)
    u_k = [ ux(t_k), uy(t_k) ]                   (2 vars)

packed as ``z = [x_0, ..., x_N, u_0, ..., u_N]``, total length
``6 * (N + 1)``. The time grid is uniform on ``[0, T]`` with
``h = T / N``.

Hermite-Simpson enforces the CR3BP dynamics on each interval
``[t_k, t_{k+1}]`` by requiring that the cubic Hermite interpolant
agrees with the dynamics at the midpoint::

    x_mid_k = 0.5 * (x_k + x_{k+1}) + (h / 8) * (f_k - f_{k+1})
    u_mid_k = 0.5 * (u_k + u_{k+1})
    f_mid_k = rhs(x_mid_k, u_mid_k)

    defect_k = x_{k+1} - x_k - (h / 6) * (f_k + 4 f_mid_k + f_{k+1}) = 0

That's ``4 * N`` defect equalities in total. Boundary conditions give
``8`` more equalities (``x_0 = [r0, v0]``, ``x_N = [rf, vf]``).

The cost is a trapezoidal approximation of ``(1/2) integral ||u||^2 dt``,

    J(z) = (h / 2) * sum_k 0.5 * (||u_k||^2 + ||u_{k+1}||^2)

which is a convex quadratic in the control block and identically zero
in the state block. Its gradient is linear in ``u`` and its Hessian is
constant and sparse, so we hand the analytical gradient to SLSQP and
let scipy finite-difference the dynamics constraint Jacobian.

References
----------
* Kelly, M., *An Introduction to Trajectory Optimization: How to Do
  Your Own Direct Collocation*, SIAM Review 59(4):849-904, 2017.
  Canonical reference for Hermite-Simpson.
* Betts, J. T., *Practical Methods for Optimal Control and Estimation
  Using Nonlinear Programming*, 2nd ed., SIAM 2010, Ch. 4.
* De Grossi, Carbone et al., *Pseudospectral QUBO transcription for
  low-thrust trajectory optimization*, Astrodynamics 9:195-215, 2025
  (for the benchmark role).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from qalunar.dynamics import PlanarCR3BP


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DirectCollocationConfig:
    """Hyperparameters for :func:`solve_energy_optimal_cr3bp`.

    Parameters
    ----------
    n_intervals : int, default 40
        Number of Hermite-Simpson intervals. Total nodes ``N + 1`` with
        ``N = n_intervals``. Must be at least 2.
    maxiter : int, default 300
        Maximum SLSQP iterations.
    tol : float, default 1e-8
        Tolerance passed to SLSQP for both objective and constraint
        convergence.
    control_bound : float | None, default None
        Optional symmetric box bound on each control component,
        ``|u_i| <= control_bound``. ``None`` means unconstrained (the
        energy-optimal formulation does not itself need a bound).
    """

    n_intervals: int = 40
    maxiter: int = 300
    tol: float = 1e-8
    control_bound: float | None = None
    feasibility_tol: float = 1e-6


@dataclass
class DirectCollocationResult:
    """Bundle returned by :func:`solve_energy_optimal_cr3bp`.

    Attributes
    ----------
    t : (N+1,) ndarray
        Uniform time grid on ``[0, T]``.
    x, y, vx, vy : (N+1,) ndarrays
        State components at each node.
    ux, uy : (N+1,) ndarrays
        Control components at each node.
    objective : float
        Final value of ``J = (1/2) integral ||u||^2 dt``, integrated with
        the same Simpson rule as the dynamics defects (``u`` linear inside
        each interval), so cost and constraints share one quadrature order.
    success : bool
        ``True`` only if SLSQP reported convergence *and* every defect and
        boundary-condition residual is below ``feasibility_tol``. SLSQP's
        own flag (``ftol`` exit) is kept in ``slsqp_success``.
    slsqp_success : bool
        Raw SLSQP convergence flag.
    n_iterations : int
        SLSQP iteration count (``result.nit``).
    max_defect : float
        Worst absolute value among the Hermite-Simpson defect
        constraints at the solution. A clean reference has this below
        ~1e-6 for ``n_intervals >= 20``.
    max_bc_error : float
        Worst absolute error on the 8 boundary-condition equalities.
    message : str
        SLSQP termination message (useful for diagnosing non-convergence).
    config : DirectCollocationConfig
        The config used for this solve.
    """

    t: NDArray[np.float64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    vx: NDArray[np.float64]
    vy: NDArray[np.float64]
    ux: NDArray[np.float64]
    uy: NDArray[np.float64]
    objective: float
    success: bool
    n_iterations: int
    max_defect: float
    max_bc_error: float
    message: str
    slsqp_success: bool
    config: DirectCollocationConfig = field(repr=False)

    @property
    def n_nodes(self) -> int:
        return int(self.t.shape[0])

    @property
    def time_of_flight(self) -> float:
        return float(self.t[-1] - self.t[0])

    def sample_position(
        self, t_query: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Linearly interpolate ``(x, y)`` onto an arbitrary time grid.

        Intended for use as ``initial_nominal`` to
        :meth:`qalunar.transcription.IndirectTfcElmTranscription.solve_sequential`:

            x_bar, y_bar = ref.sample_position(transcription.t)
            sol = transcription.solve_sequential(initial_nominal=(x_bar, y_bar))

        Linear interpolation is enough because the indirect outer loop
        only uses the nominal to evaluate the Omega gradient/Hessian
        pointwise; those are smooth in ``(x, y)`` and insensitive to
        sub-node interpolation error.

        Parameters
        ----------
        t_query : (m,) ndarray
            Query times in the same nondimensional units as ``self.t``.
            Values outside ``[0, T]`` are clipped to the endpoints.

        Returns
        -------
        (x_q, y_q) : tuple of (m,) ndarrays
            Interpolated position components.
        """
        t_query = np.asarray(t_query, dtype=np.float64)
        x_q = np.interp(t_query, self.t, self.x)
        y_q = np.interp(t_query, self.t, self.y)
        return x_q, y_q


# ---------------------------------------------------------------------------
# Packing helpers
# ---------------------------------------------------------------------------


def _pack(
    states: NDArray[np.float64], controls: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Flatten (N+1, 4) state and (N+1, 2) control blocks into a decision vector."""
    return np.concatenate([states.ravel(), controls.ravel()])


def _unpack(
    z: NDArray[np.float64], n_nodes: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Inverse of :func:`_pack`."""
    n_state = 4
    n_control = 2
    n_x = n_nodes * n_state
    states = z[:n_x].reshape(n_nodes, n_state)
    controls = z[n_x:].reshape(n_nodes, n_control)
    return states, controls


def _rhs_batch(
    dynamics: PlanarCR3BP,
    states: NDArray[np.float64],
    controls: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Vectorized CR3BP RHS on ``(n, 4)`` states and ``(n, 2)`` controls.

    Faster than a Python loop over ``dynamics.rhs``, which matters
    because this gets called inside scipy's finite-difference Jacobian
    assembly at every SLSQP iteration.
    """
    x = states[:, 0]
    y = states[:, 1]
    vx = states[:, 2]
    vy = states[:, 3]

    mu = dynamics.mu
    one_minus_mu = 1.0 - mu
    dx1 = x + mu
    dx2 = x - one_minus_mu
    r1_sq = dx1 * dx1 + y * y
    r2_sq = dx2 * dx2 + y * y
    r1_3 = r1_sq * np.sqrt(r1_sq)
    r2_3 = r2_sq * np.sqrt(r2_sq)

    omega_x = x - one_minus_mu * dx1 / r1_3 - mu * dx2 / r2_3
    omega_y = y - one_minus_mu * y / r1_3 - mu * y / r2_3

    ax = 2.0 * vy + omega_x + controls[:, 0]
    ay = -2.0 * vx + omega_y + controls[:, 1]

    out = np.empty_like(states)
    out[:, 0] = vx
    out[:, 1] = vy
    out[:, 2] = ax
    out[:, 3] = ay
    return out


# ---------------------------------------------------------------------------
# NLP assembly and solve
# ---------------------------------------------------------------------------


def solve_energy_optimal_cr3bp(
    dynamics: PlanarCR3BP,
    r0: NDArray[np.float64],
    v0: NDArray[np.float64],
    rf: NDArray[np.float64],
    vf: NDArray[np.float64],
    time_of_flight: float,
    config: DirectCollocationConfig | None = None,
    initial_guess: NDArray[np.float64] | None = None,
) -> DirectCollocationResult:
    """Solve the energy-optimal planar CR3BP transfer via Hermite-Simpson.

    Parameters
    ----------
    dynamics : PlanarCR3BP
        Planar CR3BP dynamics model.
    r0, v0, rf, vf : (2,) ndarray
        Initial/final position and velocity in synodic nondimensional
        coordinates.
    time_of_flight : float
        Nondimensional flight time ``T > 0``.
    config : DirectCollocationConfig, optional
        Solver hyperparameters. Defaults to ``DirectCollocationConfig()``.
    initial_guess : (6*(N+1),) ndarray, optional
        Packed decision-vector seed ``[state.ravel(); control.ravel()]`` in
        the same layout as the internal ``_pack``. When ``None`` (default)
        the seed is the linear interpolation of the state BCs with zero
        control -- the original behaviour. Supplying a seed enables
        multistart studies of the (nonconvex) transcribed NLP.

    Returns
    -------
    DirectCollocationResult
        Node values, objective, and diagnostics. Inspect ``.success``
        before consuming the trajectory; a failed solve should be
        re-tried with more nodes or a bounded control.
    """
    cfg = config if config is not None else DirectCollocationConfig()

    r0 = np.asarray(r0, dtype=np.float64)
    v0 = np.asarray(v0, dtype=np.float64)
    rf = np.asarray(rf, dtype=np.float64)
    vf = np.asarray(vf, dtype=np.float64)
    for name, arr in (("r0", r0), ("v0", v0), ("rf", rf), ("vf", vf)):
        if arr.shape != (2,):
            raise ValueError(f"{name} must be a length-2 vector, got shape {arr.shape}")
    if not (time_of_flight > 0.0):
        raise ValueError(f"time_of_flight must be > 0, got {time_of_flight}")
    if cfg.n_intervals < 2:
        raise ValueError(f"n_intervals must be >= 2, got {cfg.n_intervals}")
    if cfg.maxiter < 1:
        raise ValueError(f"maxiter must be >= 1, got {cfg.maxiter}")
    if cfg.tol <= 0.0:
        raise ValueError(f"tol must be > 0, got {cfg.tol}")
    if cfg.control_bound is not None and cfg.control_bound <= 0.0:
        raise ValueError(
            f"control_bound must be > 0 when set, got {cfg.control_bound}"
        )

    N = int(cfg.n_intervals)
    n_nodes = N + 1
    h = float(time_of_flight) / N
    t_grid = np.linspace(0.0, time_of_flight, n_nodes)

    # ------------------------------------------------------------------
    # Initial guess: linear interpolation of the state BCs, zero control
    # ------------------------------------------------------------------
    state_lin = np.zeros((n_nodes, 4))
    s0 = np.concatenate([r0, v0])
    sf = np.concatenate([rf, vf])
    alphas = np.linspace(0.0, 1.0, n_nodes)
    state_lin[:] = (1.0 - alphas)[:, None] * s0 + alphas[:, None] * sf
    control_lin = np.zeros((n_nodes, 2))
    z_default = _pack(state_lin, control_lin)
    if initial_guess is None:
        z0 = z_default
    else:
        z0 = np.asarray(initial_guess, dtype=np.float64).ravel()
        if z0.shape != z_default.shape:
            raise ValueError(
                f"initial_guess must have shape {z_default.shape}, "
                f"got {z0.shape}"
            )

    # ------------------------------------------------------------------
    # Objective: J = (1/2) int ||u||^2 dt with Simpson's rule on each
    # interval and u linear inside it (u_mid = (u_k + u_{k+1})/2), the same
    # quadrature the Hermite-Simpson defects use:
    #   J = (h/6) sum_k ( |u_k|^2 + u_k.u_{k+1} + |u_{k+1}|^2 )
    # A trapezoid here (O(h^2)) against O(h^4) defects biased the reported
    # baseline objective high by ~5 % at N = 40.
    # ------------------------------------------------------------------
    def objective(z: NDArray[np.float64]) -> float:
        _, u = _unpack(z, n_nodes)
        uk, uk1 = u[:-1], u[1:]
        per = (np.sum(uk * uk, axis=1) + np.sum(uk * uk1, axis=1)
               + np.sum(uk1 * uk1, axis=1))
        return (h / 6.0) * float(np.sum(per))

    def objective_grad(z: NDArray[np.float64]) -> NDArray[np.float64]:
        _, u = _unpack(z, n_nodes)
        # dJ/du_k = (h/6) (u_{k-1} + 4 u_k + u_{k+1}) inside,
        #           (h/6) (2 u_0 + u_1) and (h/6) (u_{N-1} + 2 u_N) at the ends
        g = np.zeros_like(u)
        g[:-1] += 2.0 * u[:-1] + u[1:]
        g[1:] += u[:-1] + 2.0 * u[1:]
        grad = np.zeros_like(z)
        grad[n_nodes * 4 :].reshape(n_nodes, 2)[:] = (h / 6.0) * g
        return grad

    # ------------------------------------------------------------------
    # Equality constraint: Hermite-Simpson defects
    # ------------------------------------------------------------------
    def defect_residual(z: NDArray[np.float64]) -> NDArray[np.float64]:
        states, controls = _unpack(z, n_nodes)
        f_nodes = _rhs_batch(dynamics, states, controls)     # (N+1, 4)

        x_k = states[:-1]
        x_kp1 = states[1:]
        u_k = controls[:-1]
        u_kp1 = controls[1:]
        f_k = f_nodes[:-1]
        f_kp1 = f_nodes[1:]

        x_mid = 0.5 * (x_k + x_kp1) + (h / 8.0) * (f_k - f_kp1)
        u_mid = 0.5 * (u_k + u_kp1)
        f_mid = _rhs_batch(dynamics, x_mid, u_mid)

        defect = x_kp1 - x_k - (h / 6.0) * (f_k + 4.0 * f_mid + f_kp1)
        return defect.ravel()  # (4 * N,)

    def bc_residual(z: NDArray[np.float64]) -> NDArray[np.float64]:
        states, _ = _unpack(z, n_nodes)
        return np.concatenate(
            [states[0] - s0, states[-1] - sf]  # 8 equations
        )

    constraints: list[dict[str, Any]] = [
        {"type": "eq", "fun": defect_residual},
        {"type": "eq", "fun": bc_residual},
    ]

    # ------------------------------------------------------------------
    # Optional control bounds. SLSQP bounds apply to every variable, so
    # states are unbounded (+/- inf) and only the control block carries
    # the user-supplied box.
    # ------------------------------------------------------------------
    bounds: list[tuple[float, float]] | None = None
    if cfg.control_bound is not None:
        ub = float(cfg.control_bound)
        bounds = [(-np.inf, np.inf)] * (n_nodes * 4) + [(-ub, ub)] * (n_nodes * 2)

    # ------------------------------------------------------------------
    # SLSQP call. We supply the analytical objective gradient; scipy
    # finite-differences the constraint Jacobian. For n_intervals ~ 40
    # this is a few hundred variables / few hundred constraints, which
    # SLSQP handles in under a second.
    # ------------------------------------------------------------------
    result = minimize(
        objective,
        z0,
        jac=objective_grad,
        method="SLSQP",
        constraints=constraints,
        bounds=bounds,
        options={
            "maxiter": cfg.maxiter,
            "ftol": cfg.tol,
            "disp": False,
        },
    )

    states, controls = _unpack(result.x, n_nodes)

    defect = defect_residual(result.x).reshape(N, 4)
    bc_err = bc_residual(result.x)
    max_defect = float(np.max(np.abs(defect))) if defect.size else 0.0
    max_bc_error = float(np.max(np.abs(bc_err)))

    return DirectCollocationResult(
        t=t_grid,
        x=states[:, 0].copy(),
        y=states[:, 1].copy(),
        vx=states[:, 2].copy(),
        vy=states[:, 3].copy(),
        ux=controls[:, 0].copy(),
        uy=controls[:, 1].copy(),
        objective=float(result.fun),
        success=bool(result.success) and max_defect <= cfg.feasibility_tol
        and max_bc_error <= cfg.feasibility_tol,
        slsqp_success=bool(result.success),
        n_iterations=int(result.nit),
        max_defect=max_defect,
        max_bc_error=max_bc_error,
        message=str(result.message),
        config=cfg,
    )
