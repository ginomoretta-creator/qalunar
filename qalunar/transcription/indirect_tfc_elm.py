"""Indirect TFC+ELM transcription of the planar energy-optimal CR3BP.

This module ports the Theory of Functional Connections + Extreme Learning
Machine pipeline from Gino Moretta's MATLAB ``Rendezvous_GIno.m`` (which
uses linear Hill-Clohessy-Wiltshire dynamics) to the nonlinear planar
CR3BP via a sequential linearization around a nominal trajectory.

Given boundary conditions ``(r0, v0) -> (rf, vf)`` and a flight time
``T``, the transcription produces a linear system

    A @ xi = B

whose least-squares solution gives coefficients for a closed-form
representation of the state and costates on ``[0, T]``. The unknowns
are blocked as

    xi = (xi_x, xi_y, xi_lrx, xi_lry, xi_lvx, xi_lvy)

each of length ``L = n_basis``. There are 6 coordinate fields:

* ``x, y``       -- planar position components,
* ``lambda_rx, lambda_ry`` -- position costates,
* ``lambda_vx, lambda_vy`` -- velocity costates.

The optimal control is given by Pontryagin's minimum principle for the
energy-optimal cost ``J = (1/2) int ||u||^2 dt``, which yields
``u* = -lambda_v``. After substitution the TPBVP becomes a coupled
system of first/second-order ODEs in ``(r, lambda_r, lambda_v)``.

TFC + ELM in one paragraph
--------------------------

For the position unknowns we build a constrained expression

    x(z) = F(z) @ xi_x + Cx(z)

where ``Cx`` is a cubic Hermite interpolant of the position and
velocity boundary values, and ``F`` is constructed from an ELM basis
``h(z)`` so that ``F(z0) = F(zf) = F'(z0) = F'(zf) = 0``. The net
effect is that *any* choice of ``xi_x`` gives an ``x(z)`` that
satisfies the four boundary conditions on ``(x, dx/dt)`` exactly. The
costate unknowns, which carry no boundary conditions, are represented
directly on the raw ELM basis ``H`` with ``xi_lrx`` etc.

Substituting this ansatz into the linearized optimality conditions
yields a ``6n x 6L`` linear residual system sampled at ``n`` training
points. We solve it either classically (``lsqminnorm``) or by encoding
it into a QUBO with :func:`qalunar.qubo.build_linear_lsq_qubo` and
sampling the QUBO with a quantum or simulated annealer.

Sequential linearization outer loop
-----------------------------------

Because the CR3BP is nonlinear, the gradient and Hessian of the
synodic pseudo-potential ``Omega`` must be evaluated at a *nominal*
trajectory ``(x_bar, y_bar)``. After one solve, the new trajectory
replaces the nominal and the system is reassembled and resolved, until
the change between iterates drops below a tolerance. The first-
iteration nominal can be any initial guess; :meth:`initial_nominal`
returns the cubic Hermite interpolant of the BCs (i.e. ``xi = 0``),
which is a cheap and usable seed.

References
----------
* D. Mortari, *The Theory of Connections*, Mathematics 5:57, 2017.
* G.-B. Huang et al., *Extreme learning machine: theory and
  applications*, Neurocomputing 70:489-501, 2006.
* De Grossi, Carbone et al., *Pseudospectral QUBO transcription for
  low-thrust trajectory optimization*, Astrodynamics 9:195-215, 2025.
* Gino Moretta, ``Rendezvous_GIno.m`` (QUBO v2), Sapienza Rome, 2025.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from qalunar.dynamics import PlanarCR3BP

Activation = Literal["tanh"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OuterLoopConfig:
    """Configuration for the sequential linearization outer loop.

    The outer loop updates the nominal via a damped fixed-point step,

        nominal_new = nominal_old + alpha * (candidate - nominal_old),

    where ``candidate`` is the trajectory decoded from the linearized
    solve at the current nominal. When ``line_search=True`` (the
    default) ``alpha`` is chosen by backtracking on the nonlinear
    residual of the full TPBVP: start at ``damping``, halve it until
    the Armijo condition
    ``nl_res(alpha) <= (1 - armijo_c * alpha) * nl_res_prev`` holds, or
    until ``alpha`` drops below ``min_damping``. When
    ``line_search=False`` the loop takes a fixed step of size
    ``damping`` every iteration (the Pass-1 behavior).

    Parameters
    ----------
    max_iter : int, default 50
        Maximum number of outer iterations.
    tol : float, default 1e-6
        Convergence tolerance on the infinity-norm of the undamped
        fixed-point residual ``max(max|x - x_bar|, max|y - y_bar|)``.
    damping : float, default 1.0
        Largest trial step for the line search (and the fixed step when
        ``line_search=False``). Must lie in ``(0, 1]``. ``damping = 1``
        recovers the undamped Picard step in the good case.
    line_search : bool, default True
        When ``True``, adaptively backtrack from ``damping`` down to
        ``min_damping`` using the Armijo condition on the nonlinear
        residual. When ``False``, take a fixed step of size ``damping``
        every iteration.
    min_damping : float, default 1.0 / 1024
        Lower bound on ``alpha`` during backtracking. If the Armijo
        condition is still not satisfied at this floor, the loop
        commits the minimum-damping step anyway and moves on so it can
        report a non-convergence diagnostic upstream. Ignored when
        ``line_search=False``.
    backtrack_factor : float, default 0.5
        Multiplicative shrink applied to ``alpha`` on each failed
        Armijo check. Must lie in ``(0, 1)``. Ignored when
        ``line_search=False``.
    armijo_c : float, default 1e-4
        Slope of the Armijo sufficient-decrease line. ``1e-4`` is the
        standard "any decrease" value. Ignored when
        ``line_search=False``.
    """

    max_iter: int = 50
    tol: float = 1e-6
    damping: float = 1.0
    line_search: bool = True
    min_damping: float = 1.0 / 1024.0
    backtrack_factor: float = 0.5
    armijo_c: float = 1e-4


@dataclass
class SequentialSolution:
    """Result of :meth:`IndirectTfcElmTranscription.solve_sequential`.

    Attributes
    ----------
    xi : (6 L,) ndarray
        Final coefficient vector at termination.
    trajectory : dict of (n,) ndarrays
        Decoded state/costate/control at the training points, with the
        same keys as :meth:`IndirectTfcElmTranscription.decode_trajectory`.
    converged : bool
        ``True`` if the undamped step norm fell below the configured
        tolerance. ``False`` if the loop hit ``max_iter`` first.
    n_iterations : int
        Number of outer iterations actually executed.
    history : dict of lists of floats
        Per-iteration diagnostics, one entry per executed iteration:

        * ``trajectory_change_inf``: the undamped step infinity-norm
          ``max(max|dx|, max|dy|)``. Drives the convergence test.
        * ``linearized_residual``: ``||A xi - B||_2`` at the lsq
          solution. Should stay close to zero if the matrix is full
          row rank; sudden growth signals loss of conditioning.
        * ``nonlinear_residual``: infinity norm of the full nonlinear
          dynamics residual across all 6 equations and all training
          points. Shrinks only as the outer loop actually converges
          toward a solution of the true CR3BP TPBVP.
        * ``step_size``: damping factor ``alpha`` actually taken at the
          end of this iteration after backtracking (0.0 on the
          convergence-break iteration, since no step was committed).
    """

    xi: NDArray[np.float64]
    trajectory: dict[str, NDArray[np.float64]]
    converged: bool
    n_iterations: int
    history: dict[str, list[float]]


@dataclass(frozen=True)
class IndirectTfcElmConfig:
    """Hyperparameters for :class:`IndirectTfcElmTranscription`.

    Parameters
    ----------
    n_training : int, default 20
        Number of collocation points on the mapped interval ``[-1, 1]``.
    n_basis : int, default 80
        Number of ELM basis functions (``L`` in the MATLAB code).
    activation : {"tanh"}, default "tanh"
        ELM activation function. Only tanh is currently implemented.
    weight_range : tuple[float, float], default (-3.0, 3.0)
        Uniform range for the random ELM input weights.
    bias_range : tuple[float, float], default (-3.0, 3.0)
        Uniform range for the random ELM biases.
    seed : int, default 0
        RNG seed used for the random weights/biases. A fixed seed keeps
        the ELM reproducible across outer iterations and between runs.
    """

    n_training: int = 20
    n_basis: int = 80
    activation: Activation = "tanh"
    weight_range: tuple[float, float] = (-3.0, 3.0)
    bias_range: tuple[float, float] = (-3.0, 3.0)
    seed: int = 0


# ---------------------------------------------------------------------------
# Low-level helpers (pure functions, tested directly)
# ---------------------------------------------------------------------------


def _hermite_switching(
    z: NDArray[np.float64], z0: float = -1.0, zf: float = 1.0
) -> tuple[
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
]:
    """Cubic Hermite switching functions on ``[z0, zf]``.

    The four switching functions ``om1..om4`` are the dual basis for
    imposing value-and-derivative constraints at the endpoints. For any
    scalar function ``h`` the combination

        g(z) = h(z) - om1*h(z0) - om2*h(zf) - om3*h'(z0) - om4*h'(zf)

    satisfies ``g(z0) = g(zf) = g'(z0) = g'(zf) = 0``. Equivalently:

    * ``om1(z0)=1, om1(zf)=0, om1'(z0)=0, om1'(zf)=0``
    * ``om2(z0)=0, om2(zf)=1, om2'(z0)=0, om2'(zf)=0``
    * ``om3(z0)=0, om3(zf)=0, om3'(z0)=1, om3'(zf)=0``
    * ``om4(z0)=0, om4(zf)=0, om4'(z0)=0, om4'(zf)=1``

    Returns
    -------
    (vals, d_vals, dd_vals) : triple of 4-tuples
        ``vals = (om1, om2, om3, om4)`` and the corresponding first and
        second derivatives in ``z``. Each array has the shape of ``z``.
    """
    dz = zf - z0
    u = z - z0
    u2 = u * u
    u3 = u2 * u
    d2 = dz * dz
    d3 = d2 * dz

    om1 = 1.0 + 2.0 * u3 / d3 - 3.0 * u2 / d2
    om2 = -2.0 * u3 / d3 + 3.0 * u2 / d2
    om3 = u + u3 / d2 - 2.0 * u2 / dz
    om4 = u3 / d2 - u2 / dz

    om1d = 6.0 * u2 / d3 - 6.0 * u / d2
    om2d = -6.0 * u2 / d3 + 6.0 * u / d2
    om3d = 1.0 + 3.0 * u2 / d2 - 4.0 * u / dz
    om4d = 3.0 * u2 / d2 - 2.0 * u / dz

    om1dd = 12.0 * u / d3 - 6.0 / d2
    om2dd = -12.0 * u / d3 + 6.0 / d2
    om3dd = 6.0 * u / d2 - 4.0 / dz
    om4dd = 6.0 * u / d2 - 2.0 / dz

    return (
        (om1, om2, om3, om4),
        (om1d, om2d, om3d, om4d),
        (om1dd, om2dd, om3dd, om4dd),
    )


def _tanh_basis(
    z: NDArray[np.float64],
    weights: NDArray[np.float64],
    biases: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Tanh ELM basis and its first two derivatives with respect to ``z``.

    For neuron ``j`` with input weight ``w_j`` and bias ``b_j``:

        sigma_j(z)   = tanh(w_j z + b_j)
        sigma'_j(z)  = w_j * (1 - sigma_j^2)
        sigma''_j(z) = -2 w_j^2 * sigma_j * (1 - sigma_j^2)

    Returns
    -------
    (h, h_d, h_dd) : tuple of (n, L) arrays
        Stacked neuron activations and their first and second
        ``z``-derivatives at each training point.
    """
    arg = z[:, None] * weights[None, :] + biases[None, :]
    h = np.tanh(arg)
    one_minus_h2 = 1.0 - h * h
    h_d = weights[None, :] * one_minus_h2
    h_dd = -2.0 * (weights[None, :] ** 2) * h * one_minus_h2
    return h, h_d, h_dd


def _omega_grad_hess(
    mu: float, x: NDArray[np.float64], y: NDArray[np.float64]
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Gradient and Hessian of the synodic pseudo-potential on arrays.

    Computes ``(Omega_x, Omega_y, Omega_xx, Omega_xy, Omega_yy)`` at
    every ``(x_i, y_i)`` pair, where

        Omega = (x^2 + y^2)/2 + (1 - mu)/r1 + mu/r2 + mu (1 - mu)/2

    with ``r1 = sqrt((x + mu)^2 + y^2)`` (distance from Earth) and
    ``r2 = sqrt((x - 1 + mu)^2 + y^2)`` (distance from Moon). The
    returned arrays share the broadcasted shape of ``x`` and ``y``.
    """
    one_minus_mu = 1.0 - mu
    dx1 = x + mu
    dx2 = x - one_minus_mu
    r1_sq = dx1 * dx1 + y * y
    r2_sq = dx2 * dx2 + y * y
    r1 = np.sqrt(r1_sq)
    r2 = np.sqrt(r2_sq)
    r1_3 = r1 * r1_sq
    r2_3 = r2 * r2_sq
    r1_5 = r1_3 * r1_sq
    r2_5 = r2_3 * r2_sq

    omega_x = x - one_minus_mu * dx1 / r1_3 - mu * dx2 / r2_3
    omega_y = y - one_minus_mu * y / r1_3 - mu * y / r2_3

    omega_xx = (
        1.0
        - one_minus_mu / r1_3
        - mu / r2_3
        + 3.0 * one_minus_mu * dx1 * dx1 / r1_5
        + 3.0 * mu * dx2 * dx2 / r2_5
    )
    omega_yy = (
        1.0
        - one_minus_mu / r1_3
        - mu / r2_3
        + 3.0 * one_minus_mu * y * y / r1_5
        + 3.0 * mu * y * y / r2_5
    )
    omega_xy = (
        3.0 * one_minus_mu * dx1 * y / r1_5
        + 3.0 * mu * dx2 * y / r2_5
    )

    return omega_x, omega_y, omega_xx, omega_xy, omega_yy


# ---------------------------------------------------------------------------
# Transcription class
# ---------------------------------------------------------------------------


@dataclass
class IndirectTfcElmTranscription:
    """Indirect TFC+ELM transcription of the planar energy-optimal CR3BP TPBVP.

    Parameters
    ----------
    dynamics : PlanarCR3BP
        Planar CR3BP dynamics model (only ``mu`` is actually consumed).
    r0, v0, rf, vf : (2,) ndarray
        Initial and final position/velocity boundary conditions in
        synodic nondimensional coordinates.
    time_of_flight : float
        Nondimensional flight time ``T > 0``.
    config : IndirectTfcElmConfig
        Hyperparameters (basis size, training points, RNG seed, ...).

    Notes
    -----
    All of the "static" (nominal-independent) quantities -- the ELM
    basis, its endpoint values, the Hermite switching functions, the
    constrained-expression matrix ``F`` and its derivatives, and the
    BC-only contributions ``Cx, Cy, Cx'(t), Cy'(t), Cx''(t), Cy''(t)``
    -- are precomputed once in ``__post_init__``. Each outer iteration
    then only reevaluates the ``Omega`` gradient/Hessian on the current
    nominal trajectory and reassembles the linear system.
    """

    dynamics: PlanarCR3BP
    r0: NDArray[np.float64]
    v0: NDArray[np.float64]
    rf: NDArray[np.float64]
    vf: NDArray[np.float64]
    time_of_flight: float
    config: IndirectTfcElmConfig = field(default_factory=IndirectTfcElmConfig)

    # Precomputed fields (populated in __post_init__).
    z: NDArray[np.float64] = field(init=False, repr=False)
    t: NDArray[np.float64] = field(init=False, repr=False)
    c: float = field(init=False, repr=False)
    F: NDArray[np.float64] = field(init=False, repr=False)
    Fd: NDArray[np.float64] = field(init=False, repr=False)
    Fdd: NDArray[np.float64] = field(init=False, repr=False)
    H: NDArray[np.float64] = field(init=False, repr=False)
    Hd: NDArray[np.float64] = field(init=False, repr=False)
    Cx: NDArray[np.float64] = field(init=False, repr=False)
    Cy: NDArray[np.float64] = field(init=False, repr=False)
    Cxd: NDArray[np.float64] = field(init=False, repr=False)
    Cyd: NDArray[np.float64] = field(init=False, repr=False)
    Cxdd: NDArray[np.float64] = field(init=False, repr=False)
    Cydd: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        cfg = self.config
        if cfg.n_training < 2:
            raise ValueError(f"n_training must be >= 2, got {cfg.n_training}")
        if cfg.n_basis < 1:
            raise ValueError(f"n_basis must be >= 1, got {cfg.n_basis}")
        if cfg.activation != "tanh":
            raise ValueError(
                f"only activation='tanh' is currently supported, got {cfg.activation!r}"
            )
        if not (self.time_of_flight > 0.0):
            raise ValueError(
                f"time_of_flight must be positive, got {self.time_of_flight}"
            )

        self.r0 = np.asarray(self.r0, dtype=np.float64)
        self.v0 = np.asarray(self.v0, dtype=np.float64)
        self.rf = np.asarray(self.rf, dtype=np.float64)
        self.vf = np.asarray(self.vf, dtype=np.float64)
        for name, arr in (
            ("r0", self.r0),
            ("v0", self.v0),
            ("rf", self.rf),
            ("vf", self.vf),
        ):
            if arr.shape != (2,):
                raise ValueError(
                    f"{name} must be a length-2 vector, got shape {arr.shape}"
                )

        # --------------------------------------------------------------
        # Mapped interval and physical-time grid
        # --------------------------------------------------------------
        z0, zf = -1.0, 1.0
        self.z = np.linspace(z0, zf, cfg.n_training)
        # z = z0 + c * (t - t0) with t0 = 0 and zf - z0 = c * T.
        self.c = (zf - z0) / self.time_of_flight
        self.t = (self.z - z0) / self.c
        c = self.c
        c2 = c * c

        # --------------------------------------------------------------
        # Random ELM weights/biases (reproducible via cfg.seed)
        # --------------------------------------------------------------
        rng = np.random.default_rng(cfg.seed)
        lo_w, hi_w = cfg.weight_range
        lo_b, hi_b = cfg.bias_range
        self._weights = rng.uniform(lo_w, hi_w, size=cfg.n_basis)
        self._biases = rng.uniform(lo_b, hi_b, size=cfg.n_basis)

        # --------------------------------------------------------------
        # Raw ELM basis at the training points and its z-derivatives
        # --------------------------------------------------------------
        h, h_d, h_dd = _tanh_basis(self.z, self._weights, self._biases)

        # --------------------------------------------------------------
        # Hermite switching functions on [-1, 1]
        # --------------------------------------------------------------
        (
            (om1, om2, om3, om4),
            (om1d, om2d, om3d, om4d),
            (om1dd, om2dd, om3dd, om4dd),
        ) = _hermite_switching(self.z, z0, zf)

        # Endpoint rows of the ELM basis needed by the constrained
        # expression: h(z0), h(zf), h'(z0), h'(zf).
        h0 = h[0, :]
        hf = h[-1, :]
        hd0 = h_d[0, :]
        hdf = h_d[-1, :]

        # --------------------------------------------------------------
        # Constrained expression matrix F and physical-time derivatives
        # --------------------------------------------------------------
        # F = h - om1 h(z0) - om2 h(zf) - om3 h'(z0) - om4 h'(zf)
        # F'(t)  = c  * (dF/dz);  F''(t) = c^2 * (d^2F/dz^2)
        self.F = (
            h
            - om1[:, None] * h0[None, :]
            - om2[:, None] * hf[None, :]
            - om3[:, None] * hd0[None, :]
            - om4[:, None] * hdf[None, :]
        )
        self.Fd = c * (
            h_d
            - om1d[:, None] * h0[None, :]
            - om2d[:, None] * hf[None, :]
            - om3d[:, None] * hd0[None, :]
            - om4d[:, None] * hdf[None, :]
        )
        self.Fdd = c2 * (
            h_dd
            - om1dd[:, None] * h0[None, :]
            - om2dd[:, None] * hf[None, :]
            - om3dd[:, None] * hd0[None, :]
            - om4dd[:, None] * hdf[None, :]
        )

        # Costate unknowns use the raw ELM basis (no boundary conditions).
        self.H = h
        self.Hd = c * h_d

        # --------------------------------------------------------------
        # BC-only Hermite interpolant for each position coordinate and
        # its first two physical-time derivatives. For a coordinate with
        # endpoint values (q0, qf) and endpoint time-derivatives (p0, pf):
        #
        #   C(z) = om1 q0 + om2 qf + (1/c) (om3 p0 + om4 pf)
        #
        # The 1/c on the derivative terms is the change-of-variables
        # scale d/dt = c d/dz.
        # --------------------------------------------------------------
        x0, y0 = self.r0
        xf, yf = self.rf
        vx0, vy0 = self.v0
        vxf, vyf = self.vf
        inv_c = 1.0 / c

        self.Cx = om1 * x0 + om2 * xf + inv_c * (om3 * vx0 + om4 * vxf)
        self.Cy = om1 * y0 + om2 * yf + inv_c * (om3 * vy0 + om4 * vyf)
        self.Cxd = c * (
            om1d * x0 + om2d * xf + inv_c * (om3d * vx0 + om4d * vxf)
        )
        self.Cyd = c * (
            om1d * y0 + om2d * yf + inv_c * (om3d * vy0 + om4d * vyf)
        )
        self.Cxdd = c2 * (
            om1dd * x0 + om2dd * xf + inv_c * (om3dd * vx0 + om4dd * vxf)
        )
        self.Cydd = c2 * (
            om1dd * y0 + om2dd * yf + inv_c * (om3dd * vy0 + om4dd * vyf)
        )

    # ------------------------------------------------------------------
    # Convenience sizes
    # ------------------------------------------------------------------

    @property
    def n_training(self) -> int:
        return int(self.config.n_training)

    @property
    def n_basis(self) -> int:
        return int(self.config.n_basis)

    @property
    def n_equations(self) -> int:
        """Number of scalar equations in the linear system (``6 n``)."""
        return 6 * self.n_training

    @property
    def n_unknowns(self) -> int:
        """Number of scalar unknowns in the linear system (``6 L``)."""
        return 6 * self.n_basis

    # ------------------------------------------------------------------
    # Initial guess
    # ------------------------------------------------------------------

    def initial_nominal(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Cheap first-iteration nominal: the BC-only Hermite interpolant.

        Equivalent to decoding ``xi = 0``: the position fields reduce to
        ``x = Cx``, ``y = Cy``. The resulting curve satisfies the four
        endpoint boundary conditions on ``(r, v)`` exactly but ignores
        the CR3BP dynamics; it is a convenient linearization seed for
        the outer iteration.
        """
        return self.Cx.copy(), self.Cy.copy()

    # ------------------------------------------------------------------
    # Linear system assembly
    # ------------------------------------------------------------------

    def build_linear_system(
        self,
        nominal_x: NDArray[np.float64],
        nominal_y: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Assemble the linearized indirect-optimality linear system.

        Parameters
        ----------
        nominal_x, nominal_y : (n,) ndarray
            Nominal trajectory (planar position) at the training
            points, around which the synodic pseudo-potential gradient
            and Hessian are evaluated.

        Returns
        -------
        A : (6 n, 6 L) ndarray
            Stacked block rows for the two position dynamics equations
            and the four costate dynamics equations. The unknown vector
            is blocked as
            ``xi = (xi_x, xi_y, xi_lrx, xi_lry, xi_lvx, xi_lvy)``.
        B : (6 n,) ndarray
            Right-hand side. The first ``2 n`` rows carry the
            linearization forcing and BC-residual terms; the remaining
            ``4 n`` rows are zero because the costate ODEs are
            homogeneous.

        Notes
        -----
        The six block rows encode, in order:

        1. ``d^2 x/dt^2 = 2 dy/dt + Omega_x + u_x`` with ``u = -lambda_v``
           and ``Omega_x`` linearized at the nominal.
        2. ``d^2 y/dt^2 = -2 dx/dt + Omega_y + u_y`` (same treatment).
        3. ``d lambda_rx/dt = -lambda_vx Omega_xx - lambda_vy Omega_xy``
           with ``Omega_xx, Omega_xy`` frozen at the nominal.
        4. ``d lambda_ry/dt = -lambda_vx Omega_xy - lambda_vy Omega_yy``.
        5. ``d lambda_vx/dt = -lambda_rx + 2 lambda_vy`` (Coriolis).
        6. ``d lambda_vy/dt = -lambda_ry - 2 lambda_vx``.
        """
        nominal_x = np.asarray(nominal_x, dtype=np.float64)
        nominal_y = np.asarray(nominal_y, dtype=np.float64)
        n = self.n_training
        if nominal_x.shape != (n,) or nominal_y.shape != (n,):
            raise ValueError(
                f"nominal_x and nominal_y must be 1D arrays of length n={n}, "
                f"got shapes {nominal_x.shape} and {nominal_y.shape}"
            )

        mu = self.dynamics.mu
        omx, omy, omxx, omxy, omyy = _omega_grad_hess(mu, nominal_x, nominal_y)

        F = self.F
        Fd = self.Fd
        Fdd = self.Fdd
        H = self.H
        Hd = self.Hd
        zero = np.zeros_like(F)

        def scale(w: NDArray[np.float64], M: NDArray[np.float64]) -> NDArray[np.float64]:
            """Row-scale a (n, L) matrix by an (n,) vector."""
            return w[:, None] * M

        # Row block 1: d^2 x/dt^2 - 2 dy/dt - Omega_xx x - Omega_xy y + lambda_vx = ...
        row_ddx = np.hstack(
            [
                Fdd - scale(omxx, F),     # xi_x
                -2.0 * Fd - scale(omxy, F),  # xi_y
                zero,                      # xi_lrx
                zero,                      # xi_lry
                H,                         # xi_lvx  (u_x = -lambda_vx)
                zero,                      # xi_lvy
            ]
        )

        # Row block 2: d^2 y/dt^2 + 2 dx/dt - Omega_xy x - Omega_yy y + lambda_vy = ...
        row_ddy = np.hstack(
            [
                2.0 * Fd - scale(omxy, F),   # xi_x
                Fdd - scale(omyy, F),        # xi_y
                zero,                        # xi_lrx
                zero,                        # xi_lry
                zero,                        # xi_lvx
                H,                           # xi_lvy
            ]
        )

        # Row block 3: d lambda_rx/dt + Omega_xx lambda_vx + Omega_xy lambda_vy = 0
        row_dlrx = np.hstack(
            [
                zero,
                zero,
                Hd,                         # xi_lrx
                zero,                       # xi_lry
                scale(omxx, H),             # xi_lvx
                scale(omxy, H),             # xi_lvy
            ]
        )

        # Row block 4: d lambda_ry/dt + Omega_xy lambda_vx + Omega_yy lambda_vy = 0
        row_dlry = np.hstack(
            [
                zero,
                zero,
                zero,                       # xi_lrx
                Hd,                         # xi_lry
                scale(omxy, H),             # xi_lvx
                scale(omyy, H),             # xi_lvy
            ]
        )

        # Row block 5: d lambda_vx/dt + lambda_rx - 2 lambda_vy = 0
        row_dlvx = np.hstack(
            [
                zero,
                zero,
                H,                          # xi_lrx
                zero,                       # xi_lry
                Hd,                         # xi_lvx
                -2.0 * H,                   # xi_lvy
            ]
        )

        # Row block 6: d lambda_vy/dt + lambda_ry + 2 lambda_vx = 0
        row_dlvy = np.hstack(
            [
                zero,
                zero,
                zero,                       # xi_lrx
                H,                          # xi_lry
                2.0 * H,                    # xi_lvx
                Hd,                         # xi_lvy
            ]
        )

        A = np.vstack([row_ddx, row_ddy, row_dlrx, row_dlry, row_dlvx, row_dlvy])

        # ------------------------------------------------------------------
        # Right-hand side. Position rows absorb the residual of the BC-only
        # Hermite interpolant against the linearized dynamics. Costate rows
        # are homogeneous because the Hamiltonian is quadratic in the
        # costates and the linearization has no constant term in lambda.
        # ------------------------------------------------------------------
        Cx = self.Cx
        Cy = self.Cy
        Cxd = self.Cxd
        Cyd = self.Cyd
        Cxdd = self.Cxdd
        Cydd = self.Cydd

        B_ddx = (
            -Cxdd
            + 2.0 * Cyd
            + omx
            + omxx * (Cx - nominal_x)
            + omxy * (Cy - nominal_y)
        )
        B_ddy = (
            -Cydd
            - 2.0 * Cxd
            + omy
            + omxy * (Cx - nominal_x)
            + omyy * (Cy - nominal_y)
        )
        zero_n = np.zeros(n)
        B = np.concatenate([B_ddx, B_ddy, zero_n, zero_n, zero_n, zero_n])

        return A, B

    # ------------------------------------------------------------------
    # Trajectory decoding
    # ------------------------------------------------------------------

    def _split_xi(
        self, xi: NDArray[np.float64]
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        xi = np.asarray(xi, dtype=np.float64)
        if xi.shape != (self.n_unknowns,):
            raise ValueError(
                f"xi must be a length-{self.n_unknowns} vector, got shape {xi.shape}"
            )
        L = self.n_basis
        return (
            xi[0 * L : 1 * L],
            xi[1 * L : 2 * L],
            xi[2 * L : 3 * L],
            xi[3 * L : 4 * L],
            xi[4 * L : 5 * L],
            xi[5 * L : 6 * L],
        )

    def decode_trajectory(
        self, xi: NDArray[np.float64]
    ) -> dict[str, NDArray[np.float64]]:
        """Reconstruct the state, costates and control at the training points.

        Parameters
        ----------
        xi : (6 L,) ndarray
            Coefficient vector in the block order
            ``(xi_x, xi_y, xi_lrx, xi_lry, xi_lvx, xi_lvy)``.

        Returns
        -------
        dict of (n,) ndarrays
            Keys: ``x, y, vx, vy, lambda_rx, lambda_ry, lambda_vx,
            lambda_vy, ux, uy``. The control is the Pontryagin optimum
            ``u = -lambda_v``.
        """
        xi_x, xi_y, xi_lrx, xi_lry, xi_lvx, xi_lvy = self._split_xi(xi)
        x = self.F @ xi_x + self.Cx
        y = self.F @ xi_y + self.Cy
        vx = self.Fd @ xi_x + self.Cxd
        vy = self.Fd @ xi_y + self.Cyd
        lambda_rx = self.H @ xi_lrx
        lambda_ry = self.H @ xi_lry
        lambda_vx = self.H @ xi_lvx
        lambda_vy = self.H @ xi_lvy
        return {
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy,
            "lambda_rx": lambda_rx,
            "lambda_ry": lambda_ry,
            "lambda_vx": lambda_vx,
            "lambda_vy": lambda_vy,
            "ux": -lambda_vx,
            "uy": -lambda_vy,
        }

    # ------------------------------------------------------------------
    # Nonlinear residual and sequential linearization outer loop
    # ------------------------------------------------------------------

    def nonlinear_residual(
        self, xi: NDArray[np.float64]
    ) -> dict[str, NDArray[np.float64]]:
        """Residuals of the decoded trajectory against the FULL nonlinear TPBVP.

        Unlike the linearized residual ``A xi - B`` (which is driven to
        essentially zero at every iteration of the classical lsq solve),
        this residual evaluates the true nonlinear optimality conditions
        at the decoded trajectory. The gradient/Hessian of Omega are
        recomputed at ``(x(t), y(t))`` itself instead of at the nominal.
        It is therefore zero only when the sequential linearization
        outer loop has actually converged to a solution of the true
        CR3BP TPBVP.

        Returns
        -------
        dict of (n,) ndarrays
            Six entries, one per equation, each evaluated at every
            training point:

            * ``ddx``: ``d^2 x/dt^2 - 2 dy/dt - Omega_x(x, y) + lambda_vx``
            * ``ddy``: ``d^2 y/dt^2 + 2 dx/dt - Omega_y(x, y) + lambda_vy``
            * ``dlrx``: ``d lambda_rx/dt + Omega_xx(x, y) lambda_vx + Omega_xy(x, y) lambda_vy``
            * ``dlry``: ``d lambda_ry/dt + Omega_xy(x, y) lambda_vx + Omega_yy(x, y) lambda_vy``
            * ``dlvx``: ``d lambda_vx/dt + lambda_rx - 2 lambda_vy``
            * ``dlvy``: ``d lambda_vy/dt + lambda_ry + 2 lambda_vx``

            The ``dlvx`` and ``dlvy`` rows are independent of the
            Omega nonlinearity, so they should already be small from
            the first outer iteration; the other four rows shrink as
            the nominal converges.
        """
        xi_x, xi_y, xi_lrx, xi_lry, xi_lvx, xi_lvy = self._split_xi(xi)
        traj = self.decode_trajectory(xi)
        mu = self.dynamics.mu
        omx, omy, omxx, omxy, omyy = _omega_grad_hess(mu, traj["x"], traj["y"])

        ddx = self.Fdd @ xi_x + self.Cxdd
        ddy = self.Fdd @ xi_y + self.Cydd
        dlrx = self.Hd @ xi_lrx
        dlry = self.Hd @ xi_lry
        dlvx = self.Hd @ xi_lvx
        dlvy = self.Hd @ xi_lvy

        return {
            "ddx": ddx - 2.0 * traj["vy"] - omx + traj["lambda_vx"],
            "ddy": ddy + 2.0 * traj["vx"] - omy + traj["lambda_vy"],
            "dlrx": dlrx + omxx * traj["lambda_vx"] + omxy * traj["lambda_vy"],
            "dlry": dlry + omxy * traj["lambda_vx"] + omyy * traj["lambda_vy"],
            "dlvx": dlvx + traj["lambda_rx"] - 2.0 * traj["lambda_vy"],
            "dlvy": dlvy + traj["lambda_ry"] + 2.0 * traj["lambda_vx"],
        }

    def _solve_and_evaluate(
        self,
        nominal_x: NDArray[np.float64],
        nominal_y: NDArray[np.float64],
    ) -> tuple[
        NDArray[np.float64],
        dict[str, NDArray[np.float64]],
        float,
        float,
    ]:
        """One linearized solve at a nominal, plus decoded diagnostics.

        Returns ``(xi, trajectory, linearized_residual, nonlinear_residual)``
        where ``linearized_residual`` is ``||A xi - B||_2`` and
        ``nonlinear_residual`` is the infinity-norm of the full
        nonlinear TPBVP residual at the decoded trajectory. Factored
        out of :meth:`solve_sequential` so the line-search inner loop
        can reuse it.
        """
        A, B = self.build_linear_system(nominal_x, nominal_y)
        xi, *_ = np.linalg.lstsq(A, B, rcond=None)
        traj = self.decode_trajectory(xi)
        lin_res = float(np.linalg.norm(A @ xi - B))
        nl_dict = self.nonlinear_residual(xi)
        nl_res = float(max(np.max(np.abs(r)) for r in nl_dict.values()))
        return xi, traj, lin_res, nl_res

    def solve_sequential(
        self,
        outer_config: OuterLoopConfig | None = None,
        initial_nominal: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
    ) -> SequentialSolution:
        """Run the sequential linearization outer loop.

        At each outer iteration we

        1. measure the fixed-point residual of the current
           ``(nominal, candidate)`` pair and check the convergence
           tolerance,
        2. pick a damping factor ``alpha`` — either a fixed
           ``cfg.damping`` (when ``line_search=False``) or the output
           of a backtracking Armijo line search on the nonlinear TPBVP
           residual,
        3. update the nominal to ``nominal + alpha * (candidate - nominal)``,
           and
        4. re-solve the linearized system at the new nominal to prime
           the next iteration.

        The line search evaluates each trial ``alpha`` by fully
        re-solving at the trial nominal and comparing the resulting
        fixed-point residual ``||traj(xi(n_trial)) - n_trial||_inf``
        against the residual at the previous nominal. That residual
        is the natural Newton merit for the fixed-point rootfinding
        problem ``F(n) = decoded(xi(n)) - n = 0``, and is exactly the
        quantity tracked in ``history["trajectory_change_inf"]``, so
        if the line search is working the history will be (weakly)
        monotone decreasing in that column. The cost is one extra
        linearized solve per backtrack, which is cheap compared to a
        QUBO sample.

        Parameters
        ----------
        outer_config : OuterLoopConfig, optional
            Loop hyperparameters. Defaults to ``OuterLoopConfig()``.
        initial_nominal : (x_bar, y_bar) tuple of (n,) ndarrays, optional
            Starting nominal trajectory at the training points.
            Defaults to :meth:`initial_nominal` (the BC-only Hermite
            interpolant, equivalent to ``xi = 0``).

        Returns
        -------
        SequentialSolution
            Final coefficients, decoded trajectory, convergence flag
            and per-iteration diagnostics.
        """
        cfg = outer_config if outer_config is not None else OuterLoopConfig()
        if not (0.0 < cfg.damping <= 1.0):
            raise ValueError(
                f"damping must be in the half-open interval (0, 1], got {cfg.damping}"
            )
        if cfg.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {cfg.max_iter}")
        if cfg.tol <= 0.0:
            raise ValueError(f"tol must be > 0, got {cfg.tol}")
        if cfg.line_search:
            if not (0.0 < cfg.min_damping <= cfg.damping):
                raise ValueError(
                    f"min_damping must be in (0, damping={cfg.damping}], "
                    f"got {cfg.min_damping}"
                )
            if not (0.0 < cfg.backtrack_factor < 1.0):
                raise ValueError(
                    f"backtrack_factor must be in (0, 1), got {cfg.backtrack_factor}"
                )
            if not (0.0 <= cfg.armijo_c < 1.0):
                raise ValueError(
                    f"armijo_c must be in [0, 1), got {cfg.armijo_c}"
                )

        if initial_nominal is None:
            nominal_x, nominal_y = self.initial_nominal()
        else:
            nominal_x = np.asarray(initial_nominal[0], dtype=np.float64).copy()
            nominal_y = np.asarray(initial_nominal[1], dtype=np.float64).copy()
            n = self.n_training
            if nominal_x.shape != (n,) or nominal_y.shape != (n,):
                raise ValueError(
                    f"initial_nominal arrays must have shape ({n},), got "
                    f"{nominal_x.shape} and {nominal_y.shape}"
                )

        history: dict[str, list[float]] = {
            "trajectory_change_inf": [],
            "linearized_residual": [],
            "nonlinear_residual": [],
            "step_size": [],
        }
        converged = False

        # Initial solve at the starting nominal primes xi/traj/residuals.
        xi, traj, lin_res, nl_res = self._solve_and_evaluate(nominal_x, nominal_y)

        # Safeguard for the Armijo comparison: if the current step
        # residual is at or below this threshold we accept any finite
        # trial because (1 - c*alpha) * ~0 is unachievable in floating
        # point. This only triggers at the convergence boundary.
        step_accept_floor = 1e-14

        def _step_norm(
            trajectory: dict[str, NDArray[np.float64]],
            nx: NDArray[np.float64],
            ny: NDArray[np.float64],
        ) -> float:
            dx_ = trajectory["x"] - nx
            dy_ = trajectory["y"] - ny
            return float(max(np.max(np.abs(dx_)), np.max(np.abs(dy_))))

        for _ in range(cfg.max_iter):
            # Fixed-point residual at the current nominal -- this is
            # what drives both the convergence test and the line
            # search merit comparison.
            dx = traj["x"] - nominal_x
            dy = traj["y"] - nominal_y
            step_norm = float(max(np.max(np.abs(dx)), np.max(np.abs(dy))))

            history["trajectory_change_inf"].append(step_norm)
            history["linearized_residual"].append(lin_res)
            history["nonlinear_residual"].append(nl_res)

            if step_norm < cfg.tol:
                converged = True
                # No step is taken on a converged iteration; record 0
                # so all four history lists stay the same length.
                history["step_size"].append(0.0)
                break

            if not cfg.line_search:
                # Pass-1 behavior: fixed damping, one solve per iter.
                alpha = cfg.damping
                nominal_x = nominal_x + alpha * dx
                nominal_y = nominal_y + alpha * dy
                xi, traj, lin_res, nl_res = self._solve_and_evaluate(
                    nominal_x, nominal_y
                )
                history["step_size"].append(alpha)
                continue

            # Backtracking line search: start at cfg.damping, shrink
            # by cfg.backtrack_factor until the Armijo condition on
            # the fixed-point residual is satisfied or alpha falls to
            # cfg.min_damping. Merit function:
            #   m(alpha) = ||traj(xi(n + alpha*dn)) - (n + alpha*dn)||_inf
            # which is identical to the quantity measured at the top
            # of the next iteration, so if the line search accepts
            # then the next history["trajectory_change_inf"] entry is
            # guaranteed to decrease relative to the current one.
            alpha = cfg.damping
            step_prev = step_norm
            trial_state: tuple[
                NDArray[np.float64],
                NDArray[np.float64],
                NDArray[np.float64],
                dict[str, NDArray[np.float64]],
                float,
                float,
            ] | None = None
            while True:
                trial_x = nominal_x + alpha * dx
                trial_y = nominal_y + alpha * dy
                xi_t, traj_t, lin_t, nl_t = self._solve_and_evaluate(
                    trial_x, trial_y
                )
                trial_state = (trial_x, trial_y, xi_t, traj_t, lin_t, nl_t)
                step_trial = _step_norm(traj_t, trial_x, trial_y)

                threshold = (1.0 - cfg.armijo_c * alpha) * step_prev
                if np.isfinite(step_trial) and (
                    step_prev <= step_accept_floor or step_trial <= threshold
                ):
                    break
                if alpha <= cfg.min_damping + 1e-15:
                    # Floor reached: commit the last trial anyway so
                    # the loop can either make partial progress or
                    # report non-convergence upstream.
                    break
                alpha = max(alpha * cfg.backtrack_factor, cfg.min_damping)

            assert trial_state is not None  # loop always runs at least once
            nominal_x, nominal_y, xi, traj, lin_res, nl_res = trial_state
            history["step_size"].append(alpha)

        return SequentialSolution(
            xi=xi,
            trajectory=traj,
            converged=converged,
            n_iterations=len(history["trajectory_change_inf"]),
            history=history,
        )
