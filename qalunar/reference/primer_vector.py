"""Primer-vector reconstruction along a forced CR3BP trajectory.

Given a converged solution of the binary scheduling QUBO --- a
piecewise-constant on/off schedule and the resulting nonlinear
trajectory --- we can reconstruct a Pontryagin costate ``lambda(t)``
along that trajectory by backward-integrating the costate ODE from the
terminal transversality condition.  The tangential primer-vector
analogue is the scalar switching function

    S(t) = - lambda_v(t) . v_hat(t)

where ``lambda_v`` is the velocity sub-vector of the costate and
``v_hat`` is the unit velocity vector at time ``t``.  Pontryagin's
minimum principle predicts that the optimal tangential bang-bang
control fires the engine when ``S > 0`` and coasts when ``S < 0``.
Plotting ``S(t)`` together with the QUBO's discrete schedule
visualises the conjecture that the binary QUBO --- which never
computes a costate --- recovers the same switching pattern from the
discrete side.

Background
----------
For an objective ``J = phi(x_f)`` with no running cost (the
``fuel_weight = 0`` case used in the long-arc demonstration), the
Hamiltonian is ``H = lambda^T f(x, u)`` and the costate satisfies

    dot(lambda) = - A(x(t))^T lambda,

with terminal condition ``lambda(t_f) = grad_xf phi``.  For the
weighted boundary cost ``phi = (x_f - target)^T W (x_f - target)``
this is ``lambda(t_f) = 2 W (x_f - target)``.  The factor of 2 is
inconsequential for the *sign* of ``S``, which is what determines
the switching pattern.

The reconstructor expects a forward trajectory sampled densely enough
that the local Jacobian ``A(x(t)) = jacobian_state(x(t))`` can be
treated as piecewise-linear between adjacent samples; classical RK4
on the reversed time axis is used, with the Jacobian evaluated at the
sample states (and at midpoints by averaging).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from qalunar.dynamics.cr3bp import PlanarCR3BP


__all__ = [
    "PrimerVectorTrace",
    "reconstruct_primer_vector",
]


@dataclass(frozen=True)
class PrimerVectorTrace:
    """Result of :func:`reconstruct_primer_vector`.

    Attributes
    ----------
    times : (n,) ndarray
        Time grid (same as the input trajectory).
    lambdas : (n, 4) ndarray
        Costate ``[lambda_x, lambda_y, lambda_vx, lambda_vy]`` at each
        time, obtained by backward integration from the terminal
        transversality condition.
    switching : (n,) ndarray
        Tangential switching function ``S(t) = -lambda_v . v_hat``.
        Positive values predict ``ON`` is optimal under PMP, negative
        values predict ``OFF``.
    primer_norm : (n,) ndarray
        Unconstrained primer-vector magnitude ``|| -B^T lambda || =
        || lambda_v ||``. Returned for completeness; the tangential
        switching function ``switching`` is the relevant scalar when
        the engine is tangentially constrained.
    """

    times: NDArray[np.float64]
    lambdas: NDArray[np.float64]
    switching: NDArray[np.float64]
    primer_norm: NDArray[np.float64]


def reconstruct_primer_vector(
    dynamics: PlanarCR3BP,
    times: NDArray[np.float64],
    states: NDArray[np.float64],
    target_state: NDArray[np.float64],
    target_weights: NDArray[np.float64] | None = None,
) -> PrimerVectorTrace:
    """Backward-propagate the costate along a forward trajectory.

    Parameters
    ----------
    dynamics : PlanarCR3BP
        Dynamics model, used only for ``jacobian_state``.
    times : (n,) ndarray
        Strictly increasing time grid.
    states : (n, 4) ndarray
        State at each time.  Typically the output of
        :func:`qalunar.qubo.thrust_scheduling.propagate_schedule`
        with ``return_trajectory=True`` --- i.e. the actual nonlinear
        trajectory under the converged schedule.
    target_state : (4,) ndarray
        Target final state used to set the terminal transversality
        ``lambda(t_f) = 2 W (x_f - target)``.
    target_weights : (4,) ndarray, optional
        Diagonal of the weighting matrix ``W``. Defaults to identity.

    Returns
    -------
    PrimerVectorTrace
        Costate, switching function, and primer-vector norm at every
        sample time.
    """
    times = np.asarray(times, dtype=np.float64)
    states = np.asarray(states, dtype=np.float64)
    target_state = np.asarray(target_state, dtype=np.float64)

    if states.ndim != 2 or states.shape[1] != 4:
        raise ValueError(f"states must have shape (n, 4); got {states.shape}")
    if times.shape != (states.shape[0],):
        raise ValueError(
            f"times shape {times.shape} inconsistent with states "
            f"shape {states.shape}"
        )
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("times must be strictly increasing")

    if target_weights is None:
        W_diag = np.ones(4, dtype=np.float64)
    else:
        W_diag = np.asarray(target_weights, dtype=np.float64)
        if W_diag.shape != (4,):
            raise ValueError(
                f"target_weights must have shape (4,); got {W_diag.shape}"
            )

    n = states.shape[0]

    # Terminal transversality: lambda(t_f) = 2 W (x_f - target).
    # The factor of 2 is conventional and does not affect sign(S).
    miss_f = states[-1] - target_state
    lam_f = 2.0 * W_diag * miss_f

    # Backward integration via forward RK4 in reversed time.
    # In reversed time tau = t_f - t we have dlam/dtau = + A(x(tau))^T lam.
    states_rev = states[::-1]
    times_rev = times[::-1]
    dt_steps = -np.diff(times_rev)  # positive step sizes (decreasing t)

    lambdas_rev = np.empty_like(states)
    lambdas_rev[0] = lam_f
    lam = lam_f.copy()

    for k in range(n - 1):
        dt = float(dt_steps[k])
        x0 = states_rev[k]
        x1 = states_rev[k + 1]
        x_mid = 0.5 * (x0 + x1)

        A0 = dynamics.jacobian_state(x0)
        A_mid = dynamics.jacobian_state(x_mid)
        A1 = dynamics.jacobian_state(x1)

        k1 = A0.T @ lam
        k2 = A_mid.T @ (lam + 0.5 * dt * k1)
        k3 = A_mid.T @ (lam + 0.5 * dt * k2)
        k4 = A1.T @ (lam + dt * k3)
        lam = lam + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        lambdas_rev[k + 1] = lam

    # Reverse back to original time order.
    lambdas = lambdas_rev[::-1]

    # Switching function and primer norm at every time.
    velocities = states[:, 2:4]
    speeds = np.linalg.norm(velocities, axis=1)
    # Avoid division by zero at instants of zero velocity (degenerate
    # case; not expected for a moving spacecraft).
    safe_speeds = np.where(speeds > 1e-15, speeds, 1.0)
    v_hat = velocities / safe_speeds[:, None]
    lam_v = lambdas[:, 2:4]

    switching = -np.einsum("ij,ij->i", lam_v, v_hat)
    primer_norm = np.linalg.norm(lam_v, axis=1)

    return PrimerVectorTrace(
        times=times.copy(),
        lambdas=lambdas,
        switching=switching,
        primer_norm=primer_norm,
    )
