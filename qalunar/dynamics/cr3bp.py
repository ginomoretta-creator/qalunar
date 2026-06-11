"""Planar Circular Restricted Three-Body Problem (CR3BP) in the synodic frame.

Nondimensional units, following the standard convention:

* Distance unit L = semi-major axis of the secondary's orbit about the primary
  (Earth-Moon: L ~ 384,400 km).
* Time unit T = 1 / n, where n is the mean motion of the two primaries
  (Earth-Moon: T ~ 4.343 days, so 1 nondim time unit ~= 375,700 s).
* Mass unit = m_1 + m_2, with mass parameter mu = m_2 / (m_1 + m_2).

In the rotating (synodic) frame, the primaries are stationary on the x-axis
at Earth = (-mu, 0) and Moon = (1 - mu, 0). The state vector is
[x, y, vx, vy] and the control input is the thrust acceleration [u_x, u_y]
expressed in synodic Cartesian components.

Equations of motion::

    dx/dt  = vx
    dy/dt  = vy
    dvx/dt = 2*vy + Omega_x(x, y) + u_x
    dvy/dt = -2*vx + Omega_y(x, y) + u_y

where the gradient of the synodic pseudo-potential is::

    Omega_x = x - (1 - mu) * (x + mu)   / r1**3 - mu * (x - 1 + mu) / r2**3
    Omega_y = y - (1 - mu) *  y         / r1**3 - mu *  y           / r2**3

with distances from each primary::

    r1 = sqrt((x + mu)**2 + y**2)          (from Earth)
    r2 = sqrt((x - 1 + mu)**2 + y**2)      (from Moon)

The pseudo-potential itself is
Omega = (x**2 + y**2) / 2 + (1 - mu) / r1 + mu / r2 + mu * (1 - mu) / 2,
and the Jacobi integral C_J = 2 * Omega - (vx**2 + vy**2) is conserved
along unforced trajectories.

References
----------
* Szebehely, *Theory of Orbits*, 1967, Sections 3-4.
* Koon, Lo, Marsden, Ross, *Dynamical Systems, the Three-Body Problem and
  Space Mission Design*, 2011, Chapter 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


# Earth-Moon mass parameter mu = m_moon / (m_earth + m_moon).
# Value consistent with JPL DE-series ephemerides (Koon et al. 2011, p.17).
EARTH_MOON_MU: float = 0.012150583925359


@dataclass(frozen=True)
class PlanarCR3BP:
    """Planar CR3BP dynamics with a 2D thrust-acceleration control.

    The control enters the equations of motion linearly, so the control
    Jacobian is constant and independent of the state.

    Parameters
    ----------
    mu : float
        Mass parameter m_2 / (m_1 + m_2). Defaults to the Earth-Moon value.
    """

    mu: float = EARTH_MOON_MU

    @property
    def n_state(self) -> int:
        return 4

    @property
    def n_control(self) -> int:
        return 2

    def earth_position(self) -> NDArray[np.float64]:
        """Location of the primary (larger body) in synodic coordinates."""
        return np.array([-self.mu, 0.0])

    def moon_position(self) -> NDArray[np.float64]:
        """Location of the secondary (smaller body) in synodic coordinates."""
        return np.array([1.0 - self.mu, 0.0])

    def _distances(self, x: float, y: float) -> tuple[float, float]:
        r1 = float(np.sqrt((x + self.mu) ** 2 + y ** 2))
        r2 = float(np.sqrt((x - 1.0 + self.mu) ** 2 + y ** 2))
        return r1, r2

    def rhs(
        self,
        state: NDArray[np.float64],
        control: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Time derivative of the state under the planar CR3BP with control.

        Parameters
        ----------
        state : (4,) ndarray
            ``[x, y, vx, vy]`` in synodic nondimensional coordinates.
        control : (2,) ndarray, optional
            Thrust acceleration ``[u_x, u_y]``. If ``None``, the unforced
            CR3BP is evaluated (useful for validating the Jacobi integral).

        Returns
        -------
        (4,) ndarray
            ``[vx, vy, ax, ay]``.
        """
        x, y, vx, vy = state
        r1, r2 = self._distances(x, y)

        one_minus_mu = 1.0 - self.mu
        omega_x = (
            x
            - one_minus_mu * (x + self.mu) / r1 ** 3
            - self.mu * (x - one_minus_mu) / r2 ** 3
        )
        omega_y = y - one_minus_mu * y / r1 ** 3 - self.mu * y / r2 ** 3

        ax = 2.0 * vy + omega_x
        ay = -2.0 * vx + omega_y
        if control is not None:
            ax = ax + control[0]
            ay = ay + control[1]

        return np.array([vx, vy, ax, ay])

    def jacobian_state(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Analytical state Jacobian df/dx, shape (4, 4).

        The velocity rows are trivial (identity coupling to the velocity
        block). The acceleration rows contain the second derivatives of the
        synodic pseudo-potential plus the constant Coriolis coupling.
        """
        x, y, _, _ = state
        r1, r2 = self._distances(x, y)
        one_minus_mu = 1.0 - self.mu

        r1_3 = r1 ** 3
        r2_3 = r2 ** 3
        r1_5 = r1 ** 5
        r2_5 = r2 ** 5

        omega_xx = (
            1.0
            - one_minus_mu / r1_3
            - self.mu / r2_3
            + 3.0 * one_minus_mu * (x + self.mu) ** 2 / r1_5
            + 3.0 * self.mu * (x - one_minus_mu) ** 2 / r2_5
        )
        omega_yy = (
            1.0
            - one_minus_mu / r1_3
            - self.mu / r2_3
            + 3.0 * one_minus_mu * y ** 2 / r1_5
            + 3.0 * self.mu * y ** 2 / r2_5
        )
        omega_xy = (
            3.0 * one_minus_mu * (x + self.mu) * y / r1_5
            + 3.0 * self.mu * (x - one_minus_mu) * y / r2_5
        )

        return np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [omega_xx, omega_xy, 0.0, 2.0],
                [omega_xy, omega_yy, -2.0, 0.0],
            ]
        )

    def jacobian_control(
        self, state: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64]:
        """Analytical control Jacobian df/du, shape (4, 2).

        Constant and state-independent because the control enters the
        acceleration rows linearly. The ``state`` argument is accepted for
        interface consistency with nonlinear-in-control dynamics models.
        """
        del state
        return np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )

    def jacobi_integral(self, state: NDArray[np.float64]) -> float:
        """Jacobi integral C_J = 2*Omega - v**2.

        Conserved along unforced CR3BP trajectories. A drifting Jacobi
        integral under a pure-drift integration is an unambiguous signal
        of a bug in the RHS or the integrator step.
        """
        x, y, vx, vy = state
        r1, r2 = self._distances(x, y)
        two_omega = (
            x ** 2
            + y ** 2
            + 2.0 * (1.0 - self.mu) / r1
            + 2.0 * self.mu / r2
            + self.mu * (1.0 - self.mu)
        )
        return float(two_omega - vx ** 2 - vy ** 2)

    def lagrange_triangular_points(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the planar positions of L4 and L5 (equilateral points).

        L4 and L5 sit at the apex of the equilateral triangles formed with
        the two primaries: ``(1/2 - mu, +/- sqrt(3)/2)``.
        """
        half_sqrt3 = np.sqrt(3.0) / 2.0
        l4 = np.array([0.5 - self.mu, half_sqrt3])
        l5 = np.array([0.5 - self.mu, -half_sqrt3])
        return l4, l5

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------

    def propagate(
        self,
        state0: NDArray[np.float64],
        t_span: tuple[float, float],
        n_steps: int = 1000,
        control: NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Propagate the state forward using fixed-step RK4.

        Parameters
        ----------
        state0 : (4,) ndarray
            Initial state ``[x, y, vx, vy]``.
        t_span : (t0, tf) tuple
            Start and end time.
        n_steps : int
            Number of RK4 steps.
        control : (2,) ndarray or None
            Constant thrust acceleration applied throughout. ``None``
            for unforced propagation.

        Returns
        -------
        t : (n_steps + 1,) ndarray
            Time grid.
        states : (n_steps + 1, 4) ndarray
            State at each time.
        """
        t0, tf = t_span
        dt = (tf - t0) / n_steps
        t = np.linspace(t0, tf, n_steps + 1)
        states = np.empty((n_steps + 1, 4))
        states[0] = state0

        x = np.array(state0, dtype=np.float64)
        for i in range(n_steps):
            k1 = self.rhs(x, control)
            k2 = self.rhs(x + 0.5 * dt * k1, control)
            k3 = self.rhs(x + 0.5 * dt * k2, control)
            k4 = self.rhs(x + dt * k3, control)
            x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            states[i + 1] = x

        return t, states

    def propagate_stm(
        self,
        state0: NDArray[np.float64],
        t_span: tuple[float, float],
        n_steps: int = 1000,
        control: NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Propagate state and state transition matrix using fixed-step RK4.

        Integrates the augmented system ``[state (4), Phi (4x4)]`` where
        ``dPhi/dt = A(t) Phi`` with ``A = df/dx`` evaluated along the
        trajectory and ``Phi(t0) = I_4``.

        Parameters
        ----------
        state0 : (4,) ndarray
            Initial state ``[x, y, vx, vy]``.
        t_span : (t0, tf) tuple
            Start and end time.
        n_steps : int
            Number of RK4 steps.
        control : (2,) ndarray or None
            Constant thrust acceleration applied throughout.

        Returns
        -------
        t : (n_steps + 1,) ndarray
            Time grid.
        states : (n_steps + 1, 4) ndarray
            State at each time.
        stms : (n_steps + 1, 4, 4) ndarray
            ``stms[i]`` is ``Phi(t[i], t[0])``, the STM from the
            initial time to ``t[i]``. ``stms[0] = I_4``.
        """
        t0, tf = t_span
        dt = (tf - t0) / n_steps
        t = np.linspace(t0, tf, n_steps + 1)
        states = np.empty((n_steps + 1, 4))
        stms = np.empty((n_steps + 1, 4, 4))

        states[0] = state0
        stms[0] = np.eye(4)

        # Augmented state: 4 (state) + 16 (Phi flattened) = 20 components
        aug = np.zeros(20)
        aug[:4] = state0
        aug[4:] = np.eye(4).ravel()

        def aug_rhs(z: NDArray[np.float64]) -> NDArray[np.float64]:
            s = z[:4]
            phi = z[4:].reshape(4, 4)
            ds = self.rhs(s, control)
            A = self.jacobian_state(s)
            dphi = (A @ phi).ravel()
            return np.concatenate([ds, dphi])

        for i in range(n_steps):
            k1 = aug_rhs(aug)
            k2 = aug_rhs(aug + 0.5 * dt * k1)
            k3 = aug_rhs(aug + 0.5 * dt * k2)
            k4 = aug_rhs(aug + dt * k3)
            aug = aug + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            states[i + 1] = aug[:4]
            stms[i + 1] = aug[4:].reshape(4, 4)

        return t, states, stms

    def propagate_stm_piecewise(
        self,
        state0: NDArray[np.float64],
        t_span: tuple[float, float],
        controls: list[NDArray[np.float64] | None],
        n_substeps: int = 50,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Propagate state and STM under a piecewise-constant control schedule.

        The interval ``[t0, tf]`` is divided into ``len(controls)`` equal
        subintervals. Within subinterval ``i``, the constant control
        ``controls[i]`` is applied (``None`` means coast). State and the
        STM ``Phi(t, t0)`` are integrated jointly with fixed-step RK4
        using ``n_substeps`` per subinterval, so the variational equation
        is evaluated at the actual thrust-applied state.

        Parameters
        ----------
        state0 : (4,) ndarray
            Initial state.
        t_span : (t0, tf)
            Total time interval.
        controls : list of (2,) ndarray or None
            One control vector per subinterval; ``None`` for coast.
        n_substeps : int
            RK4 substeps per subinterval.

        Returns
        -------
        t : (n_subs * n_substeps + 1,) ndarray
            Full time grid.
        states : (n_subs * n_substeps + 1, 4) ndarray
        stms : (n_subs * n_substeps + 1, 4, 4) ndarray
            ``stms[k]`` is ``Phi(t[k], t0)``.
        """
        t0, tf = t_span
        n_subs = len(controls)
        if n_subs < 1:
            raise ValueError("controls must be non-empty")
        n_total = n_subs * n_substeps
        dt = (tf - t0) / n_total

        t = np.linspace(t0, tf, n_total + 1)
        states = np.empty((n_total + 1, 4))
        stms = np.empty((n_total + 1, 4, 4))
        states[0] = state0
        stms[0] = np.eye(4)

        aug = np.zeros(20)
        aug[:4] = state0
        aug[4:] = np.eye(4).ravel()

        for i_sub in range(n_subs):
            u = controls[i_sub]

            def aug_rhs(
                z: NDArray[np.float64], _u: NDArray[np.float64] | None = u
            ) -> NDArray[np.float64]:
                s = z[:4]
                phi = z[4:].reshape(4, 4)
                ds = self.rhs(s, _u)
                A = self.jacobian_state(s)
                dphi = (A @ phi).ravel()
                return np.concatenate([ds, dphi])

            for j in range(n_substeps):
                k1 = aug_rhs(aug)
                k2 = aug_rhs(aug + 0.5 * dt * k1)
                k3 = aug_rhs(aug + 0.5 * dt * k2)
                k4 = aug_rhs(aug + dt * k3)
                aug = aug + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                k_full = i_sub * n_substeps + j + 1
                states[k_full] = aug[:4]
                stms[k_full] = aug[4:].reshape(4, 4)

        return t, states, stms
