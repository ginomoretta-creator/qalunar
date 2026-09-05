"""QUBO formulation for low-thrust on/off scheduling in the CR3BP.

Given a reference trajectory and a target final state, decide at which
time steps to fire the engine to reach the target with minimum fuel.
The decision variables are naturally binary: ``q_i = 1`` means "burn at
step i", ``q_i = 0`` means "coast".

The effect of each burn on the final state is linearized via the state
transition matrix (STM):

    delta_state(tf) ≈ Σ_i (q_i - q_nom_i) · Phi(t_i, tf) · B · u_i · dt

where ``Phi(t_i, tf)`` maps the instantaneous thrust impulse at ``t_i``
forward to the final time, ``B`` is the control influence matrix
``[[0,0],[0,0],[1,0],[0,1]]``, and ``u_i`` is the thrust acceleration
vector at step ``i``. The default linearization point is the coast arc
(``q_nom = 0``); the iterative driver :func:`solve_iterative` re-builds
the QUBO around the current schedule to close the linearization gap.

The QUBO minimizes:

    f(q) = α Σ q_i  +  || Σ q_i b_i - d_eff ||_W²

where ``b_i = Phi(t_i, tf) · B · u_i · dt`` is the precomputed effect of
burning at step i, ``d_eff = (target - x_nom(tf)) + Σ q_nom_i b_i`` is
the effective state gap (reduces to ``target - coast_final`` when
``q_nom = 0``), ``W`` weights state components, and ``α`` is the fuel
penalty weight.

This is a *natural* QUBO — the variables are inherently binary, not
artificially quantized continuous variables.

Iterative re-linearization
--------------------------
For long arcs or strong thrust the single-pass linearization is loose.
:func:`solve_iterative` wraps the QUBO solve in a nonlinear outer loop:

    q^(0) := all coast
    repeat:
        build QUBO linearized around q^(k)
        q^(k+1) := sampler(QUBO)
        propagate true nonlinear dynamics under q^(k+1) → measure miss
    until schedule unchanged or true miss stops decreasing

This is the binary analogue of the continuous-side sequential
linearization in Carbone et al. 2025.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from qalunar.dynamics.cr3bp import PlanarCR3BP


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------


@dataclass(frozen=True)
class ThrustSchedulingConfig:
    """Configuration for the on/off scheduling QUBO.

    Parameters
    ----------
    thrust_magnitude : float
        Maximum thrust acceleration (nondimensional). Applied when
        ``q_i = 1``.
    thrust_direction : str
        How thrust direction is determined at each step (single-channel
        mode):

        * ``"tangential"`` — along the velocity vector
        * ``"antitangential"`` — opposite to velocity
        * ``"fixed"`` — constant direction given by ``thrust_vector``

        Ignored when ``thrust_channels`` is set.
    thrust_vector : (2,) ndarray or None
        Fixed thrust direction (only used when ``thrust_direction="fixed"``).
        Will be normalized and scaled by ``thrust_magnitude``.
    fuel_weight : float
        Penalty weight ``α`` for fuel usage (number of burns). Higher
        values prefer fewer burns at the cost of larger miss distance.
    target_weights : (4,) ndarray or None
        Per-component weighting for the target miss penalty. Defaults
        to ``[1, 1, 1, 1]``. Set velocity weights to 0 to target
        position only.
    thrust_channels : tuple[str, ...] or None
        Multi-channel mode.  Each entry is a direction string
        (``"tangential"``, ``"antitangential"``, ``"normal"``,
        ``"antinormal"``).  When set, the QUBO has
        ``len(thrust_channels) * N`` binary variables — one per
        (channel, step) pair.  Overrides ``thrust_direction``.
    exclusive : bool
        If ``True``, add a quadratic penalty so that at most one
        channel fires per step (single-engine constraint).
    exclusion_penalty : float or None
        Penalty weight ``P`` for the exclusion constraint. ``None``
        (default) auto-scales it to twice the largest single-bit energy
        swing of the physical objective -- the smallest value that makes
        every violating state strictly worse than a neighbouring feasible
        one, so the coefficient dynamic range stays as small as the
        physics allows (a hard-coded 10.0 was ~5e5 times the physics).
        The value actually used is stored on the returned QUBO.
    impulse_quadrature : {"midpoint", "left"}
        Where inside a decision interval the finite burn is collapsed to
        an impulse when forming ``b_j``. ``"midpoint"`` (default) is
        second-order in ``dt``; ``"left"`` reproduces the original
        left-endpoint rectangle rule, whose O(dt) bias survives the
        thrust-to-zero limit and was being charged to nonlinearity.
    """

    thrust_magnitude: float = 0.01
    thrust_direction: str = "tangential"
    thrust_vector: NDArray[np.float64] | None = None
    fuel_weight: float = 0.0
    target_weights: NDArray[np.float64] | None = None
    thrust_channels: tuple[str, ...] | None = None
    exclusive: bool = False
    exclusion_penalty: float | None = None
    impulse_quadrature: str = "midpoint"


# ------------------------------------------------------------------
# Result container
# ------------------------------------------------------------------


@dataclass
class ThrustSchedulingQubo:
    """Result of :func:`build_thrust_scheduling_qubo`.

    Attributes
    ----------
    Q : (M, M) ndarray
        Symmetric QUBO matrix.  ``f(q) = q^T Q q + linear^T q + const``.
        ``M = n_channels * n_steps``.
    linear : (M,) ndarray
        Linear coefficients.
    constant : float
        Energy offset (independent of q).
    n_steps : int
        Number of decision time steps.
    n_channels : int
        Number of thrust channels (1 for single-direction mode).
    channel_labels : tuple[str, ...]
        Human-readable name of each channel.
    b_vectors : (M, n_state) ndarray
        ``b_vectors[j]`` is the effect of variable ``j`` on the final
        state.  Variable ordering is channel-major:
        ``[ch0_step0, ch0_step1, ..., ch0_stepN-1, ch1_step0, ...]``.
    target_gap : (n_state,) ndarray
        Effective state gap ``d_eff`` used by the QUBO objective.
    fuel_weight : float
        The ``α`` used in the objective.
    t_grid : (N,) ndarray
        Time at each decision point.
    coast_trajectory : (N_full, 4) ndarray
        Reference trajectory (coast arc when nominal_schedule was None,
        otherwise the trajectory under the nominal schedule).
    thrust_magnitude : float
        Per-channel thrust acceleration (nondim) used to build
        ``b_vectors``. Stored so :func:`delta_v` can be evaluated
        without re-passing the config.
    dt_decision : float
        Duration of one decision interval (nondim).
    exclusion_penalty : float
        Exclusion penalty actually applied (0 when not exclusive).
    """

    Q: NDArray[np.float64]
    linear: NDArray[np.float64]
    constant: float
    n_steps: int
    n_channels: int
    channel_labels: tuple[str, ...]
    b_vectors: NDArray[np.float64]
    target_gap: NDArray[np.float64]
    fuel_weight: float
    t_grid: NDArray[np.float64]
    coast_trajectory: NDArray[np.float64]
    thrust_magnitude: float = 0.0
    dt_decision: float = 0.0
    exclusion_penalty: float = 0.0

    def coefficient_range(self) -> float:
        """max |coefficient| / min nonzero |coefficient| over the upper
        triangle of ``Q`` and ``linear`` -- the dynamic range an analog
        annealer (1-2% coefficient precision) must resolve."""
        coeffs = np.concatenate([
            np.abs(self.Q[np.triu_indices(self.n_vars)]), np.abs(self.linear),
        ])
        nz = coeffs[coeffs > 0]
        return float(nz.max() / nz.min()) if nz.size else 1.0

    @property
    def n_vars(self) -> int:
        """Total number of binary variables (qubits)."""
        return self.n_channels * self.n_steps

    def var_index(self, channel: int, step: int) -> int:
        """Map (channel, step) to flat variable index."""
        return channel * self.n_steps + step

    def var_channel_step(self, j: int) -> tuple[int, int]:
        """Map flat variable index to (channel, step)."""
        return divmod(j, self.n_steps)

    def energy(self, q: NDArray[np.int64]) -> float:
        """Evaluate the QUBO energy for a bitstring q."""
        q = np.asarray(q, dtype=np.float64)
        return float(q @ self.Q @ q + self.linear @ q + self.constant)

    def miss_distance(self, q: NDArray[np.int64]) -> NDArray[np.float64]:
        """Linearly-predicted final-state minus target."""
        q = np.asarray(q, dtype=np.float64)
        achieved = self.b_vectors.T @ q  # (4,)
        return achieved - self.target_gap

    def n_burns(self, q: NDArray[np.int64]) -> int:
        """Total number of active burn slots across all channels."""
        return int(np.sum(q))

    def channel_schedule(self, q: NDArray[np.int64], channel: int) -> NDArray[np.int64]:
        """Extract the on/off schedule for a single channel."""
        start = channel * self.n_steps
        return np.asarray(q[start : start + self.n_steps], dtype=np.int64)

    def delta_v(self, q: NDArray[np.int64]) -> float:
        """Cumulative |u|*dt summed over active (channel, step) pairs.

        Returned in nondimensional units. This is the fuel-proportional
        Δv (∫|u| dt under a piecewise-constant per-channel control). For
        single-channel mode, equals the magnitude of velocity change
        applied to the spacecraft. For multi-channel non-exclusive mode,
        sums each channel's contribution separately (= total propellant
        consumed across engines).
        """
        return float(self.thrust_magnitude * self.dt_decision * np.sum(q))

    def delta_v_m_s(self, q: NDArray[np.int64]) -> float:
        """Δv in metres per second (Earth-Moon nondimensionalisation)."""
        from qalunar.reference.edelbaum import VELOCITY_M_S
        return self.delta_v(q) * VELOCITY_M_S

    def brute_force(self) -> tuple[NDArray[np.int64], float]:
        """Find the optimal schedule by exhaustive enumeration.

        Only feasible for ``n_vars <= 20``. Returns the best bitstring
        and its energy.
        """
        M = self.n_vars
        if M > 20:
            raise ValueError(
                f"Brute force is intractable for n_vars={M} "
                f"(2^{M} = {2**M} evaluations)"
            )
        best_q = np.zeros(M, dtype=np.int64)
        best_energy = self.energy(best_q)
        for i in range(1, 2**M):
            q = np.array(
                [int(b) for b in format(i, f"0{M}b")],
                dtype=np.int64,
            )
            e = self.energy(q)
            if e < best_energy:
                best_energy = e
                best_q = q.copy()
        return best_q, best_energy


# ------------------------------------------------------------------
# QUBO builder
# ------------------------------------------------------------------


def _thrust_direction_at_step(
    direction: str,
    state: NDArray[np.float64],
    fixed_vector: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Return the unit thrust direction vector for a given mode and state."""
    if direction == "tangential":
        v = state[2:4]
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-15:
            return np.array([1.0, 0.0])
        return v / v_norm
    elif direction == "antitangential":
        v = state[2:4]
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-15:
            return np.array([-1.0, 0.0])
        return -v / v_norm
    elif direction == "normal":
        # Perpendicular to velocity, 90° counter-clockwise
        v = state[2:4]
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-15:
            return np.array([0.0, 1.0])
        return np.array([-v[1], v[0]]) / v_norm
    elif direction == "antinormal":
        v = state[2:4]
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-15:
            return np.array([0.0, -1.0])
        return np.array([v[1], -v[0]]) / v_norm
    elif direction == "fixed":
        if fixed_vector is None:
            raise ValueError("thrust_vector required when thrust_direction='fixed'")
        tv = np.asarray(fixed_vector, dtype=np.float64)
        return tv / np.linalg.norm(tv)
    else:
        raise ValueError(f"Unknown thrust_direction: {direction!r}")


def _channels_from_config(cfg: ThrustSchedulingConfig) -> list[str]:
    if cfg.thrust_channels is not None:
        return list(cfg.thrust_channels)
    return [cfg.thrust_direction]


def build_thrust_scheduling_qubo(
    dynamics: PlanarCR3BP,
    state0: NDArray[np.float64],
    target_state: NDArray[np.float64],
    t_span: tuple[float, float],
    n_decision_steps: int,
    config: ThrustSchedulingConfig | None = None,
    n_integration_substeps: int = 50,
    nominal_schedule: NDArray[np.int64] | None = None,
    nominal_final_override: NDArray[np.float64] | None = None,
) -> ThrustSchedulingQubo:
    """Build the on/off scheduling QUBO.

    Parameters
    ----------
    dynamics : PlanarCR3BP
        Dynamics model.
    state0 : (4,) ndarray
        Initial state ``[x, y, vx, vy]``.
    target_state : (4,) ndarray
        Desired final state ``[x, y, vx, vy]``.
    t_span : (t0, tf)
        Time interval.
    n_decision_steps : int
        Number of time steps where thrust on/off is decided.  In
        single-channel mode this equals the number of QUBO variables;
        in multi-channel mode the total is ``n_channels * n_steps``.
    config : ThrustSchedulingConfig, optional
        Scheduling parameters.
    n_integration_substeps : int
        RK4 substeps per decision interval for STM accuracy.
    nominal_schedule : (M,) int ndarray, optional
        Schedule about which to linearize.  ``None`` (default) is
        equivalent to the all-zeros schedule (linearize about the coast
        arc).  When provided, the reference trajectory is propagated
        under the controls implied by ``nominal_schedule`` and the STMs
        are computed along that thrust-applied trajectory.  This is the
        building block of the iterative re-linearization loop in
        :func:`solve_iterative`.
    nominal_final_override : (4,) ndarray, optional
        Truth anchor for the effective target gap.  ``None`` (default)
        uses the final state of the internal CR3BP propagation of the
        nominal schedule.  When a higher-fidelity truth oracle drives
        the outer loop (e.g. the GMAT oracle in
        :mod:`qalunar.highfidelity`), pass the *oracle's* final state of
        the nominal schedule here: the impulse-response sensitivities
        ``b_j`` remain the cheap CR3BP ones, but the gap the QUBO is
        asked to close becomes the true model gap — the offset-free-MPC
        correction that lets the binary loop converge under model
        mismatch instead of re-proposing the CR3BP optimum forever.

    Returns
    -------
    ThrustSchedulingQubo
        The QUBO with all precomputed data for sampling and analysis.
    """
    cfg = config if config is not None else ThrustSchedulingConfig()
    if cfg.impulse_quadrature not in ("midpoint", "left"):
        raise ValueError(
            f"impulse_quadrature must be 'midpoint' or 'left', "
            f"got {cfg.impulse_quadrature!r}"
        )
    state0 = np.asarray(state0, dtype=np.float64)
    target_state = np.asarray(target_state, dtype=np.float64)

    t0, tf = t_span
    N = n_decision_steps
    channels = _channels_from_config(cfg)
    K = len(channels)
    M = K * N

    if nominal_schedule is None:
        q_nom = np.zeros(M, dtype=np.int64)
    else:
        q_nom = np.asarray(nominal_schedule, dtype=np.int64)
        if q_nom.shape != (M,):
            raise ValueError(
                f"nominal_schedule shape {q_nom.shape} does not match "
                f"expected ({M},) = n_channels * n_decision_steps"
            )

    dt = (tf - t0) / N

    # 1. Determine reference trajectory by propagating with the nominal
    #    schedule's controls.  Direction at each step is evaluated using
    #    the running state, so we propagate one decision interval at a
    #    time when the schedule is non-trivial; for the all-coast case
    #    we use the simpler single-shot propagation.
    if not q_nom.any():
        t_full, states_full, stms_full = dynamics.propagate_stm(
            state0, t_span, n_steps=N * n_integration_substeps
        )
    else:
        # Build the per-interval control list, evaluating thrust direction
        # at the start of each interval along the running trajectory.
        controls: list[NDArray[np.float64] | None] = []
        running = state0.copy()
        # Interval-by-interval propagation just to capture states at the
        # left endpoint of each interval (for direction evaluation).
        decision_states = np.empty((N, 4))
        for i in range(N):
            decision_states[i] = running
            u_total = np.zeros(2)
            active = False
            for c, direction in enumerate(channels):
                if q_nom[c * N + i] == 1:
                    u_dir = _thrust_direction_at_step(
                        direction, running, cfg.thrust_vector
                    )
                    u_total = u_total + cfg.thrust_magnitude * u_dir
                    active = True
            u_i = u_total if active else None
            controls.append(u_i)
            t_i0 = t0 + i * dt
            _, seg = dynamics.propagate(
                running, (t_i0, t_i0 + dt),
                n_steps=n_integration_substeps, control=u_i,
            )
            running = seg[-1]
        # Now propagate state+STM jointly with the captured controls.
        t_full, states_full, stms_full = dynamics.propagate_stm_piecewise(
            state0, t_span, controls, n_substeps=n_integration_substeps,
        )

    # Decision points are at the start of each interval
    decision_indices = np.arange(0, N * n_integration_substeps, n_integration_substeps)
    t_grid = t_full[decision_indices]

    # STM from t0 to tf
    phi_0f = stms_full[-1]
    B_ctrl = dynamics.jacobian_control()  # (4, 2)

    # 2. b_vectors at each (channel, step) along the reference trajectory.
    #
    # b_j approximates the convolution  int_{t_i}^{t_i+dt} Phi(tf, s) B u ds
    # of a constant-thrust interval. The thrust *direction* is frozen at the
    # interval start (the convention of ``propagate_schedule``), but the
    # STM is taken at the interval midpoint (second-order midpoint rule)
    # unless the left-endpoint rule is explicitly requested.
    half = n_integration_substeps // 2 if cfg.impulse_quadrature == "midpoint" else 0
    b_vectors = np.empty((M, 4))
    for c, direction in enumerate(channels):
        for i in range(N):
            idx = decision_indices[i]
            state_i = states_full[idx]
            u_dir = _thrust_direction_at_step(
                direction, state_i, cfg.thrust_vector
            )
            u_i = cfg.thrust_magnitude * u_dir
            phi_0i = stms_full[idx + half]
            phi_if = phi_0f @ np.linalg.inv(phi_0i)
            j = c * N + i
            b_vectors[j] = phi_if @ B_ctrl @ u_i * dt

    # 3. Effective target gap.
    #
    # Linearization:  x(tf) ≈ x_nom(tf) + Σ_j (q_j - q_nom_j) b_j
    # Minimize:       || x(tf) - target ||_W²  + α Σ q_j  [ + exclusion ]
    #
    # Let r_nom = x_nom(tf) - target  and  c = r_nom - Σ_j q_nom_j b_j.
    # Then  || x(tf) - target ||_W²  =  || c + Σ_j q_j b_j ||_W²
    #                                =  || Σ_j q_j b_j - d_eff ||_W²
    # with d_eff = -c = (target - x_nom(tf)) + Σ_j q_nom_j b_j.
    if nominal_final_override is not None:
        nominal_final = np.asarray(nominal_final_override, dtype=np.float64)
    else:
        nominal_final = states_full[-1]
    d_eff = (target_state - nominal_final) + b_vectors.T @ q_nom.astype(np.float64)

    # 4. Quadratic form.
    if cfg.target_weights is not None:
        W = np.diag(np.asarray(cfg.target_weights, dtype=np.float64))
    else:
        W = np.eye(4)

    b_W = (W @ b_vectors.T).T  # (M, 4)
    Q = b_vectors @ b_W.T

    Wd = W @ d_eff
    linear = cfg.fuel_weight * np.ones(M) - 2.0 * b_vectors @ Wd
    constant = float(d_eff @ Wd)

    # 5. Exclusion penalty (multi-channel only).
    #
    # The penalty energy is ``E_excl = P * sum_{i, c<c'} q_{c,i} q_{c',i}``,
    # so each pair contributes ``P`` to the QUBO energy when both bits
    # are 1. Because ``q^T Q q`` with symmetric ``Q`` evaluates pair
    # ``(j1, j2)`` twice (once via ``Q[j1,j2]`` and once via ``Q[j2,j1]``),
    # we add ``P/2`` on each side so the total energy contribution from
    # the pair is exactly ``P``.
    P_used = 0.0
    if cfg.exclusive and K > 1:
        if cfg.exclusion_penalty is None:
            # Largest single-bit energy swing of the physical objective:
            # removing either bit of a violating pair changes the energy
            # by at most this, so any P above it makes every violating
            # state strictly worse than a feasible neighbour.
            swing = np.abs(linear) + np.abs(Q).sum(axis=1)
            P_used = 2.0 * float(swing.max())
        else:
            P_used = float(cfg.exclusion_penalty)
        P_half = 0.5 * P_used
        for i in range(N):
            for c1 in range(K):
                for c2 in range(c1 + 1, K):
                    j1 = c1 * N + i
                    j2 = c2 * N + i
                    Q[j1, j2] += P_half
                    Q[j2, j1] += P_half

    return ThrustSchedulingQubo(
        Q=Q,
        linear=linear,
        constant=constant,
        n_steps=N,
        n_channels=K,
        channel_labels=tuple(channels),
        b_vectors=b_vectors,
        target_gap=d_eff,
        fuel_weight=cfg.fuel_weight,
        t_grid=t_grid,
        coast_trajectory=states_full,
        thrust_magnitude=cfg.thrust_magnitude,
        dt_decision=dt,
        exclusion_penalty=P_used,
    )


# ------------------------------------------------------------------
# True-nonlinear evaluation
# ------------------------------------------------------------------


def propagate_schedule(
    dynamics: PlanarCR3BP,
    state0: NDArray[np.float64],
    t_span: tuple[float, float],
    schedule: NDArray[np.int64],
    config: ThrustSchedulingConfig,
    n_integration_substeps: int = 100,
    return_trajectory: bool = False,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Propagate the true nonlinear CR3BP under a binary schedule.

    Each decision interval gets a constant control vector summed over
    its active channels.  Direction is evaluated using the running state
    at the start of each interval (consistent with the linearization
    convention).

    Returns the final state, or ``(t, states)`` when
    ``return_trajectory=True``.
    """
    cfg = config
    channels = _channels_from_config(cfg)
    K = len(channels)
    q = np.asarray(schedule, dtype=np.int64)
    N = q.size // K
    if K * N != q.size:
        raise ValueError("schedule size does not divide by n_channels")

    t0, tf = t_span
    dt = (tf - t0) / N
    state = np.asarray(state0, dtype=np.float64).copy()
    if return_trajectory:
        traj_segments: list[NDArray[np.float64]] = [state[None, :].copy()]
        time_segments: list[NDArray[np.float64]] = [np.array([t0])]

    for i in range(N):
        u_total = np.zeros(2)
        active = False
        for c, direction in enumerate(channels):
            if q[c * N + i] == 1:
                u_dir = _thrust_direction_at_step(
                    direction, state, cfg.thrust_vector
                )
                u_total = u_total + cfg.thrust_magnitude * u_dir
                active = True
        u_i = u_total if active else None

        t_i0 = t0 + i * dt
        t_seg, seg = dynamics.propagate(
            state, (t_i0, t_i0 + dt),
            n_steps=n_integration_substeps, control=u_i,
        )
        state = seg[-1]
        if return_trajectory:
            traj_segments.append(seg[1:])
            time_segments.append(t_seg[1:])

    if return_trajectory:
        traj = np.concatenate(traj_segments, axis=0)
        times = np.concatenate(time_segments)
        return times, traj
    return state


# ------------------------------------------------------------------
# Iterative re-linearization driver
# ------------------------------------------------------------------


@dataclass
class IterativeSchedulingResult:
    """Result of the iterative re-linearization solve.

    Attributes
    ----------
    schedule : (M,) ndarray
        Final binary schedule.
    final_qubo : ThrustSchedulingQubo
        QUBO at the last linearization point.
    true_final_state : (4,) ndarray
        Final state of the true nonlinear propagation under ``schedule``.
    true_miss : (4,) ndarray
        ``true_final_state - target_state``.
    iterations : int
        Number of outer iterations executed.
    schedule_history : list of (M,) ndarrays
        Schedule at each iteration.
    true_miss_norm_history : list of float
        ``||true_miss||_2`` at each iteration (one per schedule).
    converged : bool
        ``True`` only when the loop reached a fixed point (the sampler
        re-proposed the incumbent) or the improvement fell below ``tol``.
        A trust-region *rejection* (non-improving candidate) or hitting
        ``max_iters`` leaves it ``False``; see ``converged_reason``.
    converged_reason : str
        Human-readable termination reason.
    """

    schedule: NDArray[np.int64]
    final_qubo: ThrustSchedulingQubo
    true_final_state: NDArray[np.float64]
    true_miss: NDArray[np.float64]
    iterations: int
    schedule_history: list[NDArray[np.int64]]
    true_miss_norm_history: list[float]
    converged: bool
    converged_reason: str


def solve_iterative(
    dynamics: PlanarCR3BP,
    state0: NDArray[np.float64],
    target_state: NDArray[np.float64],
    t_span: tuple[float, float],
    n_decision_steps: int,
    sampler: "callable",  # type: ignore[valid-type]
    config: ThrustSchedulingConfig | None = None,
    n_integration_substeps: int = 50,
    n_truth_substeps: int = 200,
    max_iters: int = 10,
    tol: float = 1e-9,
    initial_schedule: NDArray[np.int64] | None = None,
    accept_only_improvement: bool = True,
    truth_propagator: "callable | None" = None,  # type: ignore[valid-type]
    verbose: bool = False,
) -> IterativeSchedulingResult:
    """Solve the scheduling problem with sequential re-linearization.

    At each iteration: build the QUBO linearized around the current
    schedule, sample it for a candidate, propagate the candidate under
    true nonlinear dynamics, and accept it (always, or only when the
    true miss decreases).

    Parameters
    ----------
    sampler : callable
        ``sampler(qubo: ThrustSchedulingQubo) -> NDArray[int64]`` that
        returns a candidate schedule. Common choices:
        ``lambda q: q.brute_force()[0]`` for exact (small problems),
        or a simulated-annealing wrapper for larger ones.
    accept_only_improvement : bool
        When ``True`` (default), reject candidate schedules whose true
        miss is worse than the incumbent — equivalent to a trust-region
        safeguard on the linearization. When ``False``, always accept.
    tol : float
        Stop when ``||true_miss||`` improvement between iterations falls
        below this threshold.
    truth_propagator : callable, optional
        Replacement truth oracle with signature
        ``truth_propagator(state0, t_span, schedule, config) ->
        final_state``. ``None`` (default) uses the in-house CR3BP RK4
        (:func:`propagate_schedule`). Pass
        :func:`qalunar.highfidelity.make_gmat_truth_propagator` to make
        the trust-region accept/reject decisions against NASA GMAT
        ephemeris dynamics. The QUBO linearization itself stays in the
        CR3BP; only the candidate evaluation changes, so the
        monotone-acceptance guarantee is preserved with respect to the
        supplied oracle.
    """
    cfg = config if config is not None else ThrustSchedulingConfig()
    channels = _channels_from_config(cfg)
    K = len(channels)
    M = K * n_decision_steps

    if initial_schedule is None:
        q = np.zeros(M, dtype=np.int64)
    else:
        q = np.asarray(initial_schedule, dtype=np.int64).copy()
        if q.shape != (M,):
            raise ValueError(
                f"initial_schedule shape {q.shape} != ({M},)"
            )

    target = np.asarray(target_state, dtype=np.float64)

    # Trust-region accept rule and convergence are evaluated using a
    # weighted norm consistent with the QUBO objective.  When the QUBO
    # uses position-only targeting (e.g. for station-keeping), the
    # truth-miss check should also be position-only, otherwise a phase
    # offset that leaves position correct but velocity direction
    # rotated would dominate the L2 norm and reject every candidate.
    if cfg.target_weights is not None:
        _w_diag = np.asarray(cfg.target_weights, dtype=np.float64)
    else:
        _w_diag = np.ones(4, dtype=np.float64)
    # Use sqrt(W) so weighted L2 norm equals sqrt(miss^T W miss).
    _w_sqrt = np.sqrt(_w_diag)

    def _truth_miss(qq: NDArray[np.int64]) -> tuple[NDArray[np.float64], float]:
        if truth_propagator is not None:
            x_f = truth_propagator(state0, t_span, qq, cfg)
        else:
            x_f = propagate_schedule(
                dynamics, state0, t_span, qq, cfg,
                n_integration_substeps=n_truth_substeps,
            )
        miss = x_f - target
        return miss, float(np.linalg.norm(_w_sqrt * miss))

    incumbent_miss, incumbent_miss_norm = _truth_miss(q)
    schedule_history: list[NDArray[np.int64]] = [q.copy()]
    miss_history: list[float] = [incumbent_miss_norm]

    converged = False
    reason = "max_iters reached"
    last_qubo: ThrustSchedulingQubo | None = None

    # With an external truth oracle, anchor the QUBO's effective gap at
    # the oracle's final state of the incumbent schedule (already
    # evaluated — no extra oracle calls). Without it the CR3BP-built
    # QUBO would keep re-proposing the CR3BP optimum regardless of what
    # the oracle reports, and the loop would fixed-point immediately.
    def _truth_anchor() -> NDArray[np.float64] | None:
        if truth_propagator is None:
            return None
        return incumbent_miss + target

    for it in range(max_iters):
        qubo = build_thrust_scheduling_qubo(
            dynamics, state0, target, t_span,
            n_decision_steps=n_decision_steps,
            config=cfg,
            n_integration_substeps=n_integration_substeps,
            nominal_schedule=q,
            nominal_final_override=_truth_anchor(),
        )
        last_qubo = qubo
        q_cand = np.asarray(sampler(qubo), dtype=np.int64)
        if q_cand.shape != (M,):
            raise ValueError(
                f"sampler returned shape {q_cand.shape}, expected ({M},)"
            )
        if np.array_equal(q_cand, q):
            converged = True
            reason = f"schedule fixed point at iter {it}"
            break

        cand_miss, cand_miss_norm = _truth_miss(q_cand)
        if verbose:
            print(
                f"  iter {it}: incumbent ||miss||={incumbent_miss_norm:.3e} "
                f"-> candidate ||miss||={cand_miss_norm:.3e}"
            )

        if accept_only_improvement and cand_miss_norm > incumbent_miss_norm:
            # A rejected candidate is a stall of the linear model, not a
            # fixed point: report it as such instead of as convergence.
            converged = False
            reason = (
                f"non-improving candidate at iter {it} "
                f"(cand={cand_miss_norm:.3e} > incumbent="
                f"{incumbent_miss_norm:.3e})"
            )
            break

        improvement = incumbent_miss_norm - cand_miss_norm
        q = q_cand
        incumbent_miss = cand_miss
        incumbent_miss_norm = cand_miss_norm
        schedule_history.append(q.copy())
        miss_history.append(cand_miss_norm)

        if abs(improvement) < tol:
            converged = True
            reason = f"||miss|| improvement {improvement:.3e} < tol at iter {it}"
            break
    else:
        # Loop completed without break — re-linearize once more so that
        # final_qubo is anchored at the returned schedule.
        last_qubo = build_thrust_scheduling_qubo(
            dynamics, state0, target, t_span,
            n_decision_steps=n_decision_steps,
            config=cfg,
            n_integration_substeps=n_integration_substeps,
            nominal_schedule=q,
            nominal_final_override=_truth_anchor(),
        )

    assert last_qubo is not None

    return IterativeSchedulingResult(
        schedule=q,
        final_qubo=last_qubo,
        true_final_state=incumbent_miss + target,
        true_miss=incumbent_miss,
        iterations=len(schedule_history) - 1,
        schedule_history=schedule_history,
        true_miss_norm_history=miss_history,
        converged=converged,
        converged_reason=reason,
    )
