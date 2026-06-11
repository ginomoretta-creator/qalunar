"""Demo: low-thrust on/off scheduling via QUBO in the CR3BP.

Sets up a cislunar transfer scenario, builds the scheduling QUBO,
solves it with simulated annealing and brute force, and produces
figures comparing the results.

This demonstrates QUBO on a *naturally binary* problem — the engine
on/off decision — where quantum annealing has a genuine advantage
over classical continuous optimizers.

Produces:

1. ``scheduling_trajectory.png`` — optimal thrust schedule overlaid
   on the coast arc, with burn/coast segments color-coded.
2. ``scheduling_scaling.png``   — SA solve time vs number of decision
   steps, showing the combinatorial explosion that QA can address.

Run from the project root::

    python -m scripts.run_thrust_scheduling
"""

from __future__ import annotations

import time
from pathlib import Path

import dimod
import matplotlib.pyplot as plt
import neal
import numpy as np

from qalunar.dynamics import PlanarCR3BP
from qalunar.qubo.thrust_scheduling import (
    ThrustSchedulingConfig,
    build_thrust_scheduling_qubo,
)


# ---------------------------------------------------------------------------
# Scenario: cislunar transfer — Edelbaum spiral + QUBO-scheduled Moon arc
# ---------------------------------------------------------------------------

from qalunar.dynamics.cr3bp import EARTH_MOON_MU
from qalunar.reference.edelbaum import edelbaum_spiral, LENGTH_KM

_MU = EARTH_MOON_MU
_GM = 1.0 - _MU
_MOON_X = 1.0 - _MU  # Moon position in synodic frame

# Phase 1: Edelbaum spiral LEO -> 200,000 km
_R_LEO = 6578.0 / LENGTH_KM  # ~200 km altitude
_R_HEO = 200_000.0 / LENGTH_KM
_SPIRAL = edelbaum_spiral(r1=_R_LEO, r2=_R_HEO, a_thrust=0.01)

# Phase 2: injection into Moon-flyby trajectory, QUBO corrects
# v/v_circ = 1.190 reaches ~6,000 km from Moon (perfect injection);
# v/v_circ = 1.187 misses by ~31,000 km (imperfect injection).
_V_CIRC = float(np.sqrt(_GM / _R_HEO))
_V_RATIO_PERFECT = 1.190
_V_RATIO_IMPERFECT = 1.187

# Synodic frame states at handoff
_R_SYN = np.array([-_MU + _R_HEO, 0.0])
_OMEGA_CROSS_R = np.array([-_R_SYN[1], _R_SYN[0]])
_V_SYN_PERFECT = np.array([0.0, _V_CIRC * _V_RATIO_PERFECT]) - _OMEGA_CROSS_R
_V_SYN_IMPERFECT = np.array([0.0, _V_CIRC * _V_RATIO_IMPERFECT]) - _OMEGA_CROSS_R

STATE0 = np.concatenate([_R_SYN, _V_SYN_IMPERFECT])

THRUST_MAG = 0.02
T_SPAN = (0.0, 4.0)

# Lunar orbit for display (100 km altitude)
_R_LUNAR_ORBIT = (1737.0 + 100.0) / LENGTH_KM


def _setup_scenario(
    n_steps: int = 15,
    fuel_weight: float = 0.001,
) -> dict:
    """Build the scenario and QUBO."""
    dyn = PlanarCR3BP()

    # Target: where a perfect injection would arrive
    state_perfect = np.concatenate([_R_SYN, _V_SYN_PERFECT])
    _, perfect_traj = dyn.propagate(state_perfect, T_SPAN, n_steps=8000)
    target = perfect_traj[-1]

    cfg = ThrustSchedulingConfig(
        thrust_magnitude=THRUST_MAG,
        thrust_direction="tangential",
        fuel_weight=fuel_weight,
    )
    qubo = build_thrust_scheduling_qubo(
        dyn, STATE0, target, T_SPAN,
        n_decision_steps=n_steps, config=cfg,
        n_integration_substeps=100,
    )
    return dict(dyn=dyn, qubo=qubo, target=target, cfg=cfg)


# ---------------------------------------------------------------------------
# Figure 1: Trajectory with optimal thrust schedule
# ---------------------------------------------------------------------------


def _generate_spiral_points(
    n_outer_revs: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the outermost revolutions of the Edelbaum spiral in synodic coords.

    Returns arrays (x_syn, y_syn) for plotting.
    """
    gm = 1.0 - _MU
    n_pts = 80_000
    t = np.linspace(0, _SPIRAL.tof, n_pts)
    v_t = _SPIRAL.v1 - _SPIRAL.a_thrust * t
    r_t = gm / v_t**2

    # Angle accumulation: d_theta/dt = v^3/GM
    dt = t[1] - t[0]
    omega = v_t**3 / gm
    theta = np.cumsum(omega) * dt

    # Keep only the last n_outer_revs revolutions
    mask = theta >= (theta[-1] - n_outer_revs * 2 * np.pi)
    t_sel, r_sel, theta_sel = t[mask], r_t[mask], theta[mask]

    # Earth-centred inertial -> synodic
    x_ei = r_sel * np.cos(theta_sel)
    y_ei = r_sel * np.sin(theta_sel)
    x_syn = x_ei * np.cos(t_sel) + y_ei * np.sin(t_sel) + (-_MU)
    y_syn = -x_ei * np.sin(t_sel) + y_ei * np.cos(t_sel)
    return x_syn, y_syn


def figure_trajectory(out_path: Path) -> None:
    print("\n--- Building scheduling QUBO (N=15) ---")
    res = _setup_scenario(n_steps=15, fuel_weight=0.0003)
    qubo = res["qubo"]
    dyn = res["dyn"]
    target = res["target"]

    # Brute force
    N = qubo.n_steps
    print(f"  QUBO has {N} binary variables ({2**N:,} possible schedules)")
    t0 = time.perf_counter()
    best_q_bf, best_e_bf = qubo.brute_force()
    t_bf = time.perf_counter() - t0

    # SA sampling
    h = {i: qubo.Q[i, i] + qubo.linear[i] for i in range(N)}
    J = {}
    for i in range(N):
        for j in range(i + 1, N):
            coupling = 2.0 * qubo.Q[i, j]
            if abs(coupling) > 1e-15:
                J[(i, j)] = coupling
    bqm = dimod.BinaryQuadraticModel(h, J, qubo.constant, dimod.BINARY)
    t0 = time.perf_counter()
    result = neal.SimulatedAnnealingSampler().sample(bqm, num_reads=1000, seed=42)
    t_sa = time.perf_counter() - t0
    sa_sample = result.first.sample
    sa_q = np.array([sa_sample[i] for i in range(N)], dtype=np.int64)

    # Reconstruct thrust trajectory
    dt_decision = (T_SPAN[1] - T_SPAN[0]) / N
    sub = 100
    burn_x, burn_y = [], []
    coast_seg_x, coast_seg_y = [], []
    state = STATE0.copy()
    all_achieved = [state.copy()]
    for i in range(N):
        t_start = T_SPAN[0] + i * dt_decision
        if best_q_bf[i] == 1:
            v_vel = state[2:4]
            v_n = np.linalg.norm(v_vel)
            u = THRUST_MAG * v_vel / v_n if v_n > 1e-15 else None
        else:
            u = None
        _, seg_states = dyn.propagate(
            state, (t_start, t_start + dt_decision), n_steps=sub, control=u,
        )
        state = seg_states[-1]
        all_achieved.extend(seg_states[1:])
        seg_xy = seg_states[1:]
        if best_q_bf[i] == 1:
            burn_x.extend(seg_xy[:, 0])
            burn_y.extend(seg_xy[:, 1])
        else:
            coast_seg_x.extend(seg_xy[:, 0])
            coast_seg_y.extend(seg_xy[:, 1])
    all_achieved = np.array(all_achieved)

    # Closest approach to Moon
    coast_states = qubo.coast_trajectory
    dist_coast = np.sqrt((coast_states[:, 0] - _MOON_X)**2 + coast_states[:, 1]**2)
    dist_achieved = np.sqrt((all_achieved[:, 0] - _MOON_X)**2 + all_achieved[:, 1]**2)
    ca_coast_km = dist_coast.min() * LENGTH_KM
    ca_achieved_km = dist_achieved.min() * LENGTH_KM

    dv_nondim = qubo.delta_v(best_q_bf)
    dv_m_s = qubo.delta_v_m_s(best_q_bf)
    print(f"  Burns: {qubo.n_burns(best_q_bf)}, "
          f"Δv: {dv_nondim:.4e} nondim ({dv_m_s:.1f} m/s), "
          f"Moon closest: coast={ca_coast_km:.0f} km, corrected={ca_achieved_km:.0f} km")
    print(f"  SA = BF: {np.array_equal(sa_q, best_q_bf)}")

    # --- Plot: 2 rows, left column = overview + zoom, right = Gantt ---
    fig = plt.figure(figsize=(15, 7.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1], height_ratios=[1, 1])
    ax_overview = fig.add_subplot(gs[:, 0])   # full-height overview
    ax_zoom = fig.add_subplot(gs[0, 1])       # top-right: Moon zoom
    ax_gantt = fig.add_subplot(gs[1, 1])      # bottom-right: Gantt

    earth = dyn.earth_position()
    moon = dyn.moon_position()
    L1_x = 1.0 - _MU - (_MU / 3) ** (1 / 3)
    theta_circ = np.linspace(0, 2 * np.pi, 200)

    # ---- Overview panel ----
    ax = ax_overview

    # Phase 1: Edelbaum spiral
    x_sp, y_sp = _generate_spiral_points(n_outer_revs=3)
    ax.plot(x_sp, y_sp, '-', color='tab:green', lw=0.8, alpha=0.7,
            label=f'Edelbaum spiral (last 3/{_SPIRAL.n_revs:.0f} revs)')

    # Coast arc (imperfect, no corrections)
    ax.plot(coast_states[:, 0], coast_states[:, 1], '-',
            color='gray', lw=1.0, alpha=0.4, label='Coast (no correction)')

    # QUBO-corrected trajectory
    if coast_seg_x:
        ax.plot(coast_seg_x, coast_seg_y, '.', color='tab:blue',
                ms=0.5, label='Phase 2 coast')
    if burn_x:
        ax.plot(burn_x, burn_y, '.', color='tab:red',
                ms=1.0, label='Phase 2 burns')

    # Lunar orbit circle
    ax.plot(_MOON_X + _R_LUNAR_ORBIT * np.cos(theta_circ),
            _R_LUNAR_ORBIT * np.sin(theta_circ),
            '--', color='tab:orange', lw=1.2, alpha=0.8,
            label='Target lunar orbit')

    # Handoff marker
    ax.plot(STATE0[0], STATE0[1], 'o', color='black', ms=7, zorder=10)
    ax.annotate('Handoff\n(200,000 km)', (STATE0[0], STATE0[1]),
                textcoords='offset points', xytext=(-50, -20), fontsize=7.5)

    # Primaries and L1
    ax.plot(*earth, 'o', color='tab:blue', ms=10, zorder=5)
    ax.plot(*moon, 'o', color='dimgray', ms=7, zorder=5)
    ax.plot(L1_x, 0.0, 'x', color='tab:purple', ms=7, mew=1.5, zorder=5)
    ax.annotate('Earth', earth, textcoords='offset points',
                xytext=(-8, 8), fontsize=8)
    ax.annotate('Moon', moon, textcoords='offset points',
                xytext=(5, 8), fontsize=8)
    ax.annotate('$L_1$', (L1_x, 0.0), textcoords='offset points',
                xytext=(-12, -14), fontsize=9, color='tab:purple')

    # Zoom box indicator
    zoom_half = 0.12
    zoom_cx, zoom_cy = _MOON_X, 0.0
    rect = plt.Rectangle(
        (zoom_cx - zoom_half, zoom_cy - zoom_half),
        2 * zoom_half, 2 * zoom_half,
        fill=False, edgecolor='tab:orange', lw=1.2, ls='--', zorder=8,
    )
    ax.add_patch(rect)

    ax.set_xlabel('x (synodic, nondim)')
    ax.set_ylabel('y (synodic, nondim)')
    ax.set_title('Cislunar transfer: Edelbaum spiral + QUBO scheduling')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5, loc='lower left')
    ax.set_aspect('equal')

    # ---- Moon zoom panel ----
    ax = ax_zoom

    # Coast arc near Moon
    ax.plot(coast_states[:, 0], coast_states[:, 1], '-',
            color='gray', lw=1.5, alpha=0.5, label='No correction')

    # QUBO-corrected near Moon
    ax.plot(all_achieved[:, 0], all_achieved[:, 1], '-',
            color='tab:blue', lw=1.5, alpha=0.8, label='QUBO-corrected')

    # Lunar orbit circle
    ax.plot(_MOON_X + _R_LUNAR_ORBIT * np.cos(theta_circ),
            _R_LUNAR_ORBIT * np.sin(theta_circ),
            '--', color='tab:orange', lw=1.5, alpha=0.9,
            label='Target orbit (100 km)')

    # Moon body (filled circle, to scale)
    r_moon_body = 1737.0 / LENGTH_KM
    moon_circle = plt.Circle(
        (_MOON_X, 0.0), r_moon_body,
        color='dimgray', alpha=0.3, zorder=4,
    )
    ax.add_patch(moon_circle)
    ax.plot(_MOON_X, 0.0, 'o', color='dimgray', ms=4, zorder=5)
    ax.annotate('Moon', (_MOON_X, 0.0), textcoords='offset points',
                xytext=(6, 6), fontsize=8)

    # Closest approach markers
    idx_coast_ca = dist_coast.argmin()
    idx_ach_ca = dist_achieved.argmin()
    ax.plot(coast_states[idx_coast_ca, 0], coast_states[idx_coast_ca, 1],
            'v', color='gray', ms=8, zorder=10)
    ax.annotate(f'{ca_coast_km:,.0f} km',
                (coast_states[idx_coast_ca, 0], coast_states[idx_coast_ca, 1]),
                textcoords='offset points', xytext=(-55, -12), fontsize=7.5,
                color='gray', fontweight='bold')
    ax.plot(all_achieved[idx_ach_ca, 0], all_achieved[idx_ach_ca, 1],
            'v', color='tab:blue', ms=8, zorder=10)
    ax.annotate(f'{ca_achieved_km:,.0f} km',
                (all_achieved[idx_ach_ca, 0], all_achieved[idx_ach_ca, 1]),
                textcoords='offset points', xytext=(8, -12), fontsize=7.5,
                color='tab:blue', fontweight='bold')

    ax.set_xlim(zoom_cx - zoom_half, zoom_cx + zoom_half)
    ax.set_ylim(zoom_cy - zoom_half, zoom_cy + zoom_half)
    ax.set_xlabel('x (synodic)')
    ax.set_ylabel('y (synodic)')
    ax.set_title('Moon close approach (zoom)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5, loc='upper left')
    ax.set_aspect('equal')

    # ---- Gantt chart ----
    ax = ax_gantt
    for i in range(N):
        t_start = T_SPAN[0] + i * dt_decision
        color = 'tab:red' if best_q_bf[i] == 1 else 'tab:blue'
        alpha = 1.0 if best_q_bf[i] == 1 else 0.3
        ax.barh(0, dt_decision, left=t_start, height=0.5,
                color=color, alpha=alpha, edgecolor='white', linewidth=0.5)

    ax.set_xlabel('time (nondim)')
    ax.set_title(f'Burn schedule: {qubo.n_burns(best_q_bf)}/{N} steps active')
    ax.set_yticks([])
    ax.set_xlim(T_SPAN)

    # Metrics box
    metrics = (
        f"Decision steps:    {N}\n"
        f"Search space:      2^{N} = {2**N:,}\n"
        f"Burns:             {qubo.n_burns(best_q_bf)}\n"
        f"Δv:                {dv_m_s:.1f} m/s\n"
        f"Moon closest:\n"
        f"  no correction:   {ca_coast_km:,.0f} km\n"
        f"  QUBO-corrected:  {ca_achieved_km:,.0f} km\n"
        f"Brute force:       {t_bf:.2f}s\n"
        f"SA (1000 reads):   {t_sa:.2f}s\n"
        f"SA = BF:           {np.array_equal(sa_q, best_q_bf)}"
    )
    ax.text(0.98, 0.95, metrics, transform=ax.transAxes,
            fontsize=7.5, family='monospace', va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='tab:gray', alpha=0.92))

    fig.savefig(out_path, dpi=150)
    print(f"\nsaved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Scaling — SA time vs N, brute force time vs N
# ---------------------------------------------------------------------------


def figure_scaling(out_path: Path) -> None:
    print("\n--- Scaling analysis ---")
    n_values = [8, 10, 12, 14, 16, 18, 20]
    bf_times = []
    sa_times = []
    sa_matches = []

    print(f"{'N':>4} {'2^N':>10} {'BF(ms)':>10} {'SA(ms)':>10} {'match':>6}")
    print("-" * 45)

    for n in n_values:
        res = _setup_scenario(n_steps=n, fuel_weight=0.001)
        qubo = res["qubo"]
        N = qubo.n_steps

        # Brute force
        if n <= 20:
            t0 = time.perf_counter()
            best_q_bf, best_e_bf = qubo.brute_force()
            t_bf = (time.perf_counter() - t0) * 1000
        else:
            t_bf = float('nan')
            best_q_bf = None
            best_e_bf = float('inf')

        # SA
        h = {i: qubo.Q[i, i] + qubo.linear[i] for i in range(N)}
        J = {}
        for i in range(N):
            for j in range(i + 1, N):
                coupling = 2.0 * qubo.Q[i, j]
                if abs(coupling) > 1e-15:
                    J[(i, j)] = coupling
        bqm = dimod.BinaryQuadraticModel(h, J, qubo.constant, dimod.BINARY)

        t0 = time.perf_counter()
        result = neal.SimulatedAnnealingSampler().sample(bqm, num_reads=500, seed=42)
        t_sa = (time.perf_counter() - t0) * 1000

        sa_sample = result.first.sample
        sa_q = np.array([sa_sample[i] for i in range(N)], dtype=np.int64)
        match = best_q_bf is not None and np.array_equal(sa_q, best_q_bf)

        bf_times.append(t_bf)
        sa_times.append(t_sa)
        sa_matches.append(match)

        print(f"{n:4d} {2**n:10,d} {t_bf:10.1f} {t_sa:10.1f} {'Y' if match else 'N':>6}")

    # Extrapolate brute force for larger N
    n_extra = [25, 30, 35, 40, 50]
    # BF scales as O(2^N), estimate from the measured data
    bf_rate = bf_times[-1] / (2 ** n_values[-1])  # ms per evaluation

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    # Left: solve time comparison
    ax = axes[0]
    ax.semilogy(n_values, bf_times, 'o-', color='tab:blue', lw=1.6, label='Brute force')
    ax.semilogy(n_values, sa_times, 'D-', color='tab:green', lw=1.6, label='SA (500 reads)')

    # Extrapolated brute force
    n_ext = n_values + n_extra
    bf_ext = [bf_rate * 2**n * 1 for n in n_ext]  # very rough
    ax.semilogy(n_extra, [bf_rate * 2**n for n in n_extra],
                's--', color='tab:blue', lw=1.0, alpha=0.5, label='BF extrapolated')

    # QPU reference line
    ax.axhline(0.02, color='tab:purple', ls=':', lw=1.5,
               label='D-Wave QPU anneal (~20 us)')

    # Time reference lines
    ax.axhline(1000 * 60, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax.text(n_values[0], 1000 * 60 * 1.3, '1 minute', fontsize=8, color='gray')
    ax.axhline(1000 * 3600, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax.text(n_values[0], 1000 * 3600 * 1.3, '1 hour', fontsize=8, color='gray')

    ax.set_xlabel('N (decision steps = qubits)')
    ax.set_ylabel('solve time (ms)')
    ax.set_title('Scaling: brute force vs SA vs QPU')
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=8)

    # Right: search space size
    ax = axes[1]
    n_all = list(range(5, 55, 5))
    search_space = [2**n for n in n_all]
    ax.semilogy(n_all, search_space, 'D-', color='tab:red', lw=1.6)
    for n in [10, 20, 30, 40, 50]:
        ax.annotate(f'2^{n}\n≈{2**n:.0e}', (n, 2**n),
                    textcoords='offset points', xytext=(10, 0),
                    fontsize=7, ha='left')
    ax.axhline(5000, color='tab:purple', ls=':', lw=1.5,
               label='Advantage2 qubit limit')
    ax.set_xlabel('N (decision steps)')
    ax.set_ylabel('search space size')
    ax.set_title('Combinatorial explosion')
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=9)

    fig.savefig(out_path, dpi=150)
    print(f"\nsaved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Multi-direction scheduling (tangential + normal)
# ---------------------------------------------------------------------------


def figure_multi_direction(out_path: Path) -> None:
    print("\n--- Multi-direction scheduling (tangential + normal) ---")
    dyn = PlanarCR3BP()
    N = 10  # 2 channels x 10 steps = 20 qubits (brute-forceable)
    thrust_mag = 0.02
    t_span_multi = (0.0, 1.0)  # short arc — minimal STM direction mixing

    # Generate a target that REQUIRES normal thrust to reach:
    # propagate with pure normal thrust, creating a gap that
    # tangential-only scheduling cannot close effectively.
    v0 = STATE0[2:4]
    v0_norm = np.linalg.norm(v0)
    normal_dir = np.array([-v0[1], v0[0]]) / v0_norm
    u_target = thrust_mag * normal_dir  # pure normal
    _, states_thrust = dyn.propagate(
        STATE0, t_span_multi, n_steps=5000, control=u_target,
    )
    target = states_thrust[-1]

    # Position-only targeting (velocity free): realistic for waypoint
    # problems where a subsequent maneuver corrects velocity.
    W = np.array([1.0, 1.0, 0.0, 0.0])

    # Single-channel (tangential only)
    cfg_single = ThrustSchedulingConfig(
        thrust_magnitude=thrust_mag,
        thrust_channels=("tangential",),
        fuel_weight=0.0,
        target_weights=W,
    )
    qubo_single = build_thrust_scheduling_qubo(
        dyn, STATE0, target, t_span_multi,
        n_decision_steps=N, config=cfg_single,
        n_integration_substeps=100,
    )

    # Multi-channel (tangential + normal)
    cfg_multi = ThrustSchedulingConfig(
        thrust_magnitude=thrust_mag,
        thrust_channels=("tangential", "normal"),
        fuel_weight=0.0,
        target_weights=W,
    )
    qubo_multi = build_thrust_scheduling_qubo(
        dyn, STATE0, target, t_span_multi,
        n_decision_steps=N, config=cfg_multi,
        n_integration_substeps=100,
    )

    # Multi-channel exclusive (at most one direction per step)
    cfg_excl = ThrustSchedulingConfig(
        thrust_magnitude=thrust_mag,
        thrust_channels=("tangential", "normal"),
        fuel_weight=0.0,
        target_weights=W,
        exclusive=True, exclusion_penalty=10.0,
    )
    qubo_excl = build_thrust_scheduling_qubo(
        dyn, STATE0, target, t_span_multi,
        n_decision_steps=N, config=cfg_excl,
        n_integration_substeps=100,
    )

    # Solve all three
    best_s, e_s = qubo_single.brute_force()
    best_m, e_m = qubo_multi.brute_force()
    best_x, e_x = qubo_excl.brute_force()

    # Position miss only (consistent with W)
    miss_s = np.linalg.norm(qubo_single.miss_distance(best_s)[:2])
    miss_m = np.linalg.norm(qubo_multi.miss_distance(best_m)[:2])
    miss_x = np.linalg.norm(qubo_excl.miss_distance(best_x)[:2])

    print(f"  Single (tangential):  {qubo_single.n_vars} vars, "
          f"{qubo_single.n_burns(best_s)} burns, pos_miss={miss_s:.4e}")
    print(f"  Multi (tan+normal):   {qubo_multi.n_vars} vars, "
          f"{qubo_multi.n_burns(best_m)} burns, pos_miss={miss_m:.4e}")
    print(f"  Exclusive (tan+norm): {qubo_excl.n_vars} vars, "
          f"{qubo_excl.n_burns(best_x)} burns, pos_miss={miss_x:.4e}")
    improvement = (miss_s - miss_m) / miss_s * 100
    print(f"  Multi-channel improvement: {improvement:.1f}%")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    dt_decision = (t_span_multi[1] - t_span_multi[0]) / N

    titles = [
        f"Tangential only ({qubo_single.n_vars} qubits)",
        f"Tangential + Normal ({qubo_multi.n_vars} qubits)",
        f"Exclusive ({qubo_excl.n_vars} qubits)",
    ]
    qubos = [qubo_single, qubo_multi, qubo_excl]
    bests = [best_s, best_m, best_x]
    misses = [miss_s, miss_m, miss_x]
    colors = {"tangential": "tab:red", "normal": "tab:purple"}

    for ax_idx, (ax, qubo, best_q, miss, title) in enumerate(
        zip(axes, qubos, bests, misses, titles)
    ):
        K = qubo.n_channels
        bar_height = 0.4
        for c in range(K):
            label = qubo.channel_labels[c]
            sched = qubo.channel_schedule(best_q, c)
            color = colors.get(label, "tab:gray")
            for i in range(N):
                t_start = t_span_multi[0] + i * dt_decision
                alpha = 1.0 if sched[i] == 1 else 0.15
                ax.barh(c, dt_decision, left=t_start, height=bar_height,
                        color=color, alpha=alpha, edgecolor='white', lw=0.5)
            # Label on y-axis
            ax.text(-0.15, c, label[:3].upper(), transform=ax.get_yaxis_transform(),
                    fontsize=9, va='center', ha='right', fontweight='bold',
                    color=color)

        n_burns = qubo.n_burns(best_q)
        dv_m_s = qubo.delta_v_m_s(best_q)
        ax.set_xlabel('time (nondim)')
        ax.set_title(title, fontsize=10)
        ax.set_yticks([])
        ax.set_xlim(t_span_multi)
        ax.set_ylim(-0.5, K - 0.5)
        ax.text(0.98, 0.95,
                f"burns: {n_burns}\nΔv: {dv_m_s:.0f} m/s\nmiss: {miss:.2e}",
                transform=ax.transAxes, fontsize=8.5, family='monospace',
                va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='tab:gray', alpha=0.9))

    fig.suptitle('Multi-direction thrust scheduling', fontsize=12, fontweight='bold')
    fig.savefig(out_path, dpi=150)
    print(f"\nsaved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Low-thrust on/off scheduling via QUBO")
    print("  Naturally binary — no quantization needed")
    print("=" * 60)

    figure_trajectory(out_dir / "scheduling_trajectory.png")
    figure_scaling(out_dir / "scheduling_scaling.png")
    figure_multi_direction(out_dir / "scheduling_multi_direction.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
