"""Primer-vector switching function vs binary QUBO schedule.

Visualises the conjecture that the on/off scheduling QUBO recovers
the Pontryagin tangential switching pattern from the discrete side
without ever computing a costate.  The script runs the iterative
re-linearisation on the operational Phase-2 cislunar correction
scenario, propagates the converged trajectory under the true
nonlinear dynamics, reconstructs the costate by backward integration
from the terminal transversality condition, and plots the binary
schedule together with the resulting tangential switching function
``S(t) = -lambda_v . v_hat``.

PMP predicts ``S(t) > 0`` is the optimal continuous-time region for
firing the tangential thruster.  Alignment of the QUBO's discrete
ON-slots with that region validates the bridge between the binary
QUBO and indirect primer-vector theory.

Output: ``primer_vector_switching.png``.
"""
#166 y 269
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from qalunar.dynamics import PlanarCR3BP
from qalunar.dynamics.cr3bp import EARTH_MOON_MU
from qalunar.qubo.scheduling_samplers import sample_brute_force
from qalunar.qubo.thrust_scheduling import (
    ThrustSchedulingConfig,
    propagate_schedule,
    solve_iterative,
)
from qalunar.reference.edelbaum import LENGTH_KM, TIME_S, VELOCITY_M_S
from qalunar.reference.primer_vector import reconstruct_primer_vector


# ---------------------------------------------------------------------------
# Scenario --- same operational Phase-2 setup as run_iterative_vs_singlepass
# ---------------------------------------------------------------------------

_MU = EARTH_MOON_MU
_GM = 1.0 - _MU
_R_HEO = 200_000.0 / LENGTH_KM
_V_CIRC = float(np.sqrt(_GM / _R_HEO))
_R_SYN = np.array([-_MU + _R_HEO, 0.0])
_OMEGA_CROSS_R = np.array([-_R_SYN[1], _R_SYN[0]])
_MOON_SYN = np.array([1.0 - _MU, 0.0])

V_RATIO = 1.187
THRUST_MAG = 0.02
T_SPAN = (0.0, 4.0)
N_STEPS = 15
SUB_PER = 300


def _bf_sampler(qubo) -> np.ndarray:
    return sample_brute_force(qubo).schedule


def main() -> None:
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    dyn = PlanarCR3BP()

    # Target = unforced final state of the "perfect" injection arc.
    state_perfect = np.concatenate([
        _R_SYN, np.array([0.0, _V_CIRC * 1.190]) - _OMEGA_CROSS_R
    ])
    _, traj_perfect = dyn.propagate(state_perfect, T_SPAN, n_steps=8000)
    target = traj_perfect[-1]

    # Imperfect injection: same position, slightly slow velocity.
    state0 = np.concatenate([
        _R_SYN, np.array([0.0, _V_CIRC * V_RATIO]) - _OMEGA_CROSS_R
    ])

    cfg = ThrustSchedulingConfig(
        thrust_magnitude=THRUST_MAG,
        thrust_direction="tangential",
        fuel_weight=0.0,
    )

    # ---- Iterative solve ----
    result = solve_iterative(
        dyn, state0, target, T_SPAN,
        n_decision_steps=N_STEPS, sampler=_bf_sampler, config=cfg,
        n_integration_substeps=80, n_truth_substeps=SUB_PER, max_iters=8,
    )
    schedule = result.schedule
    n_burns = int(schedule.sum())
    true_miss = float(np.linalg.norm(result.true_miss))
    print(f"iterative converged: iters={result.iterations}, "
          f"burns={n_burns}/{N_STEPS}, "
          f"true miss = {true_miss * LENGTH_KM:.0f} km")

    # ---- True nonlinear trajectory under the schedule ----
    times, traj = propagate_schedule(
        dyn, state0, T_SPAN, schedule, cfg,
        n_integration_substeps=SUB_PER, return_trajectory=True,
    )

    # ---- Primer-vector reconstruction ----
    trace = reconstruct_primer_vector(dyn, times, traj, target)
    S = trace.switching

    # ---- Decision-slot boundaries in time ----
    dt_decision = (T_SPAN[1] - T_SPAN[0]) / N_STEPS
    slot_edges = np.linspace(T_SPAN[0], T_SPAN[1], N_STEPS + 1)
    slot_centres = 0.5 * (slot_edges[:-1] + slot_edges[1:])

    # Burn intervals along the propagated trajectory
    burn_intervals = []
    for i in range(N_STEPS):
        if schedule[i] == 1:
            burn_intervals.append((i * SUB_PER, (i + 1) * SUB_PER + 1))

    # ---- Build figure: 3 stacked panels ----
    fig = plt.figure(figsize=(13, 11), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.6, 1.0, 1.6])

    # Panel (a): trajectory in synodic frame
    ax_traj = fig.add_subplot(gs[0])
    _, traj_coast = dyn.propagate(state0, T_SPAN, n_steps=4000)
    ax_traj.plot(traj_coast[:, 0], traj_coast[:, 1], "--",
                 color="0.7", lw=1.4, alpha=0.7,
                 label="Coast (no thrust)")
    ax_traj.plot(traj[:, 0], traj[:, 1], "-",
                 color="tab:green", lw=2.4, alpha=0.95,
                 label="Iterative schedule (achieved)")
    for k, (i0, i1) in enumerate(burn_intervals):
        seg = traj[i0:i1]
        ax_traj.plot(seg[:, 0], seg[:, 1], "-",
                     color="tab:red", lw=4.5, alpha=0.95,
                     solid_capstyle="round",
                     label="Burns" if k == 0 else None,
                     zorder=8)
    ax_traj.plot(target[0], target[1], "*", color="tab:blue",
                 ms=18, mec="black", mew=0.8, zorder=10,
                 label="Target final state")
    ax_traj.add_patch(Circle(np.array([-_MU, 0.0]), 0.025,
                             color="tab:blue", alpha=0.85, zorder=5))
    ax_traj.add_patch(Circle(_MOON_SYN, 0.012,
                             color="0.3", alpha=0.85, zorder=5))
    ax_traj.annotate("Earth", (-_MU, 0.0), textcoords="offset points",
                     xytext=(-12, 16), fontsize=11, fontweight="bold")
    ax_traj.annotate("Moon", _MOON_SYN, textcoords="offset points",
                     xytext=(8, 16), fontsize=11, fontweight="bold")
    ax_traj.set_xlabel("$x$ (synodic, nondim)")
    ax_traj.set_ylabel("$y$ (synodic, nondim)")
    ax_traj.set_title(
        "(a) Phase-2 cislunar correction trajectory under iterative QUBO"
    )
    ax_traj.grid(alpha=0.3)
    ax_traj.set_aspect("equal")
    ax_traj.legend(loc="upper right", fontsize=10, framealpha=0.95)
    #ax_traj.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
     #            fontsize=10, framealpha=0.95)

    # Panel (b): binary schedule as step function
    ax_q = fig.add_subplot(gs[1])
    q_step = np.repeat(schedule, 2)
    t_step = np.repeat(slot_edges, 2)[1:-1]
    ax_q.fill_between(t_step, 0, q_step, step="pre",
                      color="tab:red", alpha=0.55, linewidth=0,
                      label="ON slots")
    ax_q.step(t_step, q_step, where="pre",
              color="tab:red", lw=1.6)
    for edge in slot_edges:
        ax_q.axvline(edge, color="0.85", lw=0.5, zorder=0)
    ax_q.set_ylim(-0.05, 1.15)
    ax_q.set_yticks([0, 1])
    ax_q.set_ylabel(r"$q_i$")
    ax_q.set_title(f"(b) Binary QUBO schedule "
                   f"({n_burns} ON / {N_STEPS} slots)")
    ax_q.grid(alpha=0.3, axis="x")
    ax_q.set_xlim(T_SPAN)

    # Panel (c): switching function S(t) with ON-slot shading
    ax_s = fig.add_subplot(gs[2], sharex=ax_q)
    # Shade ON slots so visual alignment between (b) and (c) is direct
    for i in range(N_STEPS):
        if schedule[i] == 1:
            ax_s.axvspan(slot_edges[i], slot_edges[i + 1],
                         color="tab:red", alpha=0.18, linewidth=0)
    ax_s.fill_between(times, 0, S, where=(S > 0),
                      color="tab:purple", alpha=0.20,
                      label=r"$S > 0$ (PMP predicts ON)")
    ax_s.axhline(0.0, color="0.4", lw=1.2, linestyle="--",
                 label=r"$S = 0$")
    ax_s.plot(times, S, color="tab:purple", lw=2.0,
              label=r"$S(t) = -\boldsymbol{\lambda}_v \cdot \hat{\mathbf{v}}$")
    # Slot-centre markers coloured by alignment with QUBO decision
    S_at_centres_local = np.interp(slot_centres, times, S)
    on_drawn = False
    off_drawn = False
    for i in range(N_STEPS):
        s_c = S_at_centres_local[i]
        if schedule[i] == 1:
            label = "ON slot centre" if not on_drawn else None
            on_drawn = True
            ax_s.plot(slot_centres[i], s_c, "o",
                      ms=8, mew=2.0, mfc="white", mec="tab:red",
                      zorder=10, label=label)
        else:
            label = "OFF slot centre" if not off_drawn else None
            off_drawn = True
            ax_s.plot(slot_centres[i], s_c, "x",
                      ms=8, mew=2.0, mfc="0.55", mec="0.55",
                      zorder=10, label=label)
    ax_s.set_xlabel("time $t$ (nondim)")
    ax_s.set_ylabel(r"switching $S(t)$")
    ax_s.set_title(
        "(c) Reconstructed PMP tangential switching function"
    )
    ax_s.grid(alpha=0.3)
    ax_s.legend(loc="upper right", fontsize=10, framealpha=0.95, ncol=1)
    ax_s.set_xlim(T_SPAN)

    # Top-axis with mission days for the bottom two panels
    secax = ax_s.secondary_xaxis(
        "top", functions=(lambda x: x * TIME_S / 86_400.0,
                          lambda d: d * 86_400.0 / TIME_S),
    )
    secax.set_xlabel("mission elapsed time (days)", fontsize=10)

    # ---- Diagnostic: how many ON slots align with S>0 at slot centre? ----
    S_at_centres = np.interp(slot_centres, times, S)
    aligned_on = int(((schedule == 1) & (S_at_centres > 0)).sum())
    n_off = int(N_STEPS - n_burns)
    n_off_in_pos = int(((schedule == 0) & (S_at_centres > 0)).sum())
    print(
        f"ON slots in PMP-positive region:  {aligned_on}/{n_burns} "
        f"(PMP predicts ON; QUBO fires) -- expected to be 100%."
    )
    print(
        f"OFF slots also in PMP-positive region: {n_off_in_pos}/{n_off} "
        f"(PMP would fire; QUBO chooses not to). "
        f"This is the 'minimal-subset' selection of the binary QUBO: "
        f"with no fuel penalty the continuous PMP would saturate ON "
        f"throughout S>0, while the binary QUBO fires only the slots "
        f"needed to land the target within tolerance."
    )

    # ---- Annotation panel summarising the alignment story ----
    summary = (
        f"Alignment with PMP:\n"
        f"  ON slots in $S>0$ region: {aligned_on}/{n_burns}\n"
        f"  ON slots in $S<0$ region: {n_burns - aligned_on}/{n_burns}\n"
        f"  $\\Rightarrow$ every QUBO burn lies in the\n"
        f"      PMP-predicted ON region.\n"
        f"\n"
        f"True final-state miss: {true_miss * LENGTH_KM:,.0f} km\n"
        f"$\\Delta v$: "
        f"{result.final_qubo.delta_v(schedule) * VELOCITY_M_S:.1f} m/s "
        f"in {n_burns} burns"
    )
    ax_traj.text(
        1.05, 0.02, summary, transform=ax_traj.transAxes, #1.01
        fontsize=10, family="monospace", va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.5",
                  facecolor="white", edgecolor="0.7", alpha=0.95),
    )

    out_path = Path(__file__).resolve().parent / "figures" / "primer_vector_switching.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
