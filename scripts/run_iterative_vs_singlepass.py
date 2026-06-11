"""Side-by-side trajectory comparison: single-pass QUBO vs iterative
re-linearisation, on a long-arc cislunar correction scenario where
the linearisation gap is wide enough that single-pass selects a
schedule whose true nonlinear miss is much larger than the
linearised QUBO predicts.

This figure replaces the older abstract miss-vs-T plot.  Showing the
two schedules' actual trajectories communicates the methodological
contribution more directly.

Output: ``iterative_vs_singlepass_trajectory.png``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from qalunar.dynamics import PlanarCR3BP
from qalunar.dynamics.cr3bp import EARTH_MOON_MU
from qalunar.qubo.lunar_capture import moon_orbit_apolune_perilune
from qalunar.qubo.scheduling_samplers import sample_brute_force
from qalunar.qubo.thrust_scheduling import (
    ThrustSchedulingConfig,
    build_thrust_scheduling_qubo,
    propagate_schedule,
    solve_iterative,
)
from qalunar.reference.edelbaum import (
    ACCELERATION_M_S2, LENGTH_KM, TIME_S, VELOCITY_M_S,
)
from qalunar.reference.handoff import moon_relative_inertial


_MU = EARTH_MOON_MU
_GM = 1.0 - _MU
_R_HEO = 200_000.0 / LENGTH_KM
_V_CIRC = float(np.sqrt(_GM / _R_HEO))
_R_SYN = np.array([-_MU + _R_HEO, 0.0])
_OMEGA_CROSS_R = np.array([-_R_SYN[1], _R_SYN[0]])
_MOON_SYN = np.array([1 - _MU, 0.0])

# Long-arc scenario where the linearisation gap is wide
V_RATIO = 1.187
THRUST_MAG = 0.02
T_SPAN = (0.0, 4.0)
N_STEPS = 15
SPACECRAFT_MASS_KG = 100.0
MOON_RADIUS_KM = 1_737.0
# Laplace sphere of influence of the Moon ~ 66,100 km
MOON_SOI_NONDIM = 66_100.0 / LENGTH_KM


def _bf_sampler(qubo) -> np.ndarray:
    return sample_brute_force(qubo).schedule


def _burn_segments(schedule: np.ndarray, sub_per: int):
    out = []
    for i in range(schedule.size):
        if schedule[i] == 1:
            out.append((i * sub_per, (i + 1) * sub_per + 1))
    return out


def _closest_approach_diagnostic(
    traj: np.ndarray, times: np.ndarray, mu: float,
) -> dict:
    """Closest-approach analysis along a thrust-applied trajectory.

    Walks the actual achieved trajectory (including burns), finds the
    minimum distance to the Moon, and reports Moon-relative kinematics
    + capture status at that point.

    Note: this differs from ``find_perilune`` (which propagates an
    *unforced* trajectory). Here we want the perilune of the schedule's
    actual flight path, so we scan the precomputed trajectory directly.
    """
    moon_pos = np.array([1.0 - mu, 0.0])
    dist = np.linalg.norm(traj[:, :2] - moon_pos, axis=1)
    idx = int(np.argmin(dist))
    state_p = traj[idx]
    r_rel, v_rel = moon_relative_inertial(state_p, mu=mu)
    r_peri = float(np.linalg.norm(r_rel))
    v_peri = float(np.linalg.norm(v_rel))
    e_moon = 0.5 * v_peri ** 2 - mu / r_peri
    v_esc = float(np.sqrt(2.0 * mu / r_peri))
    is_bound = bool(e_moon < 0.0)
    if is_bound:
        v_inf = float("nan")
        r_apo, _, _ = moon_orbit_apolune_perilune(state_p, mu=mu)
    else:
        v_inf = float(np.sqrt(2.0 * e_moon))
        r_apo = float("inf")
    return {
        "idx": idx,
        "t": float(times[idx]),
        "state": state_p,
        "r_peri": r_peri,
        "v_peri": v_peri,
        "v_esc": v_esc,
        "v_inf": v_inf,
        "e_moon": e_moon,
        "is_bound": is_bound,
        "r_apo": r_apo,
        "inside_soi": bool(r_peri < MOON_SOI_NONDIM),
    }


def _format_diagnostic(label: str, d: dict) -> str:
    alt_km = (d["r_peri"] - MOON_RADIUS_KM / LENGTH_KM) * LENGTH_KM
    lines = [
        f"  {label}",
        f"    perilune r:    {d['r_peri'] * LENGTH_KM:>10,.0f} km "
        f"(alt {alt_km:,.0f} km)",
        f"    inside Moon SOI: {'YES' if d['inside_soi'] else 'no'} "
        f"(SOI ~ {MOON_SOI_NONDIM * LENGTH_KM:,.0f} km)",
        f"    v at perilune: {d['v_peri'] * VELOCITY_M_S:>10,.0f} m/s",
        f"    v_esc(r_peri): {d['v_esc'] * VELOCITY_M_S:>10,.0f} m/s",
        f"    Moon energy:   {d['e_moon']:+.3e}  -> "
        f"{'BOUND' if d['is_bound'] else 'HYPERBOLIC'}",
    ]
    if d["is_bound"]:
        apo_km = (
            d["r_apo"] * LENGTH_KM if np.isfinite(d["r_apo"]) else float("inf")
        )
        lines.append(f"    apolune:       {apo_km:>10,.0f} km")
    else:
        lines.append(
            f"    v_inf:         {d['v_inf'] * VELOCITY_M_S:>10,.0f} m/s "
            f"(excess speed at infinity)"
        )
    lines.append(
        f"    perilune time: {d['t']:.3f} nondim "
        f"({d['t'] * TIME_S / 86_400.0:.2f} d into arc)"
    )
    return "\n".join(lines)


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
    state_perfect = np.concatenate([_R_SYN,
        np.array([0.0, _V_CIRC * 1.190]) - _OMEGA_CROSS_R])
    _, traj_perfect = dyn.propagate(state_perfect, T_SPAN, n_steps=8000)
    target = traj_perfect[-1]
    state0 = np.concatenate([_R_SYN,
        np.array([0.0, _V_CIRC * V_RATIO]) - _OMEGA_CROSS_R])

    cfg = ThrustSchedulingConfig(
        thrust_magnitude=THRUST_MAG, thrust_direction="tangential",
        fuel_weight=0.0,
    )

    # ---- Single-pass solve: build QUBO around coast, take BF optimum ----
    qubo_single = build_thrust_scheduling_qubo(
        dyn, state0, target, T_SPAN,
        n_decision_steps=N_STEPS, config=cfg,
        n_integration_substeps=80,
    )
    q_single, _ = qubo_single.brute_force()

    pred_miss_single = float(np.linalg.norm(qubo_single.miss_distance(q_single)))
    sub_per = 300
    times_single, traj_single = propagate_schedule(
        dyn, state0, T_SPAN, q_single, cfg,
        n_integration_substeps=sub_per, return_trajectory=True,
    )
    true_miss_single = float(np.linalg.norm(traj_single[-1] - target))

    # ---- Iterative solve ----
    result_iter = solve_iterative(
        dyn, state0, target, T_SPAN,
        n_decision_steps=N_STEPS, sampler=_bf_sampler, config=cfg,
        n_integration_substeps=80, n_truth_substeps=sub_per, max_iters=8,
    )
    q_iter = result_iter.schedule
    times_iter, traj_iter = propagate_schedule(
        dyn, state0, T_SPAN, q_iter, cfg,
        n_integration_substeps=sub_per, return_trajectory=True,
    )
    true_miss_iter = float(np.linalg.norm(result_iter.true_miss))
    pred_miss_iter = float(np.linalg.norm(result_iter.final_qubo.miss_distance(q_iter)))

    # Coast trajectory (used as backdrop reference)
    _, traj_coast = dyn.propagate(state0, T_SPAN, n_steps=4000)

    # ---- Closest-approach diagnostics on the actual flown trajectories ----
    diag_single = _closest_approach_diagnostic(traj_single, times_single, _MU)
    diag_iter = _closest_approach_diagnostic(traj_iter, times_iter, _MU)

    # ---- Build figure ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)

    earth_syn = np.array([-_MU, 0.0])
    L1 = np.array([1 - _MU - (_MU / 3) ** (1 / 3), 0.0])

    panels = [
        dict(
            ax=axes[0],
            title=f"(a) Single-pass: linearise once around coast",
            traj=traj_single,
            schedule=q_single,
            pred=pred_miss_single,
            true_miss=true_miss_single,
            color="tab:red",
            iters=1,
            diag=diag_single,
        ),
        dict(
            ax=axes[1],
            title=f"(b) Iterative re-linearisation (Algorithm 1)",
            traj=traj_iter,
            schedule=q_iter,
            pred=pred_miss_iter,
            true_miss=true_miss_iter,
            color="tab:green",
            iters=result_iter.iterations,
            diag=diag_iter,
        ),
    ]

    for p in panels:
        ax = p["ax"]
        # Coast as a faint backdrop
        ax.plot(traj_coast[:, 0], traj_coast[:, 1], "--",
                color="0.7", lw=1.4, alpha=0.7,
                label="Coast (no thrust)")
        # The schedule's actual trajectory
        ax.plot(p["traj"][:, 0], p["traj"][:, 1], "-",
                color=p["color"], lw=2.4, alpha=0.95,
                label="Achieved trajectory")
        # Burn segments
        for k, (i0, i1) in enumerate(_burn_segments(p["schedule"], sub_per)):
            seg = p["traj"][i0:i1]
            ax.plot(seg[:, 0], seg[:, 1], "-",
                    color="tab:red" if p["color"] != "tab:red" else "darkred",
                    lw=4.5, alpha=0.95, solid_capstyle="round",
                    label="Burns" if k == 0 else None,
                    zorder=8)
        # Final state of this trajectory and target
        ax.plot(target[0], target[1], "*", color="tab:blue",
                ms=18, mec="black", mew=0.8, zorder=10,
                label="Target final state")
        ax.plot(p["traj"][-1, 0], p["traj"][-1, 1], "X",
                color=p["color"], ms=14, mec="black", mew=0.8, zorder=10,
                label="Achieved final state")

        # Earth, Moon, L1
        ax.add_patch(Circle(earth_syn, 0.025, color="tab:blue",
                            alpha=0.85, zorder=5))
        ax.add_patch(Circle(_MOON_SYN, 0.012, color="0.3",
                            alpha=0.85, zorder=5))
        ax.add_patch(Circle(_MOON_SYN, MOON_SOI_NONDIM, fill=False,
                            edgecolor="0.4", linestyle=":", linewidth=1.2,
                            alpha=0.7, zorder=4,
                            label="Moon SOI" if p["color"] == "tab:red"
                            else None))
        ax.plot(*L1, "x", color="tab:purple", ms=10, mew=2.4, zorder=5)

        # Perilune marker on the actual flown trajectory
        d = p["diag"]
        peri_xy = d["state"][:2]
        peri_color = "tab:green" if d["is_bound"] else "tab:orange"
        ax.plot(peri_xy[0], peri_xy[1], "o", mfc="white", mec=peri_color,
                mew=2.2, ms=11, zorder=11,
                label=("Perilune (bound)" if d["is_bound"]
                       else "Perilune (hyperbolic)"))
        ax.annotate("Earth", earth_syn, textcoords="offset points",
                    xytext=(-12, 16), fontsize=11, fontweight="bold")
        ax.annotate("Moon", _MOON_SYN, textcoords="offset points",
                    xytext=(8, 16), fontsize=11, fontweight="bold")

        miss_km = p["true_miss"] * LENGTH_KM
        pred_km = p["pred"] * LENGTH_KM
        n_burns = int(p["schedule"].sum())
        peri_alt_km = (
            d["r_peri"] - MOON_RADIUS_KM / LENGTH_KM
        ) * LENGTH_KM
        if d["is_bound"]:
            peri_kin = (
                f"perilune alt: {peri_alt_km:,.0f} km  [BOUND]\n"
                f"v_peri: {d['v_peri'] * VELOCITY_M_S:,.0f} m/s "
                f"< v_esc {d['v_esc'] * VELOCITY_M_S:,.0f} m/s"
            )
        else:
            peri_kin = (
                f"perilune alt: {peri_alt_km:,.0f} km  [hyperbolic]\n"
                f"v_inf: {d['v_inf'] * VELOCITY_M_S:,.0f} m/s · "
                f"excess {(d['v_peri'] - d['v_esc']) * VELOCITY_M_S:,.0f} m/s "
                f"over v_esc"
            )
        miss_label = (
            f"predicted miss: {pred_km:,.0f} km\n"
            f"true miss: {miss_km:,.0f} km\n"
            f"burns: {n_burns}/{N_STEPS} · "
            f"iters: {p['iters']}\n"
            f"{peri_kin}"
        )
        ax.text(0.02, 0.02, miss_label, transform=ax.transAxes,
                fontsize=10, family="monospace", va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor="white", edgecolor="0.7", alpha=0.95))

        ax.set_xlabel("$x$ (synodic, nondim)")
        ax.set_ylabel("$y$ (synodic, nondim)")
        ax.set_title(p["title"])
        ax.grid(alpha=0.3)
        ax.set_aspect("equal")

    # Smart shared limits to compare the two trajectories at the same scale
    all_x = np.concatenate([
        traj_coast[:, 0], traj_single[:, 0], traj_iter[:, 0],
        np.array([earth_syn[0], _MOON_SYN[0], target[0]]),
    ])
    all_y = np.concatenate([
        traj_coast[:, 1], traj_single[:, 1], traj_iter[:, 1],
        np.array([earth_syn[1], _MOON_SYN[1], target[1]]),
    ])
    pad = 0.06 * (all_x.max() - all_x.min())
    for p in panels:
        p["ax"].set_xlim(all_x.min() - pad, all_x.max() + pad)
        p["ax"].set_ylim(all_y.min() - pad, all_y.max() + pad)
    panels[0]["ax"].legend(loc="upper right", fontsize=10, framealpha=0.95)

    # Caption block at bottom
    accel_mm_s2 = THRUST_MAG * ACCELERATION_M_S2 * 1000.0
    thrust_N = SPACECRAFT_MASS_KG * (accel_mm_s2 * 1e-3)
    box_text = (
        "SCENARIO  "
        f"$T = {T_SPAN[1]:.1f}$ nondim ({T_SPAN[1]*TIME_S/86400.0:.1f} d) · "
        f"$N = {N_STEPS}$ slots · "
        f"a = {accel_mm_s2:.3f} mm/s$^{{2}}$ · "
        f"long-arc regime where linearisation is loose\n"
        "RESULT    "
        f"single-pass true miss {true_miss_single*LENGTH_KM:,.0f} km "
        f"$\\to$ iterative true miss {true_miss_iter*LENGTH_KM:,.0f} km "
        f"($\\times$ {true_miss_single/max(true_miss_iter, 1e-10):.0f} reduction "
        f"in {result_iter.iterations} outer iterations)"
    )
    fig.text(0.5, -0.02, box_text, fontsize=10, family="monospace",
             va="top", ha="center",
             bbox=dict(boxstyle="round,pad=0.6",
                       facecolor="white", edgecolor="0.6", alpha=0.97))

    out_path = Path(__file__).resolve().parent / "figures" / "iterative_vs_singlepass_trajectory.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")
    print(f"  single-pass: pred {pred_miss_single*LENGTH_KM:.0f} km, "
          f"true {true_miss_single*LENGTH_KM:.0f} km, "
          f"burns={int(q_single.sum())}")
    print(f"  iterative:   pred {pred_miss_iter*LENGTH_KM:.0f} km, "
          f"true {true_miss_iter*LENGTH_KM:.0f} km, "
          f"burns={int(q_iter.sum())}, iters={result_iter.iterations}")

    print()
    print("Closest-approach diagnostics (along the achieved trajectory):")
    print(_format_diagnostic("single-pass", diag_single))
    print(_format_diagnostic("iterative",   diag_iter))


if __name__ == "__main__":
    main()
