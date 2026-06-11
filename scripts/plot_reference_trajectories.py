"""Example CR3BP trajectories produced by the reference + indirect pipelines.

Produces three figures used in the project manual:

1. ``reference_gentle.png`` --- two-phase Earth-to-Moon low-thrust
   mission. Phase 1 is the analytical Edelbaum spiral from LEO out to
   a high circular Earth orbit at r = 0.5 nondim (~192,000 km),
   summarized in a caption box with Delta-v, time-of-flight and
   revolution count. Phase 2 is the energy-optimal CR3BP direct-NLP
   transfer from the Phase-1 handoff state to a rest rendezvous west
   of the Moon, plotted in the synodic frame. A right panel shows
   Phase-2 thrust-acceleration magnitude over time.

2. ``warm_start_aggressive.png`` --- a synthetic BC configuration chosen
   to stress the indirect solver's basin of attraction, where the
   BC-only Hermite seed stalls and the direct-NLP warm start recovers
   convergence. Plots three overlays in the synodic frame: (a) the
   direct-NLP classical reference, (b) the cold-start indirect
   solution, (c) the indirect solution warm-started from the reference.

3. ``convergence_history.png`` --- log-scale history of the nonlinear
   TPBVP residual across outer iterations, cold start vs. warm start,
   on the synthetic benchmark.

Run from the project root::

    python -m scripts.plot_reference_trajectories
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qalunar.dynamics import PlanarCR3BP
from qalunar.reference import (
    DirectCollocationConfig,
    edelbaum_spiral,
    solve_energy_optimal_cr3bp,
)
from qalunar.transcription import (
    IndirectTfcElmConfig,
    IndirectTfcElmTranscription,
    OuterLoopConfig,
)


# ---------------------------------------------------------------------------
# Figure-1 is a two-phase Earth->Moon mission:
#
#   Phase 1 (analytical): Edelbaum low-thrust spiral from a 6,900 km
#   circular Earth orbit (LEO, r_nondim = 0.018) to a high circular
#   Earth orbit at r_nondim = 0.5 (~192,000 km, half the Earth-Moon
#   distance). The thrust-acceleration magnitude is set to 0.1 mm/s^2
#   (a_nondim ~ 0.0368), a plausible solar-electric value. Phase 1 is
#   closed-form and delivers the Delta-v, time-of-flight and revolution
#   count directly from Edelbaum's equation; there is nothing to
#   optimize because tangential thrust is already optimal for a
#   coplanar circular-to-circular transfer.
#
#   Phase 2 (CR3BP direct NLP): starting from the Phase-1 handoff state
#   (position (-mu + 0.5, 0) in synodic coords, inertial velocity
#   (0, v2) prograde with v2 = sqrt((1-mu)/r2), mapped to synodic
#   velocity by subtracting omega x r), solve the energy-optimal
#   transfer to a rest rendezvous 0.1 west of the Moon. This is the
#   three-body leg that the whole QUBO / TFC+ELM stack targets.
#
# The point of the split is that Phase 1 would be hundreds of
# revolutions over ~2 years -- impossible for a single-polynomial TFC
# expansion to represent -- while Phase 2 is a ~3 nondim-time (~13 day)
# three-body transfer that is exactly what the tool is built for.
# Splitting the problem gets us a realistic LEO-origin mission without
# asking the CR3BP transcription to do work it cannot do.
#
# SYNTHETIC_LONG_BCS -- the synthetic numerical fixture from
# tests/test_indirect_tfc_elm.py. BCs are chosen to keep the trajectory
# away from both primaries so the Omega_xx Hessian stays bounded
# (entries scale as ~1/r^3 near either primary); this is what lets the
# indirect TPBVP have a predictable basin-of-attraction story that the
# warm-start figure illustrates. These BCs are NOT a physical
# Earth-Moon transfer.
# ---------------------------------------------------------------------------

EARTH_MOON_MU = 0.012150583925359

# Phase-1 Edelbaum spiral configuration
PHASE1_R_LEO = 0.018     # ~6,900 km circular Earth orbit
PHASE1_R_HIGH = 0.50     # ~192,200 km circular Earth orbit (handoff)
PHASE1_A_THRUST = 0.0368  # ~0.1 mm/s^2 in nondim units

# Phase-2 target: rest rendezvous 0.1 west of the Moon
PHASE2_RF = np.array([1.0 - EARTH_MOON_MU - 0.1, 0.0])
PHASE2_VF = np.array([0.0, 0.0])
PHASE2_TIME_OF_FLIGHT = 3.0

SYNTHETIC_LONG_BCS = dict(
    r0=np.array([-0.3, 0.0]),
    v0=np.array([0.0, 0.6]),
    rf=np.array([0.4, 0.2]),
    vf=np.array([-0.1, 0.0]),
    time_of_flight=2.5,
)


def _add_primaries(ax: plt.Axes, dyn: PlanarCR3BP) -> None:
    """Plot Earth and Moon markers in the synodic frame."""
    earth = dyn.earth_position()
    moon = dyn.moon_position()
    ax.plot(*earth, "o", color="tab:blue", ms=11, label="Earth", zorder=5)
    ax.plot(*moon, "o", color="dimgray", ms=6, label="Moon", zorder=5)


# ---------------------------------------------------------------------------
# Figure 1: gentle fixture reference solution
# ---------------------------------------------------------------------------


def figure_gentle_reference(out_path: Path) -> None:
    """Two-phase Earth->Moon mission plot.

    Phase 1 is summarised analytically from :func:`edelbaum_spiral`
    (no NLP involved). Phase 2 is the direct-NLP CR3BP transfer from
    the handoff state out of Phase 1 to a rest rendezvous west of the
    Moon.
    """
    dyn = PlanarCR3BP()

    # -- Phase 1: analytical Edelbaum spiral --
    spiral = edelbaum_spiral(
        r1=PHASE1_R_LEO,
        r2=PHASE1_R_HIGH,
        a_thrust=PHASE1_A_THRUST,
    )
    r0_handoff, v0_handoff = spiral.handoff_state_synodic()
    si = spiral.si_report()

    # -- Phase 2: CR3BP direct NLP from handoff to Moon rest rendezvous --
    cfg = DirectCollocationConfig(n_intervals=80, maxiter=600, tol=1e-8)
    ref = solve_energy_optimal_cr3bp(
        dynamics=dyn,
        r0=r0_handoff,
        v0=v0_handoff,
        rf=PHASE2_RF,
        vf=PHASE2_VF,
        time_of_flight=PHASE2_TIME_OF_FLIGHT,
        config=cfg,
    )
    assert ref.success, f"Phase-2 NLP solve failed: {ref.message}"

    # Use constrained_layout; tight_layout fights set_aspect('equal') and
    # leaves the position panel as a thin horizontal sliver. Width ratios
    # give the spatial plot more room than the 1D thrust plot.
    fig, axes = plt.subplots(
        1, 2, figsize=(13.0, 5.6),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
        constrained_layout=True,
    )

    # ---------- Left: synodic-frame trajectory ----------
    ax = axes[0]

    # Phase-1 exit orbit as a dashed gray circle around Earth (all the
    # revolutions would cover the same circle, so we only draw one).
    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    ex = -EARTH_MOON_MU + PHASE1_R_HIGH * np.cos(theta)
    ey = PHASE1_R_HIGH * np.sin(theta)
    ax.plot(
        ex,
        ey,
        color="tab:gray",
        lw=1.0,
        ls=":",
        alpha=0.7,
        label=f"Phase 1 exit orbit (r={PHASE1_R_HIGH})",
        zorder=1,
    )
    # Small LEO marker (mostly overlaps with Earth at this scale)
    lx = -EARTH_MOON_MU + PHASE1_R_LEO * np.cos(theta)
    ly = PHASE1_R_LEO * np.sin(theta)
    ax.plot(lx, ly, color="tab:gray", lw=0.8, alpha=0.5, zorder=1)

    # Phase 2 trajectory
    ax.plot(
        ref.x,
        ref.y,
        color="tab:orange",
        lw=2.0,
        label="Phase 2 direct-NLP CR3BP",
        zorder=4,
    )
    ax.plot(ref.x[0], ref.y[0], "o", color="tab:orange", ms=7, zorder=6)
    ax.plot(ref.x[-1], ref.y[-1], "s", color="tab:orange", ms=7, zorder=6)
    ax.annotate(
        "Phase 1 -> Phase 2\nhandoff",
        (ref.x[0], ref.y[0]),
        textcoords="offset points",
        xytext=(-34, -28),
        fontsize=8,
        ha="left",
    )
    ax.annotate(
        "rendezvous",
        (ref.x[-1], ref.y[-1]),
        textcoords="offset points",
        xytext=(6, -14),
        fontsize=8,
    )

    _add_primaries(ax, dyn)
    ax.set_aspect("equal")
    ax.set_xlim(-0.70, 1.15)
    ax.set_ylim(-0.75, 0.85)
    ax.set_xlabel("x  (synodic, nondim)")
    ax.set_ylabel("y  (synodic, nondim)")
    ax.set_title(
        f"Two-phase Earth\u2013Moon mission "
        f"(Phase 2: T = {PHASE2_TIME_OF_FLIGHT} nondim)"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)

    # Phase-1 analytical summary caption, anchored to the plot
    phase1_text = (
        "Phase 1  (analytical Edelbaum spiral)\n"
        f"  LEO  {si['r1_km']}  ->  high orbit  {si['r2_km']}\n"
        f"  $\\Delta v$ = {si['dv_m_s']}   "
        f"TOF = {si['tof_days']} ({si['tof_years']})\n"
        f"  {si['n_revs']} revs   at  {si['a_thrust_mm_s2']}"
    )
    ax.text(
        0.02,
        0.02,
        phase1_text,
        transform=ax.transAxes,
        fontsize=7.5,
        family="monospace",
        va="bottom",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor="tab:gray",
            alpha=0.92,
        ),
    )

    # ---------- Right: Phase-2 thrust magnitude over time ----------
    ax = axes[1]
    u_mag = np.sqrt(ref.ux ** 2 + ref.uy ** 2)
    ax.plot(ref.t, u_mag, color="tab:orange", lw=1.6)
    ax.fill_between(ref.t, 0.0, u_mag, color="tab:orange", alpha=0.15)
    ax.set_xlabel("t  (nondim)")
    ax.set_ylabel(r"$\|u\|$  (nondim acceleration)")
    ax.set_title(
        f"Phase 2 thrust magnitude   |   "
        f"J = (1/2)$\\int\\|u\\|^2$dt = {ref.objective:.3f}"
    )
    ax.grid(alpha=0.3)

    fig.savefig(out_path, dpi=150)
    print(
        f"saved: {out_path}   "
        f"(Phase-1 dv={si['dv_m_s']}, TOF={si['tof_days']};  "
        f"Phase-2 iters={ref.n_iterations}, defect={ref.max_defect:.1e})"
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: warm-start comparison on the aggressive fixture
# ---------------------------------------------------------------------------


def _solve_indirect(
    bcs: dict,
    dyn: PlanarCR3BP,
    initial_nominal: tuple[np.ndarray, np.ndarray] | None,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Run solve_sequential and return (x, y, nonlinear_residual history)."""
    cfg = IndirectTfcElmConfig(n_training=20, n_basis=80, seed=7)
    trans = IndirectTfcElmTranscription(dynamics=dyn, config=cfg, **bcs)
    sol = trans.solve_sequential(
        OuterLoopConfig(max_iter=max_iter, tol=1e-7, damping=1.0, line_search=True),
        initial_nominal=initial_nominal,
    )
    return sol.trajectory["x"], sol.trajectory["y"], sol.history["nonlinear_residual"]


def figure_warm_start_aggressive(out_path: Path) -> tuple[list[float], list[float]]:
    dyn = PlanarCR3BP()

    # Direct-NLP reference
    ref_cfg = DirectCollocationConfig(n_intervals=40, maxiter=400, tol=1e-8)
    ref = solve_energy_optimal_cr3bp(dynamics=dyn, config=ref_cfg, **SYNTHETIC_LONG_BCS)
    assert ref.success, f"synthetic reference solve failed: {ref.message}"

    # Cold start indirect
    x_cold, y_cold, hist_cold = _solve_indirect(
        SYNTHETIC_LONG_BCS, dyn, initial_nominal=None, max_iter=40
    )

    # Warm start indirect -- need the transcription's t grid to resample
    trans_cfg = IndirectTfcElmConfig(n_training=20, n_basis=80, seed=7)
    trans = IndirectTfcElmTranscription(
        dynamics=dyn, config=trans_cfg, **SYNTHETIC_LONG_BCS
    )
    x_bar, y_bar = ref.sample_position(trans.t)
    x_warm, y_warm, hist_warm = _solve_indirect(
        SYNTHETIC_LONG_BCS, dyn, initial_nominal=(x_bar, y_bar), max_iter=40
    )

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.plot(
        ref.x,
        ref.y,
        color="tab:orange",
        lw=2.2,
        label="direct-NLP reference",
        zorder=3,
    )
    ax.plot(
        x_cold,
        y_cold,
        color="tab:red",
        lw=1.4,
        ls="--",
        label="indirect, BC-only seed (stalled)",
        zorder=2,
    )
    ax.plot(
        x_warm,
        y_warm,
        color="tab:green",
        lw=1.6,
        label="indirect, warm-started",
        zorder=4,
    )
    # Start/end markers on the reference
    ax.plot(ref.x[0], ref.y[0], "o", color="black", ms=5, zorder=5)
    ax.plot(ref.x[-1], ref.y[-1], "s", color="black", ms=5, zorder=5)
    ax.annotate(
        "start",
        (ref.x[0], ref.y[0]),
        textcoords="offset points",
        xytext=(-12, -10),
        fontsize=8,
    )
    ax.annotate(
        "end",
        (ref.x[-1], ref.y[-1]),
        textcoords="offset points",
        xytext=(6, 4),
        fontsize=8,
    )

    _add_primaries(ax, dyn)
    ax.set_aspect("equal")
    ax.set_xlabel("x  (synodic, nondim)")
    ax.set_ylabel("y  (synodic, nondim)")
    ax.set_title(
        f"synthetic benchmark, T = {SYNTHETIC_LONG_BCS['time_of_flight']} nondim"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(
        f"saved: {out_path}   "
        f"(cold final NL={hist_cold[-1]:.2e}, warm final NL={hist_warm[-1]:.2e})"
    )
    plt.close(fig)
    return hist_cold, hist_warm


# ---------------------------------------------------------------------------
# Figure 3: convergence history, cold vs warm start
# ---------------------------------------------------------------------------


def figure_convergence(
    out_path: Path, hist_cold: list[float], hist_warm: list[float]
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    iters_cold = np.arange(1, len(hist_cold) + 1)
    iters_warm = np.arange(1, len(hist_warm) + 1)
    ax.semilogy(
        iters_cold,
        hist_cold,
        color="tab:red",
        marker="o",
        ms=4,
        lw=1.2,
        label="BC-only Hermite seed",
    )
    ax.semilogy(
        iters_warm,
        hist_warm,
        color="tab:green",
        marker="s",
        ms=4,
        lw=1.2,
        label="direct-NLP warm start",
    )
    ax.set_xlabel("outer iteration $k$")
    ax.set_ylabel(r"$\|$nonlinear TPBVP residual$\|_\infty$")
    ax.set_title("Sequential-linearization convergence, long transfer")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower left", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_gentle_reference(out_dir / "reference_gentle.png")
    hist_cold, hist_warm = figure_warm_start_aggressive(
        out_dir / "warm_start_aggressive.png"
    )
    figure_convergence(
        out_dir / "convergence_history.png", hist_cold, hist_warm
    )


if __name__ == "__main__":
    main()
