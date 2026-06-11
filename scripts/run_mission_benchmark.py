"""Benchmark the qalunar all-binary mission against real lunar missions.

This places the end-to-end result of ``scripts/run_full_mission_demo`` (a
100 kg microsatellite flown to the Moon with a single throttleable ion/Hall
thruster, every burn an on/off binary command) next to flown reference
missions that span the three Earth-Moon transfer regimes:

* **SMART-1** (ESA, 2003) -- the apples-to-apples benchmark: solar-electric
  low-thrust *spiral* out from GTO to lunar orbit. Same propulsion regime as
  qalunar.
* **CAPSTONE** (NASA/Advanced Space, 2022) -- a recent *low-energy ballistic*
  lunar transfer: tiny transfer/capture dV bought with months of time of
  flight by exploiting solar perturbations.
* **Apollo (TLI + LOI)** -- the classical *impulsive chemical* direct
  transfer: high dV, three to four days.

The comparison metric is the dV budget (and the resulting propellant fraction
via the rocket equation), with time of flight and initial thrust-to-mass as
the explanatory axes. The point is *not* "qalunar is cheaper than SMART-1":
the two start from different orbits (GEO vs GTO) and at very different
thrust-to-mass ratios. The point is that the all-binary QUBO mission lands in
the right place for an electric-propulsion cislunar transfer -- a few km/s of
dV at a ~15-20% propellant fraction -- and that its time of flight is governed
by thrust-to-mass exactly as the flown missions are.

Caveats (kept explicit, because the comparison is only fair with them):
  1. Starting orbit: qalunar departs a *circular* GEO; SMART-1 departs a GTO
     with a ~650 km perigee, which costs extra dV to raise. qalunar's lower
     Phase-1 dV is partly this head start.
  2. Thrust-to-mass: qalunar's 350 mN / 100 kg = 3.5 mm/s^2 is ~18x SMART-1's
     0.19 mm/s^2. This -- not any algorithmic magic -- is why qalunar's time
     of flight is ~1 month vs SMART-1's ~13 months. 350 mN on a 100 kg bus is
     an *optimistic* (high-power-cluster) EP assumption.
  3. Duty cycle: SMART-1 thrust only ~1/3-1/2 of each orbit (power, eclipse,
     thermal); qalunar's Phase-1 Edelbaum arc is modelled as continuous.
  4. Planar idealization: qalunar is planar CR3BP; the flown missions are 3D
     and also change inclination.

Sources (see module footer for URLs):
  SMART-1: ESA / Wikipedia / IEPC-2007 operational papers.
  CAPSTONE: Advanced Space / BLT cheat sheet.
  Apollo TLI/LOI: NASA delta-v budgets.

Run::

    python -m scripts.run_mission_benchmark
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Mission budget record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MissionBudget:
    """A single Earth->Moon mission's headline budget.

    All fields are best-available published figures (or, for qalunar, the
    output of ``run_full_mission_demo``). ``dv_total_ms`` is the propulsive
    dV credited to the spacecraft to go from its starting orbit to a lunar
    orbit; ``tof_days`` is the corresponding transfer time of flight.
    """

    name: str
    regime: str                 # "EP spiral" | "low-energy" | "impulsive"
    dv_total_ms: float          # m/s
    tof_days: float             # days
    wet_mass_kg: float          # launch/initial mass
    thrust_mn: float            # representative thrust (main level)
    isp_s: float                # specific impulse (for propellant estimate)
    start_orbit: str
    note: str = ""

    @property
    def accel_mm_s2(self) -> float:
        """Initial thrust-to-mass as an acceleration in mm/s^2."""
        return (self.thrust_mn * 1e-3) / self.wet_mass_kg * 1e3

    @property
    def prop_fraction(self) -> float:
        """Propellant mass fraction from the rocket equation at ``isp_s``."""
        ve = self.isp_s * 9.80665
        return float(1.0 - np.exp(-self.dv_total_ms / ve))

    @property
    def prop_mass_kg(self) -> float:
        return self.wet_mass_kg * self.prop_fraction


# ---------------------------------------------------------------------------
# qalunar end-to-end result (from scripts/run_full_mission_demo, 2026-06-05)
# ---------------------------------------------------------------------------
# Phase 1 (Edelbaum spiral GEO->200,000 km + injection boost): 1924.3 m/s, 6.4 d
# Phase 2 (cislunar QUBO correction, cruise 5.4 mN):              16.4 m/s, ~17 d
# Phase 3 (sliding-window QUBO lunar capture, main 350 mN):      561.4 m/s, 8.0 d
# Total: 2502.1 m/s, ~31.8 days. Single 350 mN / 5.4 mN ion-Hall thruster.
QALUNAR_PHASES = {
    "Phase 1 (Edelbaum spiral + injection)": 1924.3,
    "Phase 2 (cislunar QUBO correction)": 16.4,
    "Phase 3 (sliding-window QUBO capture)": 561.4,
}
QALUNAR_DV_TOTAL = sum(QALUNAR_PHASES.values())   # 2502.1 m/s
QALUNAR_TOF_DAYS = 31.8
# Isp of a representative high-power Hall thruster, matched to SMART-1's
# PPS-1350 (Isp ~1640 s) so the propellant comparison is on equal footing.
QALUNAR_ISP_S = 1640.0


def qalunar_budget() -> MissionBudget:
    return MissionBudget(
        name="qalunar (this work)",
        regime="EP spiral",
        dv_total_ms=QALUNAR_DV_TOTAL,
        tof_days=QALUNAR_TOF_DAYS,
        wet_mass_kg=100.0,
        thrust_mn=350.0,
        isp_s=QALUNAR_ISP_S,
        start_orbit="GEO circular (35,786 km alt)",
        note="all-binary on/off schedule; planar CR3BP",
    )


def qalunar_smart1_point_budget() -> MissionBudget:
    """The qalunar transfer re-run at SMART-1's own operating point.

    From ``run_smart1_operating_point``: an Edelbaum spiral at
    $a_0 = 70\\,\\mathrm{mN}/367\\,\\mathrm{kg}$ from a GTO-energy circular start
    to the Moon's SOI gives 2900 m/s of Earth-escape spiral over 136 revs and a
    duty-adjusted 440 days; adding the (thrust-independent) lunar-capture
    brating of the demo (~561 m/s) totals ~3461 m/s -- landing on SMART-1.
    """
    return MissionBudget(
        name="qalunar @ SMART-1 pt",
        regime="EP spiral",
        dv_total_ms=2900.0 + 561.4,   # spiral-to-SOI + capture (energy-based)
        tof_days=440.0,               # 176 d at 100% duty / 0.40 duty cycle
        wet_mass_kg=367.0,
        thrust_mn=70.0,
        isp_s=1640.0,
        start_orbit="GTO-energy circular (a=24,629 km)",
        note="qalunar method at SMART-1 a_0 and start energy; 40% duty",
    )


# ---------------------------------------------------------------------------
# Flown reference missions
# ---------------------------------------------------------------------------


def reference_missions() -> list[MissionBudget]:
    return [
        MissionBudget(
            name="SMART-1 (ESA, 2003)",
            regime="EP spiral",
            dv_total_ms=3500.0,          # ~3.5 km/s GTO -> lunar orbit
            tof_days=410.0,              # ~13.5 months launch -> lunar capture
            wet_mass_kg=367.0,           # launch mass (287 kg dry, 82 kg Xe)
            thrust_mn=70.0,              # PPS-1350 Hall, nominal
            isp_s=1640.0,                # ~16,000 m/s exhaust
            start_orbit="GTO 7035 x 42,223 km",
            note="~207 revs, thrust 1/3-1/2 of each orbit, 74 kg Xe used",
        ),
        MissionBudget(
            name="CAPSTONE (2022)",
            regime="low-energy",
            dv_total_ms=200.0,           # BLT transfer+capture is tiny (~tens m/s);
                                         # ~200 m/s incl. correction/clean-up margin
            tof_days=125.0,              # ~4 months ballistic lunar transfer
            wet_mass_kg=25.0,            # 12U cubesat, ~25 kg
            thrust_mn=25.0,              # hydrazine RCS (representative)
            isp_s=200.0,
            start_orbit="post-Photon BLT injection",
            note="Photon kick stage did the energy-raising; BLT capture ~tens m/s",
        ),
        MissionBudget(
            name="Apollo (TLI + LOI)",
            regime="impulsive",
            dv_total_ms=3107.0 + 900.0,  # TLI ~3.1 km/s from LEO + LOI ~0.9 km/s
            tof_days=3.5,                # ~3-4 days direct transfer
            wet_mass_kg=45_000.0,        # CSM+LM stack order of magnitude
            thrust_mn=900_000_000.0,     # S-IVB J-2 ~900 kN (impulsive regime)
            isp_s=421.0,                 # J-2 chemical (vacuum)
            start_orbit="LEO ~185 km parking",
            note="impulsive chemical direct transfer (regime anchor)",
        ),
    ]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_table(missions: list[MissionBudget]) -> None:
    print("=" * 100)
    print("  Earth -> Moon mission benchmark: qalunar all-binary vs flown missions")
    print("=" * 100)
    hdr = (f"  {'Mission':<24}{'Regime':<12}{'dV [m/s]':>10}"
           f"{'TOF [d]':>9}{'a0 [mm/s2]':>12}{'m_wet [kg]':>11}"
           f"{'prop frac':>11}")
    print(hdr)
    print("  " + "-" * 96)
    for m in missions:
        print(f"  {m.name:<24}{m.regime:<12}{m.dv_total_ms:>10,.0f}"
              f"{m.tof_days:>9,.0f}{m.accel_mm_s2:>12.3f}{m.wet_mass_kg:>11,.0f}"
              f"{m.prop_fraction:>11.1%}")
    print("  " + "-" * 96)


def _print_smart1_focus(q: MissionBudget, s: MissionBudget) -> None:
    print()
    print("-" * 100)
    print("  Apples-to-apples: qalunar vs SMART-1 (both solar-electric low-thrust)")
    print("-" * 100)
    rows = [
        ("Total dV [m/s]", f"{q.dv_total_ms:,.0f}", f"{s.dv_total_ms:,.0f}",
         "comparable order of magnitude; qalunar lower partly due to GEO vs GTO start"),
        ("Time of flight [days]", f"{q.tof_days:,.0f}", f"{s.tof_days:,.0f}",
         f"qalunar ~{s.tof_days / q.tof_days:.0f}x faster, tracking its ~18x higher a0"),
        ("Initial a0 [mm/s^2]", f"{q.accel_mm_s2:.3f}", f"{s.accel_mm_s2:.3f}",
         f"ratio {q.accel_mm_s2 / s.accel_mm_s2:.0f}x (qalunar is optimistic high-power EP)"),
        ("Propellant fraction", f"{q.prop_fraction:.1%}", f"{s.prop_fraction:.1%}",
         f"both ~15-20% at Isp~{s.isp_s:.0f}s; qalunar {q.prop_mass_kg:.1f} kg of {q.wet_mass_kg:.0f} kg"),
    ]
    w = max(len(r[0]) for r in rows)
    print(f"  {'':<{w}}{'qalunar':>12}{'SMART-1':>12}   comment")
    for label, a, b, comment in rows:
        print(f"  {label:<{w}}{a:>12}{b:>12}   {comment}")
    print()
    print("  Phase dV breakdown (qalunar):")
    for label, dv in QALUNAR_PHASES.items():
        print(f"    {label:<42}{dv:>9,.1f} m/s")
    print(f"    {'TOTAL':<42}{QALUNAR_DV_TOTAL:>9,.1f} m/s")


def _write_csv(missions: list[MissionBudget], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "name", "regime", "dv_total_ms", "tof_days", "accel_mm_s2",
            "wet_mass_kg", "thrust_mn", "isp_s", "prop_fraction",
            "prop_mass_kg", "start_orbit", "note",
        ])
        for m in missions:
            writer.writerow([
                m.name, m.regime, f"{m.dv_total_ms:.1f}", f"{m.tof_days:.1f}",
                f"{m.accel_mm_s2:.4f}", f"{m.wet_mass_kg:.1f}",
                f"{m.thrust_mn:.1f}", f"{m.isp_s:.1f}",
                f"{m.prop_fraction:.4f}", f"{m.prop_mass_kg:.2f}",
                m.start_orbit, m.note,
            ])


def _plot(missions: list[MissionBudget], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    regime_color = {
        "EP spiral": "#1f77b4",
        "low-energy": "#2ca02c",
        "impulsive": "#d62728",
    }

    fig, (ax_dv, ax_sc) = plt.subplots(
        1, 2, figsize=(13, 5.2), constrained_layout=True,
    )

    # ---- (a) dV bar chart ----
    names = [m.name for m in missions]
    dvs = [m.dv_total_ms for m in missions]
    colors = [regime_color[m.regime] for m in missions]
    bars = ax_dv.bar(range(len(missions)), dvs, color=colors)
    # Mark qalunar with a hatch so it stands out.
    for m, bar in zip(missions, bars):
        if m.name.startswith("qalunar"):
            bar.set_hatch("////")
            bar.set_edgecolor("black")
            bar.set_linewidth(1.2)
    ax_dv.set_xticks(range(len(missions)))
    ax_dv.set_xticklabels([n.split(" (")[0].replace(" ", "\n", 1)
                           for n in names], fontsize=8)
    ax_dv.set_ylabel("Total transfer dV [m/s]")
    ax_dv.set_title("(a) dV budget to lunar orbit")
    for i, m in enumerate(missions):
        ax_dv.text(i, m.dv_total_ms + 60, f"{m.dv_total_ms:,.0f}",
                   ha="center", va="bottom", fontsize=8)
    ax_dv.grid(axis="y", alpha=0.3)

    # ---- (b) dV vs TOF scatter, log-log, sized by thrust-to-mass ----
    seen = set()
    for m in missions:
        c = regime_color[m.regime]
        lab = m.regime if m.regime not in seen else None
        seen.add(m.regime)
        size = min(60 + 240 * (m.accel_mm_s2 / 3.5), 360)   # area ~ a0, capped
        marker = "*" if m.name.startswith("qalunar") else "o"
        ax_sc.scatter(m.tof_days, m.dv_total_ms, s=size if marker == "o" else 420,
                      c=c, marker=marker, edgecolors="black",
                      linewidths=1.0, label=lab, zorder=3, alpha=0.9)
        # Default label to the right; nudge the two near-coincident points
        # (qalunar@SMART-1-pt and SMART-1) apart so both stay readable.
        if m.name.startswith("qalunar @ SMART-1"):
            ax_sc.annotate(m.name, (m.tof_days * 0.55, m.dv_total_ms - 230),
                           fontsize=8, va="center", ha="right")
        elif m.name.startswith("SMART-1"):
            ax_sc.annotate("SMART-1", (m.tof_days * 1.10, m.dv_total_ms + 120),
                           fontsize=8, va="center")
        else:
            ax_sc.annotate(m.name.split(" (")[0],
                           (m.tof_days * 1.06, m.dv_total_ms),
                           fontsize=8, va="center")
    ax_sc.set_xscale("log")
    ax_sc.set_xlabel("Time of flight [days, log]")
    ax_sc.set_ylabel("Total transfer dV [m/s]")
    ax_sc.set_title("(b) dV vs time of flight (marker = qalunar; size ~ a$_0$)")
    ax_sc.grid(True, which="both", alpha=0.3)
    ax_sc.set_xlim(2, 1100)
    ax_sc.set_ylim(0, 4500)
    ax_sc.legend(title="propulsion regime", loc="lower left")

    fig.suptitle(
        "qalunar all-binary QUBO mission vs flown Earth-Moon missions",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"\n  figure written to {out_path}")


def main() -> None:
    q = qalunar_budget()
    q_s1 = qalunar_smart1_point_budget()
    refs = reference_missions()
    missions = [q, q_s1] + refs

    _print_table(missions)
    smart1 = next(m for m in refs if m.name.startswith("SMART-1"))
    _print_smart1_focus(q, smart1)

    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(missions, out_dir / "mission_benchmark.csv")
    _plot(missions, out_dir / "mission_benchmark.png")
    print(f"  table written to {out_dir / 'mission_benchmark.csv'}")

    print()
    print("  Sources:")
    print("    SMART-1: en.wikipedia.org/wiki/SMART-1; "
          "esa.int SMART-1 operations; electricrocket.org/IEPC/245.pdf")
    print("    CAPSTONE: advancedspace.com/missions/capstone; BLT cheat sheet")
    print("    Apollo TLI/LOI: en.wikipedia.org/wiki/Trans-lunar_injection; "
          "en.wikipedia.org/wiki/Delta-v_budget")


if __name__ == "__main__":
    main()
