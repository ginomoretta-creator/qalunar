"""CONAE 12U CubeSat HEO acquisition: duty-cycle-limited perigee raising,
scheduled by the binary QUBO, GMAT-in-the-loop.

A power-limited CubeSat electric-propulsion mission analysis: a 40 mN
Hall-effect thruster raises the perigee of a highly eccentric disposal/transfer
ellipse by prograde burns at apogee, subject to a thrust window of at most 15%
of the orbital period (RQ-MIS-03) and an orbit-averaged power budget below
15 W (RQ-MIS-05). A tangential impulse at an apsis moves only the opposite
apsis, so perigee is raised by burning at apogee (and apogee, as on SMART-1,
by burning at perigee).

Two experiments, both flown in GMAT under real ephemerides:

  (a) Multi-apogee climb: fire the 40 mN HET at each of N_APOGEES apogees; the
      perigee climbs out of the disposal ellipse toward the operational orbit.
  (b) Duty-cycle slot selection: split one orbit into N equal-time slots,
      measure each slot's perigee-radius gain g_j in GMAT, and a cardinality
      QUBO (budget K = floor(0.15 N), sampled with simulated annealing and
      cross-checked against the closed-form top-K ground state) selects
      which slots fire. The selected schedule is then flown back in GMAT so
      the first-order superposition assumption is checked, not assumed.

Run:  python -m scripts.run_conae_duty_cycle
      python -m scripts.run_conae_duty_cycle --replot   (figure only, no GMAT)
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from qalunar.highfidelity import find_gmat_console

MU_EARTH = 398_600.4418            # km^3/s^2
EARTH_RADIUS_KM = 6_378.137

# CONAE reference ellipse (representative; not the true mission orbit).
# The apogee radius is the CONAE value. The injection perigee is set to a
# realistic upper-stage transfer altitude: the earlier a = 41646 km,
# e = 0.847 pair put the perigee at 6371 km radius, i.e. 7 km *below* the
# equatorial surface, where no vehicle survives a pass and no atmosphere
# model is defined. Changing it moves a by 0.3% and e by 0.7%.
PERIGEE_ALT_KM = 250.0
APOGEE_RADIUS_KM = 76_922.0
RP_KM = EARTH_RADIUS_KM + PERIGEE_ALT_KM
SMA_KM = 0.5 * (RP_KM + APOGEE_RADIUS_KM)
ECC = (APOGEE_RADIUS_KM - RP_KM) / (APOGEE_RADIUS_KM + RP_KM)
INC_DEG = 39.0
RAAN_DEG = 0.0
AOP_DEG = 0.0
TA0_DEG = 150.0                    # injection true anomaly (paper Table 3)

# 40 mN Hall thruster on the 12U bus: 11 kg platform + propulsion system =
# 13 kg dry, plus the xenon loaded for the whole cislunar mission. Phase 1
# therefore flies at the WET mass; the 8.05 m/s-per-apogee figure of the
# thruster trade (sized at 13 kg) is not reachable in 2626 s at 16.5 kg.
THRUST_N = 0.040
ISP_S = 1200.0
DRY_MASS_KG = 13.0
XE_LOADED_KG = 3.5
WET_MASS_KG = DRY_MASS_KG + XE_LOADED_KG
OPERATING_POWER_W = 493.5          # thruster operating power (thruster table)
G0 = 9.80665

# Atmospheric drag in the truth model (MSISE-90, mean solar activity). On:
# with a 250 km injection perigee the per-pass decay competes with the
# thrust-induced gain and must be in the truth model.
DRAG = True
APOGEE_BURN_S = 2626.0             # per-apogee burn duration (Table: 40 mN HET)
N_APOGEES = 5                      # 5 x 8 m/s = 40 m/s total (RQ-MIS-01)

N_SLOTS = 24
DUTY_CYCLE = 0.15                  # RQ-MIS-03: <= 15% of the period
N_BURNS = int(np.floor(DUTY_CYCLE * N_SLOTS))   # a ceiling: floor, never round

FIG_DIR = Path(__file__).resolve().parent / "figures"

# Light "publication" palette (consistent with the SMART-1 figures).
C_BG = "white"
C_EARTH = "#1f5fc0"
C_EARTH_GLOW = "#bcd4ff"
C_ORBIT = "#0e7c8b"
C_BURN = "#d9531e"
C_COAST = "#8a8f9c"
C_FAINT = "#c2c7d0"
C_TEXT = "#1e2430"


def _period() -> float:
    return 2.0 * np.pi * np.sqrt(SMA_KM ** 3 / MU_EARTH)


def _sat_header(thrust_n: float, has_burn: bool, ta_deg: float) -> list[str]:
    lines = [
        "Create CoordinateSystem ECI;",
        "ECI.Origin = Earth;",
        "ECI.Axes = MJ2000Eq;",
        "",
        "Create Spacecraft Sat;",
        "Sat.DateFormat = UTCGregorian;",
        "Sat.Epoch = '01 Jan 2026 00:00:00.000';",
        "Sat.CoordinateSystem = EarthMJ2000Eq;",
        "Sat.DisplayStateType = Keplerian;",
        f"Sat.SMA = {SMA_KM:.6f};",
        f"Sat.ECC = {ECC:.9f};",
        f"Sat.INC = {INC_DEG};",
        f"Sat.RAAN = {RAAN_DEG};",
        f"Sat.AOP = {AOP_DEG};",
        f"Sat.TA = {ta_deg};",
        f"Sat.DryMass = {DRY_MASS_KG};",
    ]
    if DRAG:
        lines += ["Sat.Cd = 2.2;", "Sat.DragArea = 0.12;"]
    if has_burn:
        lines += [
            "Sat.Tanks = {XeTank};",
            "Sat.Thrusters = {Hall};",
            "Sat.PowerSystem = SolarP;",
            "",
            "Create ElectricTank XeTank;",
            "XeTank.AllowNegativeFuelMass = false;",
            f"XeTank.FuelMass = {XE_LOADED_KG};",
            "",
            "Create ElectricThruster Hall;",
            "Hall.CoordinateSystem = Local;",
            "Hall.Origin = Earth;",
            "Hall.Axes = VNB;",
            "Hall.ThrustDirection1 = 1;",
            "Hall.ThrustDirection2 = 0;",
            "Hall.ThrustDirection3 = 0;",
            "Hall.DecrementMass = true;",
            "Hall.Tank = {XeTank};",
            "Hall.ThrustModel = ConstantThrustAndIsp;",
            f"Hall.ConstantThrust = {thrust_n:.6f};",
            f"Hall.Isp = {ISP_S};",
            "Hall.MaximumUsablePower = 2;",
            "Hall.MinimumUsablePower = 0.01;",
            "",
            "Create SolarPowerSystem SolarP;",
            # GMAT's power system gates the thruster on sunlight (DualCone
            # eclipses) but does not model a battery: the 493.5 W burn is drawn
            # from stored energy, and the orbit-averaged power budget
            # (RQ-MIS-05) is book-kept analytically from burn time and period.
            "SolarP.InitialMaxPower = 2;",
            "SolarP.AnnualDecayRate = 0;",
            "SolarP.Margin = 0;",
            "SolarP.ShadowModel = 'DualCone';",
            "SolarP.ShadowBodies = {Earth};",
            "",
            "Create FiniteBurn Burn;",
            "Burn.Thrusters = {Hall};",
        ]
    lines += [
        "",
        "Create ForceModel FM;",
        "FM.CentralBody = Earth;",
        "FM.PrimaryBodies = {Earth};",
        "FM.GravityField.Earth.Degree = 4;",
        "FM.GravityField.Earth.Order = 4;",
        "FM.PointMasses = {Luna, Sun};",
    ]
    if DRAG:
        lines += [
            "FM.Drag.AtmosphereModel = 'MSISE90';",
            "FM.Drag.F107 = 150;",
            "FM.Drag.F107A = 150;",
            "FM.Drag.MagneticIndex = 3;",
        ]
    lines += [
        "",
        "Create Propagator Prop;",
        "Prop.FM = FM;",
        "Prop.Type = RungeKutta89;",
        "Prop.InitialStepSize = 30;",
        "Prop.Accuracy = 1e-11;",
        "Prop.MinStep = 0;",
        "Prop.MaxStep = 600;",
        "",
    ]
    return lines


def _run(script_text: str, report: str, console: Path) -> np.ndarray:
    with tempfile.NamedTemporaryFile("w", suffix=".script", delete=False,
                                     encoding="ascii") as fh:
        fh.write(script_text)
        path = Path(fh.name)
    # One log file per run: concurrent GmatConsole instances that share the
    # default GmatLog.txt fail at start-up ("specified log file is not a
    # valid log file").
    log = path.with_suffix(".log")
    try:
        proc = subprocess.run([str(console), "-l", str(log), "-r", str(path)],
                              cwd=str(console.parent),
                              capture_output=True, text=True, timeout=600)
        raw = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0 or not re.search(r"Mission run completed", raw):
            raise RuntimeError("GMAT failed:\n" + "\n".join(raw.splitlines()[-18:]))
        for d in (console.parent.parent / "output", console.parent / "output",
                  console.parent):
            p = d / report
            if p.exists():
                rows = [[float(v) for v in ln.split()]
                        for ln in p.read_text().strip().splitlines() if ln.strip()]
                p.unlink(missing_ok=True)
                return np.asarray(rows)
        raise FileNotFoundError(report)
    finally:
        path.unlink(missing_ok=True)
        log.unlink(missing_ok=True)


def _coast_profile_script(report: str) -> str:
    dt = _period() / N_SLOTS
    lines = ["% CONAE coast profile (per-slot TA / speed / perigee)", ""]
    lines += _sat_header(0.0, False, TA0_DEG)
    lines += [f"Create ReportFile Rep;", f"Rep.Filename = '{report}';",
              "Rep.Precision = 12;", "Rep.WriteHeaders = false;", "",
              "BeginMissionSequence;", ""]
    for _ in range(N_SLOTS):
        lines.append("Report Rep Sat.Earth.TA Sat.ECI.VMAG Sat.Earth.RadPer "
                     "Sat.Earth.RMAG;")
        lines.append(f"Propagate Prop(Sat) {{Sat.ElapsedSecs = {dt:.6f}}};")
    return "\n".join(lines) + "\n"


def _slot_fly_script(schedule: np.ndarray, report: str) -> str:
    dt = _period() / schedule.size
    lines = ["% CONAE duty-cycle slot schedule", ""]
    lines += _sat_header(THRUST_N, bool(schedule.any()), TA0_DEG)
    lines += [f"Create ReportFile Rep;", f"Rep.Filename = '{report}';",
              "Rep.Precision = 12;", "Rep.WriteHeaders = false;", "",
              "BeginMissionSequence;", ""]
    i, n = 0, schedule.size
    while i < n:
        if schedule[i] == 1:
            lines += ["BeginFiniteBurn Burn(Sat);",
                      f"Propagate Prop(Sat) {{Sat.ElapsedSecs = {dt:.6f}}};",
                      "EndFiniteBurn Burn(Sat);"]
            i += 1
        else:
            j = i
            while j < n and schedule[j] == 0:
                j += 1
            lines.append(f"Propagate Prop(Sat) {{Sat.ElapsedSecs = {(j - i) * dt:.6f}}};")
            i = j
    lines.append("Report Rep Sat.Earth.RadPer Sat.Earth.RadApo Sat.Earth.SMA;")
    return "\n".join(lines) + "\n"


def _climb_script(report: str) -> str:
    """Fire the 40 mN HET at each of N_APOGEES apogees; report after each burn
    and, at the end, at the following perigee passage (true altitude)."""
    lines = ["% CONAE multi-apogee perigee climb", ""]
    lines += _sat_header(THRUST_N, True, TA0_DEG)
    lines += [f"Create ReportFile Rep;", f"Rep.Filename = '{report}';",
              "Rep.Precision = 12;", "Rep.WriteHeaders = false;", "",
              "BeginMissionSequence;", "",
              "Report Rep Sat.Earth.RadPer Sat.Earth.RadApo Sat.ECI.VMAG "
              "Sat.XeTank.FuelMass Sat.ElapsedDays Sat.Earth.Altitude;"]
    for _ in range(N_APOGEES):
        lines += [
            "Propagate Prop(Sat) {Sat.Earth.Apoapsis};",
            "BeginFiniteBurn Burn(Sat);",
            f"Propagate Prop(Sat) {{Sat.ElapsedSecs = {APOGEE_BURN_S:.4f}}};",
            "EndFiniteBurn Burn(Sat);",
            "Report Rep Sat.Earth.RadPer Sat.Earth.RadApo Sat.ECI.VMAG "
            "Sat.XeTank.FuelMass Sat.ElapsedDays Sat.Earth.Altitude;",
        ]
    # Final perigee passage: the altitude actually flown through.
    lines += [
        "Propagate Prop(Sat) {Sat.Earth.Periapsis};",
        "Report Rep Sat.Earth.RadPer Sat.Earth.RadApo Sat.ECI.VMAG "
        "Sat.XeTank.FuelMass Sat.ElapsedDays Sat.Earth.Altitude;",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replot", action="store_true",
                    help="regenerate the figure from the cached CSVs (no GMAT)")
    args = ap.parse_args()
    if args.replot:
        _replot()
        return

    console = find_gmat_console()
    if not console.exists():
        raise SystemExit(f"GmatConsole not found at {console}")

    period = _period()
    rp0 = SMA_KM * (1 - ECC)
    ra0 = SMA_KM * (1 + ECC)
    print("=" * 90)
    print("  CONAE 12U CubeSat: duty-cycle-limited perigee raising (GMAT + binary QUBO)")
    print("=" * 90)
    print(f"  reference ellipse  a={SMA_KM:.0f} km  e={ECC:.4f}  "
          f"rp={rp0:.0f} km ({rp0-EARTH_RADIUS_KM:.0f} km alt)  ra={ra0:.0f} km  "
          f"period={period/3600:.2f} h  injection TA={TA0_DEG:.0f} deg  "
          f"drag={'on' if DRAG else 'off'}")
    slot_s = period / N_SLOTS
    print(f"  40 mN HET, Isp {ISP_S:.0f} s, {WET_MASS_KG:.1f} kg wet at start of "
          f"Phase 1; duty budget {DUTY_CYCLE:.0%} -> K = floor({DUTY_CYCLE}*{N_SLOTS})"
          f" = {N_BURNS} slots = {N_BURNS/N_SLOTS:.1%} of the period "
          f"({N_BURNS*slot_s:.0f} s; orbit-averaged power "
          f"{OPERATING_POWER_W*N_BURNS*slot_s/period:.1f} W)\n")

    # ---- (b) per-slot profile + perigee-gain g_j ----
    prof = _run(_coast_profile_script("conae_prof.txt"), "conae_prof.txt", console)
    ta_deg, v_kms, rper_coast_prof, rmag = prof.T

    print(f"  measuring per-slot perigee gain in GMAT ({N_SLOTS} runs) ...",
          flush=True)
    rper_coast = float(_run(_slot_fly_script(np.zeros(N_SLOTS, dtype=np.int64),
                                             "conae_coast.txt"),
                            "conae_coast.txt", console)[-1][0])

    def _gain(j: int) -> float:
        q = np.zeros(N_SLOTS, dtype=np.int64)
        q[j] = 1
        rper_j = float(_run(_slot_fly_script(q, f"conae_slot{j}.txt"),
                            f"conae_slot{j}.txt", console)[-1][0])
        return rper_j - rper_coast

    with ThreadPoolExecutor(max_workers=6) as pool:
        g = np.array(list(pool.map(_gain, range(N_SLOTS))))

    apo_slot = int(np.argmax(g))
    print(f"  largest perigee gain at slot {apo_slot} (TA {ta_deg[apo_slot]:.0f} deg, "
          f"v {v_kms[apo_slot]:.2f} km/s): {g[apo_slot]:.1f} km vs "
          f"{g.min():.1f} km at perigee\n")

    # Cardinality QUBO, sampled with simulated annealing and cross-checked
    # against the closed-form top-K ground state (for a linear objective the
    # two must coincide; the QUBO form is what an annealer ingests).
    from qalunar.qubo.cardinality import solve_cardinality_qubo
    q_qubo, qinfo = solve_cardinality_qubo(g, N_BURNS, num_reads=5000, seed=1)
    on = np.flatnonzero(q_qubo)
    print(f"  QUBO (SA, {qinfo['num_reads']} reads, penalty {qinfo['penalty']:.3g}, "
          f"coefficient range {qinfo['coefficient_range']:.2g}) selected "
          f"{int(q_qubo.sum())} slots, true anomalies "
          f"{sorted(f'{t:+.0f}' for t in (ta_deg[on]))} deg (180 = apogee); "
          f"matches top-K: {qinfo['matches_top_k']}")

    # Validate the chosen combination in the truth model: the QUBO assumed
    # the per-slot gains superpose; measure how well they actually do.
    rper_sched = float(_run(_slot_fly_script(q_qubo, "conae_sched.txt"),
                            "conae_sched.txt", console)[-1][0])
    gain_predicted = float(g[q_qubo == 1].sum())
    gain_measured = rper_sched - rper_coast
    print(f"  flown back in GMAT: perigee gain {gain_measured:.1f} km measured vs "
          f"{gain_predicted:.1f} km predicted by superposition "
          f"({100*(gain_measured-gain_predicted)/gain_predicted:+.1f}%)\n")

    # ---- multi-apogee perigee climb ----
    print(f"  flying the {N_APOGEES}-apogee perigee climb in GMAT ...", flush=True)
    climb_all = _run(_climb_script("conae_climb.txt"), "conae_climb.txt", console)
    climb = climb_all[:N_APOGEES + 1]          # rows after each apogee burn
    final_perigee_alt = float(climb_all[-1, 5])  # altitude at the last perigee pass
    rper_climb = climb[:, 0]
    fuel_climb = climb[:, 3]
    m_climb = DRY_MASS_KG + fuel_climb
    # Delta-v measured from the truth model's propellant depletion (rocket
    # equation), never assumed from the thruster table.
    dv_pass = ISP_S * G0 * np.log(m_climb[:-1] / m_climb[1:])
    dv_total = float(dv_pass.sum())
    print(f"  perigee {rper_climb[0]:.0f} -> {rper_climb[-1]:.0f} km "
          f"(+{rper_climb[-1]-rper_climb[0]:.0f} km) over {N_APOGEES} apogees; "
          f"measured dv {dv_total:.1f} m/s total "
          f"({dv_pass.min():.2f}-{dv_pass.max():.2f} m/s per pass; RQ-MIS-02 needs "
          f">= 8), xenon {fuel_climb[0]-fuel_climb[-1]:.4f} kg, "
          f"{climb[-1, 4]:.2f} d; orbit-averaged power per pass "
          f"{OPERATING_POWER_W*APOGEE_BURN_S/period:.1f} W; altitude at the "
          f"final perigee passage {final_perigee_alt:.0f} km\n")

    _save_csvs(ta_deg, v_kms, g, q_qubo, rper_climb, dv_pass, fuel_climb,
               qinfo, gain_predicted, gain_measured)
    _draw(ta_deg, g, q_qubo, rper_climb, dv_pass)


def _save_csvs(ta_deg, v_kms, g, q_qubo, rper_climb, dv_pass=None,
               fuel_climb=None, qinfo=None, gain_predicted=None,
               gain_measured=None) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if qinfo is not None:
        with (FIG_DIR / "conae_duty_validation.csv").open(
                "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["key", "value"])
            for k, v in qinfo.items():
                w.writerow([k, v])
            w.writerow(["n_slots", N_SLOTS])
            w.writerow(["duty_budget", DUTY_CYCLE])
            w.writerow(["k", N_BURNS])
            w.writerow(["duty_flown", N_BURNS / N_SLOTS])
            w.writerow(["wet_mass_kg", WET_MASS_KG])
            w.writerow(["gain_predicted_km", f"{gain_predicted:.4f}"])
            w.writerow(["gain_measured_km", f"{gain_measured:.4f}"])
    with (FIG_DIR / "conae_duty_cycle.csv").open("w", newline="",
                                                 encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["slot", "true_anomaly_deg", "speed_kms",
                    "perigee_gain_km", "qubo_fires"])
        for i in range(len(ta_deg)):
            w.writerow([i, f"{ta_deg[i]:.3f}", f"{v_kms[i]:.6f}",
                        f"{g[i]:.4f}", int(q_qubo[i])])
    with (FIG_DIR / "conae_climb.csv").open("w", newline="",
                                            encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["apogee_pass", "perigee_radius_km", "fuel_kg", "dv_pass_m_s"])
        for i, rp in enumerate(rper_climb):
            fuel = "" if fuel_climb is None else f"{fuel_climb[i]:.6f}"
            dv = "" if (dv_pass is None or i == 0) else f"{dv_pass[i-1]:.4f}"
            w.writerow([i, f"{rp:.4f}", fuel, dv])


def _replot() -> None:
    duty = FIG_DIR / "conae_duty_cycle.csv"
    climb = FIG_DIR / "conae_climb.csv"
    if not duty.exists() or not climb.exists():
        raise SystemExit("cached CSVs not found; run once without --replot first.")
    ta, g, q = [], [], []
    with duty.open(newline="", encoding="utf-8") as fh:
        rd = csv.reader(fh)
        next(rd, None)
        for row in rd:
            if row:
                ta.append(float(row[1])); g.append(float(row[3]))
                q.append(int(row[4]))
    rper, dvs = [], []
    with climb.open(newline="", encoding="utf-8") as fh:
        rd = csv.reader(fh)
        next(rd, None)
        for row in rd:
            if row:
                rper.append(float(row[1]))
                if len(row) > 3 and row[3]:
                    dvs.append(float(row[3]))
    _draw(np.asarray(ta), np.asarray(g), np.asarray(q), np.asarray(rper),
          np.asarray(dvs) if dvs else None)


def _draw(ta_deg, g, q_qubo, rper_climb, dv_pass=None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    on = np.asarray(q_qubo, dtype=bool)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 5.2))
    fig.patch.set_facecolor(C_BG)

    # ---------- (a) perigee climb ----------
    alt = rper_climb - EARTH_RADIUS_KM
    passes = np.arange(rper_climb.size)
    axA.set_facecolor(C_BG)
    axA.grid(True, color=C_FAINT, lw=0.5, alpha=0.6)
    axA.set_axisbelow(True)
    axA.plot(passes, alt, "-o", color=C_ORBIT, lw=1.8, ms=7,
             markerfacecolor=C_BURN, markeredgecolor=C_TEXT, markeredgewidth=0.6)
    axA.axhline(0.0, color=C_FAINT, lw=1.0, ls="--")
    axA.annotate("Earth surface", (passes[-1], 0.0), textcoords="offset points",
                 xytext=(-4, 6), color=C_TEXT, fontsize=8, ha="right", alpha=0.8)
    axA.set_xlabel("apogee burn #")
    axA.set_ylabel("perigee altitude (km)")
    if dv_pass is not None and len(dv_pass):
        dv_txt = (f"{len(dv_pass)} apogee burns, {np.mean(dv_pass):.1f} m/s each "
                  f"({np.sum(dv_pass):.0f} m/s total, measured)")
    else:
        dv_txt = f"{N_APOGEES} apogee burns of {APOGEE_BURN_S:.0f} s"
    axA.set_title("(a)  Perigee raised out of the disposal ellipse\n" + dv_txt,
                  color=C_TEXT, fontsize=10.5, pad=8)
    axA.set_xticks(passes)
    for s in axA.spines.values():
        s.set_color("#9aa0ab")
    axA.tick_params(colors=C_TEXT, labelsize=8)
    axA.xaxis.label.set_color(C_TEXT); axA.yaxis.label.set_color(C_TEXT)

    # ---------- (b) slot selection on the orbit ----------
    p = SMA_KM * (1.0 - ECC ** 2)
    nu = np.linspace(0.0, 2.0 * np.pi, 720)
    r = p / (1.0 + ECC * np.cos(nu))
    ex, ey = r * np.cos(nu) * 1e-3, r * np.sin(nu) * 1e-3
    nus = np.deg2rad(np.asarray(ta_deg, dtype=float))
    rs = p / (1.0 + ECC * np.cos(nus))
    sx, sy = rs * np.cos(nus) * 1e-3, rs * np.sin(nus) * 1e-3

    axB.set_facecolor(C_BG)
    axB.set_aspect("equal")
    axB.grid(True, color=C_FAINT, lw=0.5, alpha=0.6)
    axB.set_axisbelow(True)
    for k in (2, 1):
        axB.plot(ex, ey, color=C_ORBIT, lw=1.2 + 2.0 * k, alpha=0.05,
                 solid_capstyle="round", zorder=2)
    axB.plot(ex, ey, color=C_ORBIT, lw=1.2, zorder=3)
    axB.add_patch(plt.Circle((0, 0), EARTH_RADIUS_KM * 1e-3 * 1.9,
                             color=C_EARTH_GLOW, alpha=0.18, lw=0, zorder=4))
    axB.add_patch(plt.Circle((0, 0), EARTH_RADIUS_KM * 1e-3, color=C_EARTH,
                             lw=0, zorder=5))
    axB.scatter(sx[~on], sy[~on], s=46, color=C_COAST, edgecolors="white",
                lw=0.6, zorder=6, label=f"coast slot ({int((~on).sum())})")
    for xx, yy in zip(sx[on], sy[on]):
        axB.scatter([xx], [yy], s=340, color=C_BURN, alpha=0.14, lw=0, zorder=6)
        axB.scatter([xx], [yy], s=170, color=C_BURN, alpha=0.20, lw=0, zorder=6)
    axB.scatter(sx[on], sy[on], s=96, color=C_BURN, edgecolors=C_TEXT, lw=0.6,
                zorder=8, label=f"QUBO fires here ({int(on.sum())})")
    rp, ra = SMA_KM * (1 - ECC) * 1e-3, SMA_KM * (1 + ECC) * 1e-3
    axB.annotate("apogee\n(raise perigee here)", (-ra, 0),
                 textcoords="offset points", xytext=(14, 14), color=C_TEXT,
                 fontsize=9, ha="left")
    axB.annotate("perigee", (rp, 0), textcoords="offset points",
                 xytext=(8, 8), color=C_TEXT, fontsize=9, alpha=0.85)
    axB.set_title("(b)  The QUBO fires at apogee to raise perigee\n"
                  f"{int(on.sum())} of {on.size} slots = {on.sum()/on.size:.1%} duty "
                  f"(budget <= {DUTY_CYCLE:.0%}; tangential burn at an apsis "
                  "moves the opposite apsis)",
                  color=C_TEXT, fontsize=10.5, pad=8)
    axB.set_xlabel("x  ($10^3$ km, perifocal — Earth at focus)")
    axB.set_ylabel("y  ($10^3$ km)")
    for s in axB.spines.values():
        s.set_color("#9aa0ab")
    axB.tick_params(colors=C_TEXT, labelsize=8)
    axB.xaxis.label.set_color(C_TEXT); axB.yaxis.label.set_color(C_TEXT)
    leg = axB.legend(loc="upper left", fontsize=9, framealpha=0.0)
    for t in leg.get_texts():
        t.set_color(C_TEXT)
    b = SMA_KM * np.sqrt(1 - ECC ** 2) * 1e-3
    pad = 0.12 * (ra + rp)
    axB.set_xlim(-ra - pad, rp + 0.30 * ra + pad)
    axB.set_ylim(-b - pad, b + pad)

    fig.tight_layout(pad=1.4)
    out = FIG_DIR / "conae_duty_cycle.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=C_BG, bbox_inches="tight")
    print(f"  figure written to {out}")


if __name__ == "__main__":
    main()
