"""Full CONAE 12U CubeSat cislunar mission as a single continuous GMAT trace,
rendered as a publication trajectory figure (light, OrbitView-like).

  Phase 1  raise the perigee out of the transfer ellipse with apogee burns;
  Phase 2  raise the apogee to lunar distance with continuous tangential
           thrust (the spiral);
  Phase 3  capture by apse-targeted anti-tangential braking arcs (Moon VNB -V):
           bind at the arrival periselene with the aposelene inside the Hill
           sphere, then lower the periselene at aposelene and the aposelene at
           periselene; then a 50-day coast.

This is the continuous-thrust (100 % duty, 493.5 W) reference of the paper,
not a mission design: the 12U power budget cannot supply it.

The departure epoch sets the phasing of the lunar encounter. A 2-h scan over
the two encounter windows of this spiral (23-24 Jan and 5-7 Feb 2026) gives a
minimum periselene of ~31,600 km; 23 Jan 22:00 is the slowest arrival
(E ~ -0.002 km^2/s^2 at 31,618 km). A single long brake from periselene kills
the angular momentum and drives the periselene below the surface; braking at
the apses keeps it up.

Flown headless in GMAT (DE405, Earth 4x4 + Sun + Moon); the full ephemeris is
logged and cached so the figure can be regenerated with --replot (no GMAT).

Run:  python -m scripts.run_conae_lunar_trajectory
      python -m scripts.run_conae_lunar_trajectory --replot
"""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from qalunar.highfidelity import find_gmat_console
from scripts.run_smart1_trajectory_figure import (
    C_BG, C_EARTH, C_EARTH_GLOW, C_MOON, C_SPIRAL, C_CAPTURE, C_FAINT, C_TEXT,
    _glow, _disk, _style_axes,
)

FIG_DIR = Path(__file__).resolve().parent / "figures"
EARTH_RADIUS_KM = 6_378.137
MOON_RADIUS_KM = 1_737.4
MOON_SOI_KM = 66_100.0
G0 = 9.80665

# Same reference ellipse as run_conae_duty_cycle: CONAE apogee radius, 250 km
# injection perigee (the earlier a/e pair put the perigee below the surface).
PERIGEE_ALT_KM, APOGEE_RADIUS_KM = 250.0, 76_922.0
_RP = EARTH_RADIUS_KM + PERIGEE_ALT_KM
SMA_KM = 0.5 * (_RP + APOGEE_RADIUS_KM)
ECC = (APOGEE_RADIUS_KM - _RP) / (APOGEE_RADIUS_KM + _RP)
INC_DEG = 39.0
TA0_DEG = 150.0                                 # injection true anomaly (paper Table 3)
THRUST_N, ISP_S, DRY_MASS_KG, XE_KG = 0.040, 1200.0, 13.0, 3.5
OPERATING_POWER_W = 493.5
APOGEE_BURN_S = 2626.0
N_PHASE1 = 5                                    # five apogee passes, as in the text
APO_TARGET_KM = 370_000.0
ENCOUNTER_EPOCH = "23 Jan 2026 22:00:00.000"   # departure = encounter phasing (see docstring)
RA_BIND_KM = 40_000.0                           # bind with the aposelene inside the Hill sphere
RP_TARGET_KM = 12_000.0                         # lower the periselene to this at aposelene ...
RA_TARGET_KM = 20_000.0                         # ... and the aposelene to this at periselene
N_CAPTURE_REVS = 6
POST_COAST_S = 50.0 * 86400.0                   # 50 d post-capture coast (the stability claim)


def _burn_until(param: str, op: str, value: float, max_iter: int = 300) -> str:
    """GMAT block: brake in 600 s steps until `param op value` (fuel/iteration guards)."""
    return f"""   BeginFiniteBurn Brake(Sat);
   goflag = 1;
   niter = 0;
   While goflag > 0.5
      Propagate Prop(Sat) {{Sat.ElapsedSecs = 600}};
      niter = niter + 1;
      If {param} {op} {value:.1f}
         goflag = 0;
      EndIf
      If Sat.XeTank.FuelMass < 0.03
         goflag = 0;
      EndIf
      If niter > {max_iter}
         goflag = 0;
      EndIf
   EndWhile
   EndFiniteBurn Brake(Sat);"""


def build_script(report: str) -> str:
    return f"""% Auto-generated: CONAE 12U CubeSat -> Moon, single continuous trace.
Create CoordinateSystem MoonView;
MoonView.Origin = Luna;
MoonView.Axes = MJ2000Eq;

Create Spacecraft Sat;
Sat.DateFormat = UTCGregorian;
Sat.Epoch = '{ENCOUNTER_EPOCH}';
Sat.CoordinateSystem = EarthMJ2000Eq;
Sat.DisplayStateType = Keplerian;
Sat.SMA = {SMA_KM:.6f};
Sat.ECC = {ECC:.9f};
Sat.INC = {INC_DEG};
Sat.RAAN = 0;
Sat.AOP = 0;
Sat.TA = {TA0_DEG};
Sat.DryMass = {DRY_MASS_KG};
Sat.Tanks = {{XeTank}};
Sat.Thrusters = {{Hall, Brk}};
Sat.PowerSystem = SolarP;

Create ElectricTank XeTank;
XeTank.AllowNegativeFuelMass = false;
XeTank.FuelMass = {XE_KG};

Create ElectricThruster Hall;
Hall.CoordinateSystem = Local;
Hall.Origin = Earth;
Hall.Axes = VNB;
Hall.ThrustDirection1 = 1;
Hall.ThrustDirection2 = 0;
Hall.ThrustDirection3 = 0;
Hall.DecrementMass = true;
Hall.Tank = {{XeTank}};
Hall.ThrustModel = ConstantThrustAndIsp;
Hall.ConstantThrust = {THRUST_N:.6f};
Hall.Isp = {ISP_S};
Hall.MaximumUsablePower = 2;
Hall.MinimumUsablePower = 0.01;

Create ElectricThruster Brk;
Brk.CoordinateSystem = Local;
Brk.Origin = Luna;
Brk.Axes = VNB;
Brk.ThrustDirection1 = -1;
Brk.ThrustDirection2 = 0;
Brk.ThrustDirection3 = 0;
Brk.DecrementMass = true;
Brk.Tank = {{XeTank}};
Brk.ThrustModel = ConstantThrustAndIsp;
Brk.ConstantThrust = {THRUST_N:.6f};
Brk.Isp = {ISP_S};
Brk.MaximumUsablePower = 2;
Brk.MinimumUsablePower = 0.01;

Create SolarPowerSystem SolarP;
SolarP.InitialMaxPower = 1.9;
SolarP.AnnualDecayRate = 0;
SolarP.Margin = 0;
SolarP.ShadowModel = 'DualCone';
SolarP.ShadowBodies = {{Earth}};

Create FiniteBurn Spiral;
Spiral.Thrusters = {{Hall}};
Create FiniteBurn Brake;
Brake.Thrusters = {{Brk}};

Create ForceModel FM;
FM.CentralBody = Earth;
FM.PrimaryBodies = {{Earth}};
FM.GravityField.Earth.Degree = 4;
FM.GravityField.Earth.Order = 4;
FM.PointMasses = {{Luna, Sun}};

Create Propagator Prop;
Prop.FM = FM;
Prop.Type = RungeKutta89;
Prop.InitialStepSize = 60;
Prop.Accuracy = 1e-9;
Prop.MinStep = 0;
Prop.MaxStep = 1800;

Create ReportFile Eph;
Eph.Filename = '{report}';
Eph.Precision = 10;
Eph.WriteHeaders = false;
Eph.Add = {{Sat.ElapsedDays, Sat.EarthMJ2000Eq.X, Sat.EarthMJ2000Eq.Y, Sat.EarthMJ2000Eq.Z, Sat.MoonView.X, Sat.MoonView.Y, Sat.MoonView.Z, Sat.XeTank.FuelMass}};

Create ReportFile Mile;
Mile.Filename = 'conae_lunar_milestones.txt';
Mile.Precision = 10;
Mile.WriteHeaders = false;

Create OrbitView EarthView;
EarthView.SolverIterations = Current;
EarthView.UpperLeft = [0.01 0.01];
EarthView.Size = [0.6 0.95];
EarthView.Add = {{Sat, Earth, Luna}};
EarthView.CoordinateSystem = EarthMJ2000Eq;
EarthView.DrawObject = [true true true];
EarthView.ViewPointReference = Earth;
EarthView.ViewPointVector = [0 0 1100000];
EarthView.ViewDirection = Earth;
EarthView.ViewScaleFactor = 1;
EarthView.ViewUpCoordinateSystem = EarthMJ2000Eq;
EarthView.ViewUpAxis = X;
EarthView.Axes = On;
EarthView.XYPlane = On;

Create OrbitView MoonViewPort;
MoonViewPort.SolverIterations = Current;
MoonViewPort.UpperLeft = [0.62 0.01];
MoonViewPort.Size = [0.37 0.95];
MoonViewPort.Add = {{Sat, Luna}};
MoonViewPort.CoordinateSystem = MoonView;
MoonViewPort.DrawObject = [true true];
MoonViewPort.ViewPointReference = Luna;
MoonViewPort.ViewPointVector = [0 0 120000];
MoonViewPort.ViewDirection = Luna;
MoonViewPort.ViewScaleFactor = 1;
MoonViewPort.ViewUpCoordinateSystem = MoonView;
MoonViewPort.ViewUpAxis = X;
MoonViewPort.Axes = On;
MoonViewPort.XYPlane = Off;

Create Variable goflag niter kk;

BeginMissionSequence;

% --- Phase 1: raise perigee with apogee burns ---
For niter = 1:{N_PHASE1};
   Propagate Prop(Sat) {{Sat.Earth.Apoapsis}};
   BeginFiniteBurn Spiral(Sat);
   Propagate Prop(Sat) {{Sat.ElapsedSecs = {APOGEE_BURN_S:.1f}}};
   EndFiniteBurn Spiral(Sat);
EndFor;
Report Mile Sat.ElapsedDays Sat.Earth.RadPer Sat.Earth.RadApo Sat.Luna.RMAG Sat.Luna.Energy Sat.XeTank.FuelMass;

% --- Phase 2: continuous spiral until apogee reaches lunar distance ---
BeginFiniteBurn Spiral(Sat);
goflag = 1;
niter = 0;
While goflag > 0.5
   Propagate Prop(Sat) {{Sat.ElapsedSecs = 14400}};
   niter = niter + 1;
   goflag = 0;
   If Sat.Earth.RadApo < {APO_TARGET_KM:.1f}
      goflag = 1;
   EndIf
   If niter > 90
      goflag = 0;
   EndIf
EndWhile
EndFiniteBurn Spiral(Sat);
Report Mile Sat.ElapsedDays Sat.Earth.RadPer Sat.Earth.RadApo Sat.Luna.RMAG Sat.Luna.Energy Sat.XeTank.FuelMass;

% --- coast to the lunar encounter (stop at SOI entry) ---
goflag = 1;
niter = 0;
While goflag > 0.5
   Propagate Prop(Sat) {{Sat.ElapsedSecs = 10800}};
   niter = niter + 1;
   goflag = 0;
   If Sat.Luna.RMAG > {MOON_SOI_KM:.1f}
      goflag = 1;
   EndIf
   If niter > 200
      goflag = 0;
   EndIf
EndWhile
Report Mile Sat.ElapsedDays Sat.Earth.RadPer Sat.Earth.RadApo Sat.Luna.RMAG Sat.Luna.Energy Sat.XeTank.FuelMass;

% --- coast to the arrival periselene ---
Propagate Prop(Sat) {{Sat.Luna.Periapsis, Sat.ElapsedSecs = 400000}};
Report Mile Sat.ElapsedDays Sat.Luna.RMAG Sat.Luna.Energy Sat.XeTank.FuelMass;

% --- Phase 3: capture by apse-targeted braking arcs (Moon VNB -V) ---
% (1) bind at the arrival periselene, aposelene inside the Hill sphere
   BeginFiniteBurn Brake(Sat);
   goflag = 1;
   niter = 0;
   While goflag > 0.5
      Propagate Prop(Sat) {{Sat.ElapsedSecs = 600}};
      niter = niter + 1;
      If Sat.Luna.Energy < 0
         If Sat.Luna.RadApo < {RA_BIND_KM:.1f}
            If Sat.Luna.RadApo > 0
               goflag = 0;
            EndIf
         EndIf
      EndIf
      If Sat.XeTank.FuelMass < 0.03
         goflag = 0;
      EndIf
      If niter > 400
         goflag = 0;
      EndIf
   EndWhile
   EndFiniteBurn Brake(Sat);
Report Mile Sat.ElapsedDays Sat.Luna.RMAG Sat.Luna.Energy Sat.Luna.RadPer Sat.Luna.RadApo Sat.XeTank.FuelMass;
% (2) lower the periselene at aposelene and the aposelene at periselene
For kk = 1:{N_CAPTURE_REVS}
   Propagate Prop(Sat) {{Sat.Luna.Apoapsis, Sat.ElapsedSecs = 2500000}};
   If Sat.Luna.RadPer > {RP_TARGET_KM:.1f}
{_burn_until("Sat.Luna.RadPer", "<", RP_TARGET_KM)}
   EndIf
   Propagate Prop(Sat) {{Sat.Luna.Periapsis, Sat.ElapsedSecs = 2500000}};
   If Sat.Luna.RadApo > {RA_TARGET_KM:.1f}
{_burn_until("Sat.Luna.RadApo", "<", RA_TARGET_KM)}
   EndIf
   Report Mile Sat.ElapsedDays Sat.Luna.RMAG Sat.Luna.Energy Sat.Luna.RadPer Sat.Luna.RadApo Sat.XeTank.FuelMass;
EndFor

% --- 50-day post-capture coast ---
Propagate Prop(Sat) {{Sat.ElapsedSecs = {POST_COAST_S:.1f}}};
Report Mile Sat.ElapsedDays Sat.Luna.RMAG Sat.Luna.Energy Sat.Luna.RadPer Sat.Luna.RadApo Sat.XeTank.FuelMass;
"""


def _generate() -> np.ndarray:
    console = find_gmat_console()
    if not console.exists():
        raise SystemExit(f"GmatConsole not found at {console}")
    report = "conae_lunar_ephem.txt"
    with tempfile.NamedTemporaryFile("w", suffix=".script", delete=False,
                                     encoding="ascii") as fh:
        fh.write(build_script(report))
        tmp = Path(fh.name)
    print("  flying CONAE CubeSat -> Moon in GMAT (spiral + capture) ...", flush=True)
    try:
        proc = subprocess.run([str(console), str(tmp)], cwd=str(console.parent),
                              capture_output=True, text=True, timeout=1800)
        raw = (proc.stdout or "") + (proc.stderr or "")
        if not re.search(r"Mission run completed", raw):
            raise SystemExit("GMAT failed:\n" + "\n".join(raw.splitlines()[-25:]))
        for d in (console.parent.parent / "output", console.parent / "output",
                  console.parent):
            mp = d / "conae_lunar_milestones.txt"
            if mp.exists():
                print("  milestones (day | ... ):")
                for ln in mp.read_text().strip().splitlines():
                    print("    ", ln)
                mp.unlink(missing_ok=True)
            p = d / report
            if p.exists():
                rows = [[float(v) for v in ln.split()]
                        for ln in p.read_text().strip().splitlines() if ln.strip()]
                p.unlink(missing_ok=True)
                return np.asarray(rows)
        raise SystemExit(f"ephemeris {report} not found")
    finally:
        tmp.unlink(missing_ok=True)


def _encounter(rm: np.ndarray) -> tuple[int, int]:
    """Index of SOI entry and of the arrival periselene (first minimum after it)."""
    i_soi = int(np.argmax(rm < MOON_SOI_KM))
    k = int(np.argmax(np.diff(rm[i_soi:]) > 0))
    return i_soi, i_soi + k


def _summary(data: np.ndarray) -> None:
    """Numbers quoted in the paper, read from the cached ephemeris."""
    if data.shape[1] < 8:
        return
    days, rm, fuel = data[:, 0], np.linalg.norm(data[:, 4:7], axis=1), data[:, 7]
    i_soi, enc = _encounter(rm)
    burning = np.r_[np.diff(fuel) < -1e-9, False]
    idx = np.flatnonzero(burning[enc:]) + enc
    fb, lb = int(idx[0]), int(idx[-1]) + 1
    mdot = THRUST_N / (ISP_S * G0)
    xe = fuel[fb] - fuel[lb]
    dv = ISP_S * G0 * np.log((DRY_MASS_KG + fuel[fb]) / (DRY_MASS_KG + fuel[lb]))
    on_h = xe / mdot / 3600.0
    coast = days >= days[-1] - POST_COAST_S / 86400.0      # the final 50-day coast
    rc, dc = rm[coast], days[coast]
    mins = np.flatnonzero((rc[1:-1] < rc[:-2]) & (rc[1:-1] < rc[2:])) + 1
    spiral_idx = np.flatnonzero(burning[:i_soi])
    span_d = days[lb] - days[fb]
    allowed_h = span_d * 24.0 * 15.0 / OPERATING_POWER_W   # thrust time the 15 W budget allows
    print(f"  arrival periselene {rm[enc]:,.0f} km at day {days[enc]:.2f}")
    print(f"  capture: {span_d:.2f} d from first to last braking arc, thrust-on "
          f"{on_h:.1f} h, dv {dv:.0f} m/s, xenon {xe:.3f} kg, "
          f"{on_h * OPERATING_POWER_W / 1000:.1f} kWh; fuel left {fuel[lb]:.3f} kg; "
          f"{on_h / allowed_h:.1f}x the thrust time a 15 W budget allows over that span")
    print(f"  coast {dc[-1] - dc[0]:.1f} d: {len(mins)} periselene passages, periselene "
          f"{rc[mins[0]]:,.0f} -> {rc[mins[-1]]:,.0f} km, min altitude {rc.min() - MOON_RADIUS_KM:,.0f} km, "
          f"max {rc.max():,.0f} km, inside SOI {100 * np.mean(rc < MOON_SOI_KM):.0f} %")
    print(f"  (spiral + Phase 1 xenon {fuel[0] - fuel[spiral_idx[-1] + 1]:.3f} kg)")


def _plot(data: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    days = data[:, 0]
    sat_e = data[:, 1:4]
    sat_m = data[:, 4:7]
    moon_e = sat_e - sat_m
    rm = np.linalg.norm(sat_m, axis=1)
    i_soi, enc = _encounter(rm)
    burning = (np.r_[np.diff(data[:, 7]) < -1e-9, False] if data.shape[1] > 7
               else np.zeros(len(days), bool))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.0, 6.0))
    fig.patch.set_facecolor(C_BG)

    xe, ye = sat_e[:, 0] * 1e-3, sat_e[:, 1] * 1e-3
    mxe, mye = moon_e[:, 0] * 1e-3, moon_e[:, 1] * 1e-3
    _style_axes(axA)
    r_moon = np.hypot(mxe[enc], mye[enc])
    axA.add_patch(plt.Circle((0, 0), r_moon, fill=False, ec=C_FAINT, ls=(0, (6, 6)),
                             lw=1.0, alpha=0.7, zorder=2))
    _glow(axA, xe[:enc + 1], ye[:enc + 1], C_SPIRAL, lw=1.2, zorder=3)
    _disk(axA, 0, 0, EARTH_RADIUS_KM * 1e-3, C_EARTH, glow=C_EARTH_GLOW, zorder=8)
    _disk(axA, mxe[enc], mye[enc], MOON_RADIUS_KM * 1e-3 * 4.0, C_MOON, zorder=7)
    axA.scatter([xe[0]], [ye[0]], s=24, color=C_TEXT, edgecolors="white",
                lw=0.6, zorder=9)
    axA.annotate("start\n(transfer ellipse)", (xe[0], ye[0]),
                 textcoords="offset points", xytext=(14, 12), color=C_TEXT,
                 fontsize=8)
    axA.annotate("Moon\n(encounter)", (mxe[enc], mye[enc]),
                 textcoords="offset points", xytext=(10, 6), color=C_TEXT,
                 fontsize=8)
    axA.set_title("(a)  Earth-centred: 12U CubeSat spirals from its ellipse to "
                  "the Moon", color=C_TEXT, fontsize=10.5, pad=8)
    axA.set_xlabel("x  ($10^3$ km, Earth MJ2000Eq)")
    axA.set_ylabel("y  ($10^3$ km)")
    lim = 1.18 * r_moon
    axA.set_xlim(-lim, lim)
    axA.set_ylim(-lim, lim)

    after = np.arange(len(days)) >= i_soi
    near = after & (rm < 1.15 * MOON_SOI_KM)
    xm, ym = sat_m[:, 0], sat_m[:, 1]
    _style_axes(axB)
    axB.add_patch(plt.Circle((0, 0), MOON_SOI_KM, fill=False, ec=C_FAINT,
                             ls=(0, (5, 5)), lw=1.0, alpha=0.8, zorder=2))
    axB.annotate("SOI", (0, MOON_SOI_KM), textcoords="offset points",
                 xytext=(6, -12), color=C_TEXT, fontsize=8, alpha=0.8)
    axB.plot(np.where(near, xm, np.nan), np.where(near, ym, np.nan), color=C_CAPTURE,
             lw=0.7, alpha=0.85, zorder=3, label="coast")
    arc = near & burning
    axB.plot(np.where(arc, xm, np.nan), np.where(arc, ym, np.nan), color=C_TEXT,
             lw=2.2, solid_capstyle="round", zorder=4, label="braking arcs")
    _disk(axB, 0, 0, MOON_RADIUS_KM, C_MOON, zorder=8)
    axB.scatter([xm[i_soi]], [ym[i_soi]], s=24, color=C_TEXT, edgecolors="white",
                lw=0.6, zorder=9)
    axB.annotate("SOI entry", (xm[i_soi], ym[i_soi]), textcoords="offset points",
                 xytext=(8, 6), color=C_TEXT, fontsize=8)
    axB.legend(loc="lower right", fontsize=8, frameon=False)
    axB.set_title("(b)  Moon-centred: braking-arc capture and 50-day coast",
                  color=C_TEXT, fontsize=10.5, pad=8)
    axB.set_xlabel("x  (km, Moon MJ2000Eq)")
    axB.set_ylabel("y  (km)")
    s = 1.18 * MOON_SOI_KM
    axB.set_xlim(-s, s)
    axB.set_ylim(-s, s)

    fig.tight_layout(pad=1.4)
    out = FIG_DIR / "conae_lunar_trajectory.png"
    fig.savefig(out, dpi=200, facecolor=C_BG, bbox_inches="tight")
    print(f"  figure written to {out}")
    print(f"  spiral+coast to encounter: day {days[enc]:.1f}; "
          f"{int(near.sum())} ephemeris points near the Moon")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replot", action="store_true")
    ap.add_argument("--viz", action="store_true",
                    help="write the GMAT mission script (with OrbitViews) to "
                         "figures/conae_lunar_mission.script for the GMAT GUI; "
                         "no GMAT run")
    args = ap.parse_args()
    if args.viz:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        out = FIG_DIR / "conae_lunar_mission.script"
        out.write_text(build_script("conae_lunar_ephem.txt"), encoding="ascii")
        print(f"  GMAT mission script written to {out}")
        print(f'  open in GMAT:  GMAT.exe --run "{out}"')
        return
    cache = FIG_DIR / "conae_lunar_trajectory.npz"
    if args.replot:
        if not cache.exists():
            raise SystemExit(f"{cache} not found; run once without --replot first.")
        data = np.load(cache)["ephem"]
    else:
        data = _generate()
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, ephem=data)
        print(f"  ephemeris cached to {cache}  ({data.shape[0]} points)")
    _summary(data)
    _plot(data)


if __name__ == "__main__":
    main()
