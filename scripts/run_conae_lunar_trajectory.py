"""Full CONAE 12U CubeSat cislunar mission as a single continuous GMAT trace,
rendered as a publication trajectory figure (light, OrbitView-like).

  Phase 1  raise the perigee out of the disposal ellipse with apogee burns
           (collision avoidance);
  Phase 2  raise the apogee to lunar distance with continuous tangential
           thrust (the spiral);
  Phase 3  at the lunar encounter, anti-tangential braking (Moon VNB -V)
           to bind the orbit.

Flown headless in GMAT (DE405, Earth 4x4 + Sun); the full position ephemeris is
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

# Same reference ellipse as run_conae_duty_cycle: CONAE apogee radius, 250 km
# injection perigee (the earlier a/e pair put the perigee below the surface).
PERIGEE_ALT_KM, APOGEE_RADIUS_KM = 250.0, 76_922.0
_RP = EARTH_RADIUS_KM + PERIGEE_ALT_KM
SMA_KM = 0.5 * (_RP + APOGEE_RADIUS_KM)
ECC = (APOGEE_RADIUS_KM - _RP) / (APOGEE_RADIUS_KM + _RP)
INC_DEG = 39.0
TA0_DEG = 150.0                                 # injection true anomaly (paper Table 3)
THRUST_N, ISP_S, DRY_MASS_KG, XE_KG = 0.040, 1200.0, 13.0, 3.5
APOGEE_BURN_S = 2626.0
N_PHASE1 = 5                                    # five apogee passes, as in the text
APO_TARGET_KM = 370_000.0
ENCOUNTER_EPOCH = "24 Jan 2026 12:00:00.000"   # +23.5 d -> 13,051 km perilune (XE 3.5)
BRAKE_S = 90_000.0                              # anti-tangential capture window
POST_COAST_S = 50.0 * 86400.0                   # 50 d post-capture coast (the stability claim)


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
Eph.Add = {{Sat.ElapsedDays, Sat.EarthMJ2000Eq.X, Sat.EarthMJ2000Eq.Y, Sat.EarthMJ2000Eq.Z, Sat.MoonView.X, Sat.MoonView.Y, Sat.MoonView.Z}};

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

Create Variable goflag niter;

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

% --- coast to the encounter periselene (brake AT periapsis, not on infall) ---
Propagate Prop(Sat) {{Sat.Luna.Periapsis, Sat.ElapsedSecs = 400000}};

% --- Phase 3: anti-tangential braking (Moon -V) capture at periselene.
% Braking at periselene is Oberth-efficient AND preserves/raises the perilune,
% giving a bound orbit whose perilune stays well above the surface (a brake
% begun at SOI entry instead drags the perilune sub-surface -> impact). ---
BeginFiniteBurn Brake(Sat);
Propagate Prop(Sat) {{Sat.ElapsedSecs = {BRAKE_S:.1f}}};
EndFiniteBurn Brake(Sat);
Report Mile Sat.ElapsedDays Sat.Earth.RadPer Sat.Earth.RadApo Sat.Luna.RMAG Sat.Luna.Energy Sat.XeTank.FuelMass;

% --- short post-capture coast ---
Propagate Prop(Sat) {{Sat.ElapsedSecs = {POST_COAST_S:.1f}}};
Report Mile Sat.ElapsedDays Sat.Earth.RadPer Sat.Earth.RadApo Sat.Luna.RMAG Sat.Luna.Energy Sat.XeTank.FuelMass;
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


def _plot(data: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    days = data[:, 0]
    sat_e = data[:, 1:4]
    sat_m = data[:, 4:7]
    moon_e = sat_e - sat_m

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.0, 6.0))
    fig.patch.set_facecolor(C_BG)

    xe, ye = sat_e[:, 0] * 1e-3, sat_e[:, 1] * 1e-3
    mxe, mye = moon_e[:, 0] * 1e-3, moon_e[:, 1] * 1e-3
    _style_axes(axA)
    enc = int(np.argmin(np.linalg.norm(sat_m, axis=1)))
    r_moon = np.hypot(mxe[enc], mye[enc])
    axA.add_patch(plt.Circle((0, 0), r_moon, fill=False, ec=C_FAINT, ls=(0, (6, 6)),
                             lw=1.0, alpha=0.7, zorder=2))
    _glow(axA, xe[:enc + 1], ye[:enc + 1], C_SPIRAL, lw=1.2, zorder=3)
    _disk(axA, 0, 0, EARTH_RADIUS_KM * 1e-3, C_EARTH, glow=C_EARTH_GLOW, zorder=8)
    _disk(axA, mxe[enc], mye[enc], MOON_RADIUS_KM * 1e-3 * 4.0, C_MOON, zorder=7)
    axA.scatter([xe[0]], [ye[0]], s=24, color=C_TEXT, edgecolors="white",
                lw=0.6, zorder=9)
    axA.annotate("start\n(disposal ellipse)", (xe[0], ye[0]),
                 textcoords="offset points", xytext=(8, -16), color=C_TEXT,
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

    rm = np.linalg.norm(sat_m, axis=1)
    near = rm < 1.15 * MOON_SOI_KM
    xm, ym = sat_m[near, 0], sat_m[near, 1]
    _style_axes(axB)
    axB.add_patch(plt.Circle((0, 0), MOON_SOI_KM, fill=False, ec=C_FAINT,
                             ls=(0, (5, 5)), lw=1.0, alpha=0.8, zorder=2))
    axB.annotate("SOI", (0, MOON_SOI_KM), textcoords="offset points",
                 xytext=(6, -12), color=C_TEXT, fontsize=8, alpha=0.8)
    _glow(axB, xm, ym, C_CAPTURE, lw=1.5, zorder=3)
    _disk(axB, 0, 0, MOON_RADIUS_KM, C_MOON, zorder=8)
    if near.any():
        axB.scatter([xm[0]], [ym[0]], s=24, color=C_TEXT, edgecolors="white",
                    lw=0.6, zorder=9)
        axB.annotate("SOI entry", (xm[0], ym[0]), textcoords="offset points",
                     xytext=(8, 6), color=C_TEXT, fontsize=8)
    axB.set_title("(b)  Moon-centred: anti-tangential braking capture",
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
    _plot(data)


if __name__ == "__main__":
    main()
