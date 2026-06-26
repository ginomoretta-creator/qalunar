"""Exploratory: can the CONAE 12U CubeSat spiral from its eccentric ellipse to
the Moon? Phase 1 raises the perigee (apogee burns); Phase 2 raises the apogee
with continuous tangential thrust until the lunar SOI. Reports the apogee
growth and closest lunar approach over a coarse launch-epoch scan, to find a
real encounter and the timescale.

Run:  python -m scripts.run_conae_lunar_explore
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from qalunar.highfidelity import find_gmat_console
from qalunar.highfidelity.gmat_oracle import epoch_plus_seconds

MU_EARTH = 398_600.4418
EARTH_RADIUS_KM = 6_378.137
SMA_KM = 41_646.0
ECC = 0.847020121
INC_DEG = 39.0
THRUST_N = 0.040
ISP_S = 1200.0
DRY_MASS_KG = 13.0
XE_KG = 3.5
APOGEE_BURN_S = 2626.0
N_PHASE1 = 6                  # apogee burns to raise perigee out of reentry
MOON_SOI_KM = 66_100.0
APO_TARGET_KM = 370_000.0    # stop Phase-2 thrust when apogee reaches lunar dist
SPIRAL_STEP_S = 14_400.0     # 4 h thrust steps
SPIRAL_MAX_STEPS = 90        # cap Phase-2 spiral
COAST_STEP_S = 21_600.0      # 6 h coast steps
COAST_STEPS = 80             # ~20 d coast for the lunar encounter


def build_script(epoch: str, report: str) -> str:
    sma, ecc = SMA_KM, ECC
    return f"""% CONAE CubeSat -> Moon exploration (Phase 1 perigee raise + Phase 2 spiral)
Create Spacecraft Sat;
Sat.DateFormat = UTCGregorian;
Sat.Epoch = '{epoch}';
Sat.CoordinateSystem = EarthMJ2000Eq;
Sat.DisplayStateType = Keplerian;
Sat.SMA = {sma:.6f};
Sat.ECC = {ecc:.9f};
Sat.INC = {INC_DEG};
Sat.RAAN = 0;
Sat.AOP = 0;
Sat.TA = 0;
Sat.DryMass = {DRY_MASS_KG};
Sat.Tanks = {{XeTank}};
Sat.Thrusters = {{Hall}};
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

Create SolarPowerSystem SolarP;
SolarP.InitialMaxPower = 1.9;
SolarP.AnnualDecayRate = 0;
SolarP.Margin = 0;
SolarP.ShadowModel = 'None';

Create FiniteBurn Spiral;
Spiral.Thrusters = {{Hall}};

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

Create ReportFile Rep;
Rep.Filename = '{report}';
Rep.Precision = 10;
Rep.WriteHeaders = false;

Create Variable goflag niter;

BeginMissionSequence;

% --- Phase 1: raise perigee with apogee burns ---
For niter = 1:{N_PHASE1};
   Propagate Prop(Sat) {{Sat.Earth.Apoapsis}};
   BeginFiniteBurn Spiral(Sat);
   Propagate Prop(Sat) {{Sat.ElapsedSecs = {APOGEE_BURN_S:.1f}}};
   EndFiniteBurn Spiral(Sat);
EndFor;
Report Rep Sat.ElapsedDays Sat.Earth.RadPer Sat.Earth.RadApo Sat.Luna.RMAG;

% --- Phase 2: continuous tangential spiral until apogee reaches lunar dist ---
BeginFiniteBurn Spiral(Sat);
goflag = 1;
niter = 0;
While goflag > 0.5
   Propagate Prop(Sat) {{Sat.ElapsedSecs = {SPIRAL_STEP_S:.1f}}};
   niter = niter + 1;
   Report Rep Sat.ElapsedDays Sat.Earth.RadPer Sat.Earth.RadApo Sat.Luna.RMAG;
   goflag = 0;
   If Sat.Earth.RadApo < {APO_TARGET_KM:.1f}
      goflag = 1;
   EndIf
   If niter > {SPIRAL_MAX_STEPS}
      goflag = 0;
   EndIf
EndWhile
EndFiniteBurn Spiral(Sat);

% --- Phase 3: coast; let the Moon come to the apogee (encounter phasing) ---
For niter = 1:{COAST_STEPS};
   Propagate Prop(Sat) {{Sat.ElapsedSecs = {COAST_STEP_S:.1f}}};
   Report Rep Sat.ElapsedDays Sat.Earth.RadPer Sat.Earth.RadApo Sat.Luna.RMAG;
EndFor;
"""


def run_epoch(epoch: str, console: Path) -> tuple[float, float, float]:
    report = "conae_lunar_explore.txt"
    with tempfile.NamedTemporaryFile("w", suffix=".script", delete=False,
                                     encoding="ascii") as fh:
        fh.write(build_script(epoch, report))
        tmp = Path(fh.name)
    try:
        proc = subprocess.run([str(console), str(tmp)], cwd=str(console.parent),
                              capture_output=True, text=True, timeout=900)
        raw = (proc.stdout or "") + (proc.stderr or "")
        if not re.search(r"Mission run completed", raw):
            return (float("nan"),) * 3
        for d in (console.parent.parent / "output", console.parent / "output",
                  console.parent):
            p = d / report
            if p.exists():
                rows = np.asarray([[float(v) for v in ln.split()]
                                   for ln in p.read_text().strip().splitlines()
                                   if ln.strip()])
                p.unlink(missing_ok=True)
                days = rows[:, 0]
                rapo = rows[:, 2]
                moon = rows[:, 3]
                return float(moon.min()), float(rapo.max()), float(days[-1])
        return (float("nan"),) * 3
    finally:
        tmp.unlink(missing_ok=True)


def main() -> None:
    console = find_gmat_console()
    if not console.exists():
        raise SystemExit(f"GmatConsole not found at {console}")
    # Coarse synodic-month scan; refine around the dip in 0.25 d steps. With
    # XE_KG = 3.5 the encounter falls at +23.5 d (24 Jan 2026 12:00), perilune
    # ~13,000 km -- the epoch hardcoded in run_conae_lunar_trajectory. NOTE: the
    # phasing is mass-sensitive, so re-scan if the propulsion config changes.
    base = "01 Jan 2026 00:00:00.000"
    print("scanning launch epochs (Phase 1 + Phase 2 spiral) ...")
    print(f"{'epoch offset [d]':>16} {'min Moon [km]':>14} {'max apo [km]':>14} "
          f"{'t_end [d]':>10}")
    best = None
    offsets = [round(x, 1) for x in np.arange(0.0, 30.0, 2.0)]
    for off_d in offsets:
        epoch = epoch_plus_seconds(base, off_d * 86400.0)
        moon_min, apo_max, t_end = run_epoch(epoch, console)
        flag = "  <-- SOI!" if moon_min < MOON_SOI_KM else ""
        print(f"{off_d:>16} {moon_min:>14,.0f} {apo_max:>14,.0f} {t_end:>10.1f}{flag}")
        if best is None or moon_min < best[1]:
            best = (off_d, moon_min, apo_max, t_end)
    print(f"\nbest: epoch +{best[0]} d -> closest Moon {best[1]:,.0f} km "
          f"(apogee reached {best[2]:,.0f} km, spiral {best[3]:.1f} d)")


if __name__ == "__main__":
    main()
