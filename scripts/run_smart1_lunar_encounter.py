"""Stage 1 of the real SMART-1 lunar encounter: phase search over launch epoch.

The Earth-escape spiral from SMART-1's real GTO is deterministic given the
thruster and the start epoch; whether it actually meets the Moon depends on
the *phase* -- where the Moon is when the apogee finally reaches lunar
distance (~day 130-150). SMART-1 solved this with months of resonant-encounter
targeting. Here we exploit the one free knob we control, the launch epoch:
scanning it across a synodic month sweeps the lunar phase at encounter, so
some epoch yields a close lunar pass.

For each trial epoch this flies the continuous-thrust spiral (real ephemeris:
Earth 4x4 + Luna + Sun) in 1-day chunks, recording the Moon-relative distance,
and reports the closest lunar approach reached before fuel-out / the time cap.
The best epoch is the seed for Stage 2 (refine the encounter + run the
binary-QUBO capture).

Run:  python -m scripts.run_smart1_lunar_encounter
"""

from __future__ import annotations

import csv
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from qalunar.highfidelity import find_gmat_console
from qalunar.highfidelity.gmat_oracle import epoch_plus_seconds


EARTH_RADIUS_KM = 6_378.137
GTO_PERIGEE_ALT = 654.0
GTO_APOGEE_ALT = 35_885.0
INC_DEG = 7.0
THRUST_MN = 88.0
ISP_S = 1650.0
DRY_MASS_KG = 285.0
XE_LOADED_KG = 82.0
BASE_EPOCH = "28 Sep 2003 00:00:00.000"

MOON_SOI_KM = 66_100.0
CHUNK_DAYS = 1.0
MAX_DAYS = 200
FIG_DIR = Path(__file__).resolve().parent / "figures"

# Phase scan: one synodic month of start epochs.
N_EPOCHS = 14
EPOCH_STEP_DAYS = 2.0


def _gto() -> tuple[float, float]:
    rp = EARTH_RADIUS_KM + GTO_PERIGEE_ALT
    ra = EARTH_RADIUS_KM + GTO_APOGEE_ALT
    return 0.5 * (rp + ra), (ra - rp) / (ra + rp)


def _scan_script(epoch: str, report: str) -> str:
    sma, ecc = _gto()
    chunk_s = CHUNK_DAYS * 86_400.0
    return f"""% SMART-1 lunar-encounter phase probe, epoch {epoch}
Create Spacecraft Sat;
Sat.DateFormat = UTCGregorian;
Sat.Epoch = '{epoch}';
Sat.CoordinateSystem = EarthMJ2000Eq;
Sat.DisplayStateType = Keplerian;
Sat.SMA = {sma:.6f};
Sat.ECC = {ecc:.8f};
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
XeTank.FuelMass = {XE_LOADED_KG};

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
Hall.ConstantThrust = {THRUST_MN * 1e-3:.6f};
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

BeginFiniteBurn Spiral(Sat);
goflag = 1;
niter = 0;
While goflag > 0.5
   Propagate Prop(Sat) {{Sat.ElapsedSecs = {chunk_s:.1f}}};
   niter = niter + 1;
   Report Rep Sat.ElapsedDays Sat.Earth.RadApo Sat.Luna.RMAG Sat.XeTank.FuelMass;
   goflag = 0;
   If Sat.Luna.RMAG > {MOON_SOI_KM:.1f}
      goflag = 1;
   EndIf
   If Sat.XeTank.FuelMass < 3
      goflag = 0;
   EndIf
   If niter > {MAX_DAYS}
      goflag = 0;
   EndIf
EndWhile
EndFiniteBurn Spiral(Sat);
"""


def _run(script_text: str, report: str, console: Path) -> np.ndarray:
    with tempfile.NamedTemporaryFile(
        "w", suffix=".script", delete=False, encoding="ascii"
    ) as fh:
        fh.write(script_text)
        path = Path(fh.name)
    try:
        proc = subprocess.run(
            [str(console), str(path)], cwd=str(console.parent),
            capture_output=True, text=True, timeout=900,
        )
        raw = (proc.stdout or "") + (proc.stderr or "")
        if not re.search(r"Mission run completed", raw):
            raise RuntimeError("GMAT failed:\n" + "\n".join(raw.splitlines()[-15:]))
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


def main() -> None:
    console = find_gmat_console()
    if not console.exists():
        raise SystemExit(f"GmatConsole not found at {console}")

    print("=" * 88)
    print("  SMART-1 lunar-encounter phase search (launch-epoch scan)")
    print(f"  GTO {GTO_PERIGEE_ALT:.0f} x {GTO_APOGEE_ALT:.0f} km, 88 mN Hall, "
          f"{N_EPOCHS} epochs x {EPOCH_STEP_DAYS:.0f} d from {BASE_EPOCH}")
    print("=" * 88)
    print(f"  {'epoch':<26}{'days@close':>12}{'min lunar dist [km]':>22}"
          f"{'SOI?':>6}{'apo@end [km]':>16}")
    print("  " + "-" * 80)

    rows = []
    for i in range(N_EPOCHS):
        epoch = epoch_plus_seconds(BASE_EPOCH, i * EPOCH_STEP_DAYS * 86_400.0)
        log = _run(_scan_script(epoch, f"smart1_enc_{i}.txt"),
                   f"smart1_enc_{i}.txt", console)
        days, apo, rmag, fuel = log.T
        kmin = int(np.argmin(rmag))
        soi = bool(rmag[kmin] <= MOON_SOI_KM)
        rows.append(dict(epoch=epoch, day_close=round(float(days[kmin]), 1),
                         min_lunar_km=round(float(rmag[kmin]), 1), soi=soi,
                         apo_end_km=round(float(apo[-1]), 1)))
        print(f"  {epoch:<26}{days[kmin]:>12.1f}{rmag[kmin]:>22,.0f}"
              f"{('YES' if soi else 'no'):>6}{apo[-1]:>16,.0f}", flush=True)

    rows.sort(key=lambda r: r["min_lunar_km"])
    best = rows[0]
    print("  " + "-" * 80)
    print(f"\n  closest encounter: epoch {best['epoch']} -> "
          f"{best['min_lunar_km']:,.0f} km at day {best['day_close']} "
          f"({'enters SOI' if best['soi'] else 'no SOI entry'})")

    out = FIG_DIR / "smart1_lunar_encounter_scan.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["epoch"]))
    print(f"  scan table written to {out}")
    if best["soi"]:
        print("\n  -> Stage 2: refine this epoch and run the binary-QUBO capture.")
    else:
        print("\n  -> no epoch entered the SOI; will refine around the closest "
              "or extend thrust strategy in Stage 2.")


if __name__ == "__main__":
    main()
