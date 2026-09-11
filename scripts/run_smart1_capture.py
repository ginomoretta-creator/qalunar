"""Stage 2 of the real SMART-1 lunar encounter: binary-QUBO capture.

Takes the lunar encounter found by run_smart1_lunar_encounter (launch
04 Oct 2003, the spiral from the real GTO reaches the lunar SOI near
day 140 at a near-parabolic ~7,000 km perilune, Moon-relative energy
~+0.033 km^2/s^2) and applies the project's binary on/off scheduler to
fire the *just-necessary* anti-tangential burns near perilune that drop
the Moon-relative energy below zero -- i.e. capture into a bound lunar
orbit.

Everything is in real GMAT ephemeris (Earth 4x4 + Luna + Sun). The
braking thruster is the same PPS-1350 Hall but pointed anti-velocity
*relative to the Moon* (Local VNB, Origin = Luna, ThrustDirection1 =
-1). The capture window straddles the SOI passage; it is split into N
slots and the QUBO chooses which to fire:

   minimise  (sum_j q_j dE_j - dE_needed)^2 + alpha * sum_j q_j

where dE_j is the Moon-relative specific-energy change from firing slot
j alone (measured in GMAT), and dE_needed = E0 - E_target drives the
final energy just below zero. The fuel term alpha makes it spend the
*fewest* burns -- the ones near perilune, by the Oberth effect.

Run:  python -m scripts.run_smart1_capture
"""

from __future__ import annotations

import csv
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from qalunar.highfidelity import find_gmat_console
from qalunar.highfidelity.gmat_oracle import epoch_plus_seconds


EARTH_RADIUS_KM = 6_378.137
GTO_PERIGEE_ALT, GTO_APOGEE_ALT, INC_DEG = 654.0, 35_885.0, 7.0
THRUST_MN, ISP_S, DRY_MASS_KG, XE_LOADED_KG = 88.0, 1650.0, 285.0, 82.0
ENCOUNTER_EPOCH = "04 Oct 2003 00:00:00.000"
GM_LUNA = 4_902.8005821478
MOON_RADIUS_KM = 1_737.4
MOON_SOI_KM = 66_100.0

CAPTURE_WINDOW_DAYS = 3.0
N_SLOTS = 16
E_TARGET = -0.080           # km^2/s^2 (the cached smart1_capture.csv was solved with this)
# Tiebreaker only: must be << the per-slot energy benefit ~2|dE||d| ~ 2e-4,
# or the fuel term swamps the capture objective and the QUBO refuses to brake.
FUEL_WEIGHT = 1e-5
FIG_DIR = Path(__file__).resolve().parent / "figures"


def _gto():
    rp, ra = EARTH_RADIUS_KM + GTO_PERIGEE_ALT, EARTH_RADIUS_KM + GTO_APOGEE_ALT
    return 0.5 * (rp + ra), (ra - rp) / (ra + rp)


def _run(script: str, report: str, console: Path, timeout: float = 600) -> np.ndarray:
    with tempfile.NamedTemporaryFile("w", suffix=".script", delete=False,
                                     encoding="ascii") as fh:
        fh.write(script)
        path = Path(fh.name)
    log = path.with_suffix(".log")     # private log per run (parallel-safe)
    try:
        proc = subprocess.run([str(console), "-l", str(log), "-r", str(path)],
                              cwd=str(console.parent),
                              capture_output=True, text=True, timeout=timeout)
        raw = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0 or not re.search(r"Mission run completed", raw):
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
        log.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Stage A: spiral from the GTO to the SOI entry, hand off the Cartesian state
# ---------------------------------------------------------------------------

def _spiral_to_soi(console: Path) -> tuple[np.ndarray, float, float]:
    sma, ecc = _gto()
    script = f"""% SMART-1 spiral to lunar SOI, hand off state.
Create Spacecraft Sat;
Sat.DateFormat = UTCGregorian;
Sat.Epoch = '{ENCOUNTER_EPOCH}';
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
Rep.Filename = 'smart1_cap_handoff.txt';
Rep.Precision = 14;
Rep.WriteHeaders = false;

Create Variable goflag niter;

BeginMissionSequence;
BeginFiniteBurn Spiral(Sat);
goflag = 1;
niter = 0;
While goflag > 0.5
   Propagate Prop(Sat) {{Sat.ElapsedSecs = 43200}};
   niter = niter + 1;
   goflag = 0;
   If Sat.Luna.RMAG > {MOON_SOI_KM:.1f}
      goflag = 1;
   EndIf
   If niter > 320
      goflag = 0;
   EndIf
EndWhile
EndFiniteBurn Spiral(Sat);
Report Rep Sat.ElapsedSecs Sat.EarthMJ2000Eq.X Sat.EarthMJ2000Eq.Y Sat.EarthMJ2000Eq.Z Sat.EarthMJ2000Eq.VX Sat.EarthMJ2000Eq.VY Sat.EarthMJ2000Eq.VZ Sat.XeTank.FuelMass Sat.Luna.RMAG Sat.Luna.Energy;
"""
    row = _run(script, "smart1_cap_handoff.txt", console)[-1]
    elapsed_s = float(row[0])
    state = row[1:7].astype(float)
    fuel = float(row[7])
    print(f"  SOI entry at +{elapsed_s/86400:.2f} d, Moon dist {row[8]:,.0f} km, "
          f"E_moon {row[9]:+.4f}, fuel {fuel:.1f} kg")
    return state, elapsed_s, fuel


# ---------------------------------------------------------------------------
# Stage B: capture-window flight under an anti-tangential binary schedule
# ---------------------------------------------------------------------------

def _capture_flight_script(state: np.ndarray, epoch: str, fuel: float,
                           schedule: np.ndarray, report: str) -> str:
    dt = CAPTURE_WINDOW_DAYS * 86_400.0 / schedule.size
    # RLE the schedule into burn/coast segments
    segs, i = [], 0
    while i < schedule.size:
        b = int(schedule[i]); j = i
        while j < schedule.size and int(schedule[j]) == b:
            j += 1
        segs.append((b, j - i)); i = j

    seq = []
    for b, length in segs:
        dur = length * dt
        if b == 1:
            seq += [
                "BeginFiniteBurn Brake(Sat);",
                f"Propagate Prop(Sat) {{Sat.ElapsedSecs = {dur:.4f}}};",
                "EndFiniteBurn Brake(Sat);",
            ]
        else:
            seq.append(f"Propagate Prop(Sat) {{Sat.ElapsedSecs = {dur:.4f}}};")
    seq_str = "\n".join(seq)

    return f"""% SMART-1 lunar capture window, anti-tangential (Moon-relative) braking.
Create Spacecraft Sat;
Sat.DateFormat = UTCGregorian;
Sat.Epoch = '{epoch}';
Sat.CoordinateSystem = EarthMJ2000Eq;
Sat.DisplayStateType = Cartesian;
Sat.X = {state[0]:.10f};
Sat.Y = {state[1]:.10f};
Sat.Z = {state[2]:.10f};
Sat.VX = {state[3]:.12f};
Sat.VY = {state[4]:.12f};
Sat.VZ = {state[5]:.12f};
Sat.DryMass = {DRY_MASS_KG};
Sat.Tanks = {{XeTank}};
Sat.Thrusters = {{Brk}};
Sat.PowerSystem = SolarP;

Create ElectricTank XeTank;
XeTank.AllowNegativeFuelMass = false;
XeTank.FuelMass = {fuel:.4f};

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
Brk.ConstantThrust = {THRUST_MN * 1e-3:.6f};
Brk.Isp = {ISP_S};
Brk.MaximumUsablePower = 2;
Brk.MinimumUsablePower = 0.01;

Create SolarPowerSystem SolarP;
SolarP.InitialMaxPower = 1.9;
SolarP.AnnualDecayRate = 0;
SolarP.Margin = 0;
SolarP.ShadowModel = 'None';

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
Prop.InitialStepSize = 30;
Prop.Accuracy = 1e-10;
Prop.MinStep = 0;
Prop.MaxStep = 600;

Create ReportFile Rep;
Rep.Filename = '{report}';
Rep.Precision = 12;
Rep.WriteHeaders = false;

BeginMissionSequence;
{seq_str}
Report Rep Sat.Luna.Energy Sat.Luna.RMAG Sat.XeTank.FuelMass;
"""


def _fly_capture(state, epoch, fuel, schedule, console, tag) -> tuple[float, float, float]:
    report = f"smart1_cap_{tag}.txt"
    row = _run(_capture_flight_script(state, epoch, fuel, schedule, report),
               report, console)[-1]
    return float(row[0]), float(row[1]), float(row[2])   # E_moon, RMAG, fuel


def main() -> None:
    global CAPTURE_WINDOW_DAYS, E_TARGET
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=CAPTURE_WINDOW_DAYS,
                    help="capture window length in days")
    ap.add_argument("--e-target", type=float, default=E_TARGET,
                    help="target Moon-relative energy (km^2/s^2, negative)")
    args = ap.parse_args()
    CAPTURE_WINDOW_DAYS = args.window
    E_TARGET = args.e_target

    console = find_gmat_console()
    if not console.exists():
        raise SystemExit(f"GmatConsole not found at {console}")

    print("=" * 90)
    print("  SMART-1 lunar capture from the real GTO encounter "
          "(binary-QUBO anti-tangential braking)")
    print("=" * 90)

    print("\n[A] spiralling from the 04 Oct 2003 GTO to the lunar SOI ...", flush=True)
    state, elapsed_s, fuel = _spiral_to_soi(console)
    cap_epoch = epoch_plus_seconds(ENCOUNTER_EPOCH, elapsed_s)

    coast = np.zeros(N_SLOTS, dtype=np.int64)
    e0, rmag0, _ = _fly_capture(state, cap_epoch, fuel, coast, console, "coast")
    print(f"\n[B] capture window {CAPTURE_WINDOW_DAYS} d, {N_SLOTS} slots, "
          f"anti-tangential (Moon VNB -V)")
    print(f"    coast (no braking): final E_moon {e0:+.4f} km^2/s^2 "
          f"({'bound' if e0 < 0 else 'UNBOUND'})")

    # ---- per-slot energy change from a single braking slot (parallel) ----
    print(f"    measuring per-slot dE in GMAT ({N_SLOTS} runs) ...", flush=True)

    def _slot_dE(j: int) -> float:
        q = np.zeros(N_SLOTS, dtype=np.int64); q[j] = 1
        ej, _, _ = _fly_capture(state, cap_epoch, fuel, q, console, f"s{j}")
        return ej - e0

    with ThreadPoolExecutor(max_workers=6) as pool:
        dE = np.array(list(pool.map(_slot_dE, range(N_SLOTS))))

    # ---- capture QUBO: hit E_target with fewest burns ----
    # minimise (sum q_j dE_j - d)^2 + alpha sum q_j,  d = E_target - e0
    d = E_TARGET - e0
    Q = np.outer(dE, dE)
    lin = FUEL_WEIGHT * np.ones(N_SLOTS) - 2.0 * dE * d
    best_q, best_E = None, np.inf
    for m in range(1 << N_SLOTS):
        q = np.frombuffer(np.array([m], dtype=">i8").tobytes(), dtype=np.uint8)
        q = np.unpackbits(q)[-N_SLOTS:].astype(np.float64)
        val = q @ Q @ q + lin @ q
        if val < best_E:
            best_E, best_q = val, q.copy()
    schedule = best_q.astype(np.int64)

    e_cap, rmag_cap, fuel_cap = _fly_capture(state, cap_epoch, fuel, schedule,
                                             console, "qubo")
    # dv in m/s: (thrust[N]/mass[kg]) * burn_seconds; thrust/mass is m/s^2
    slot_s = CAPTURE_WINDOW_DAYS * 86400.0 / N_SLOTS
    dv = int(schedule.sum()) * (THRUST_MN * 1e-3) / (DRY_MASS_KG + fuel) * slot_s
    on = np.flatnonzero(schedule)
    print(f"    QUBO fires {int(schedule.sum())}/{N_SLOTS} slots "
          f"(indices {list(on)})")
    print(f"    predicted dE sum {schedule @ dE:+.4f}, needed {d:+.4f}")

    print("\n" + "-" * 90)
    print(f"  RESULT: final E_moon {e_cap:+.5f} km^2/s^2  "
          f"-> {'CAPTURED (bound)' if e_cap < 0 else 'still unbound'}")
    if e_cap < 0:
        sma = -GM_LUNA / (2.0 * e_cap)
        print(f"  bound lunar orbit: a = {sma:,.0f} km, "
              f"period {2*np.pi*np.sqrt(sma**3/GM_LUNA)/3600:,.1f} h, "
              f"Moon dist now {rmag_cap:,.0f} km")
    print(f"  braking burns {int(schedule.sum())}/{N_SLOTS}  ~dV {dv:,.0f} m/s  "
          f"xenon left {fuel_cap:.1f} kg")
    print("-" * 90)

    out = FIG_DIR / "smart1_capture.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["slot", "dE_km2_s2", "qubo_fires"])
        for j in range(N_SLOTS):
            w.writerow([j, f"{dE[j]:.6e}", int(schedule[j])])
        w.writerow([])
        w.writerow(["e0", f"{e0:.6f}"])
        w.writerow(["e_capture", f"{e_cap:.6f}"])
        w.writerow(["captured", e_cap < 0])
    print(f"  table written to {out}")


if __name__ == "__main__":
    main()
