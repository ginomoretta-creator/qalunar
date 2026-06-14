"""GMAT GUI visualisation of the SMART-1 duty-cycle / Oberth experiment.

Flies three SMART-1 spacecraft side by side from the real GTO, each
spending the same 40 % duty (10 of 24 slots per orbit) but with a
different firing strategy, and shows them in an Earth-inertial OrbitView
so the perigee-thrust advantage is visible as a faster-climbing orbit:

* ``SatQUBO``  (green)  - the QUBO optimum: burns clustered at perigee
* ``SatEven``  (yellow) - evenly-spread burns
* ``SatApo``   (red)    - burns clustered at apogee (worst case)

Each orbit re-anchors at perigee (``Propagate {Earth.Periapsis}``) so the
fixed-duration slot pattern stays phase-locked to perigee as the orbit
grows. The QUBO schedule is read from the deterministic
``smart1_duty_cycle.csv`` produced by ``run_smart1_duty_cycle.py``.

Open in the GMAT GUI (File > Open Script > Run), or:
    GMAT.exe --run scripts\\figures\\smart1_duty_viz.script

Run:  python -m scripts.run_smart1_duty_viz
"""

from __future__ import annotations

import csv
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from qalunar.highfidelity import find_gmat_console


MU_EARTH = 398_600.4418
EARTH_RADIUS_KM = 6_378.137
GTO_PERIGEE_ALT = 654.0
GTO_APOGEE_ALT = 35_885.0
INC_DEG = 7.0
THRUST_MN = 88.0
ISP_S = 1650.0
DRY_MASS_KG = 285.0
XE_LOADED_KG = 82.0
EPOCH = "28 Sep 2003 00:00:00.000"
N_SLOTS = 24
N_BURNS = 10
N_ORBITS = 40

FIG_DIR = Path(__file__).resolve().parent / "figures"


def _gto() -> tuple[float, float, float]:
    rp = EARTH_RADIUS_KM + GTO_PERIGEE_ALT
    ra = EARTH_RADIUS_KM + GTO_APOGEE_ALT
    sma = 0.5 * (rp + ra)
    ecc = (ra - rp) / (ra + rp)
    period = 2.0 * np.pi * np.sqrt(sma ** 3 / MU_EARTH)
    return sma, ecc, period


def _load_schedules() -> dict[str, np.ndarray]:
    csv_path = FIG_DIR / "smart1_duty_cycle.csv"
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found; run run_smart1_duty_cycle first.")
    gains = np.zeros(N_SLOTS)
    qubo = np.zeros(N_SLOTS, dtype=np.int64)
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            s = int(row["slot"])
            gains[s] = float(row["energy_gain_J_per_kg"])
            qubo[s] = int(row["qubo_fires"])
    # evenly-spread and apogee-clustered (worst-K gains) duty schedules
    even = np.zeros(N_SLOTS, dtype=np.int64)
    even[np.clip(np.round(np.linspace(0, N_SLOTS, N_BURNS, endpoint=False)
                          ).astype(int), 0, N_SLOTS - 1)] = 1
    apo = np.zeros(N_SLOTS, dtype=np.int64)
    apo[np.argsort(gains)[:N_BURNS]] = 1
    return {"QUBO": qubo, "Even": even, "Apo": apo}


def _rle(schedule: np.ndarray) -> list[tuple[int, int]]:
    """Run-length encode into (bit, length) segments."""
    segs: list[tuple[int, int]] = []
    for b in schedule.astype(int):
        if segs and segs[-1][0] == b:
            segs[-1] = (b, segs[-1][1] + 1)
        else:
            segs.append((b, 1))
    return segs


def _sat_block(name: str, color: str, sma: float, ecc: float,
               thrust_n: float) -> list[str]:
    return [
        f"Create Spacecraft Sat{name};",
        f"Sat{name}.DateFormat = UTCGregorian;",
        f"Sat{name}.Epoch = '{EPOCH}';",
        f"Sat{name}.CoordinateSystem = EarthMJ2000Eq;",
        f"Sat{name}.DisplayStateType = Keplerian;",
        f"Sat{name}.SMA = {sma:.6f};",
        f"Sat{name}.ECC = {ecc:.8f};",
        f"Sat{name}.INC = {INC_DEG};",
        f"Sat{name}.RAAN = 0;",
        f"Sat{name}.AOP = 0;",
        f"Sat{name}.TA = 0;",
        f"Sat{name}.DryMass = {DRY_MASS_KG};",
        f"Sat{name}.Tanks = {{Tank{name}}};",
        f"Sat{name}.Thrusters = {{Thr{name}}};",
        f"Sat{name}.PowerSystem = Pow{name};",
        f"Sat{name}.OrbitColor = {color};",
        "",
        f"Create ElectricTank Tank{name};",
        f"Tank{name}.AllowNegativeFuelMass = false;",
        f"Tank{name}.FuelMass = {XE_LOADED_KG};",
        "",
        f"Create ElectricThruster Thr{name};",
        f"Thr{name}.CoordinateSystem = Local;",
        f"Thr{name}.Origin = Earth;",
        f"Thr{name}.Axes = VNB;",
        f"Thr{name}.ThrustDirection1 = 1;",
        f"Thr{name}.ThrustDirection2 = 0;",
        f"Thr{name}.ThrustDirection3 = 0;",
        f"Thr{name}.DecrementMass = true;",
        f"Thr{name}.Tank = {{Tank{name}}};",
        f"Thr{name}.ThrustModel = ConstantThrustAndIsp;",
        f"Thr{name}.ConstantThrust = {thrust_n:.6f};",
        f"Thr{name}.Isp = {ISP_S};",
        f"Thr{name}.MaximumUsablePower = 2;",
        f"Thr{name}.MinimumUsablePower = 0.01;",
        "",
        f"Create SolarPowerSystem Pow{name};",
        f"Pow{name}.InitialMaxPower = 1.9;",
        f"Pow{name}.AnnualDecayRate = 0;",
        f"Pow{name}.Margin = 0;",
        f"Pow{name}.ShadowModel = 'None';",
        "",
        f"Create FiniteBurn Burn{name};",
        f"Burn{name}.Thrusters = {{Thr{name}}};",
        "",
    ]


def _orbit_blocks(name: str, schedule: np.ndarray, slot_s: float) -> list[str]:
    """One orbit of the schedule for Sat<name>, then re-anchor at perigee."""
    lines = [f"% --- Sat{name}, one orbit ---"]
    for bit, length in _rle(schedule):
        dur = length * slot_s
        if bit == 1:
            lines += [
                f"BeginFiniteBurn Burn{name}(Sat{name});",
                f"Propagate Prop(Sat{name}) {{Sat{name}.ElapsedSecs = {dur:.6f}}};",
                f"EndFiniteBurn Burn{name}(Sat{name});",
            ]
        else:
            lines.append(
                f"Propagate Prop(Sat{name}) {{Sat{name}.ElapsedSecs = {dur:.6f}}};")
    # re-anchor at the next perigee so the fixed slots stay phase-locked
    lines.append(f"Propagate Prop(Sat{name}) {{Sat{name}.Earth.Periapsis}};")
    return lines


def build_viz_script(report_name: str) -> str:
    sma, ecc, period = _gto()
    slot_s = period / N_SLOTS
    thrust_n = THRUST_MN * 1e-3
    sched = _load_schedules()

    colors = {"QUBO": "Lime", "Even": "Yellow", "Apo": "Red"}
    lines = [
        "% Auto-generated by scripts/run_smart1_duty_viz.py",
        "% SMART-1 duty-cycle strategies side by side from the real GTO.",
        "",
    ]
    for name in ("QUBO", "Even", "Apo"):
        lines += _sat_block(name, colors[name], sma, ecc, thrust_n)

    lines += [
        "Create ForceModel FM;",
        "FM.CentralBody = Earth;",
        "FM.PrimaryBodies = {Earth};",
        "FM.GravityField.Earth.Degree = 2;",
        "FM.GravityField.Earth.Order = 0;",
        "",
        "Create Propagator Prop;",
        "Prop.FM = FM;",
        "Prop.Type = RungeKutta89;",
        "Prop.InitialStepSize = 30;",
        "Prop.Accuracy = 1e-11;",
        "Prop.MinStep = 0;",
        "Prop.MaxStep = 300;",
        "",
        "Create OrbitView View;",
        "View.SolverIterations = Current;",
        "View.UpperLeft = [0.02 0.02];",
        "View.Size = [0.96 0.92];",
        "View.Add = {SatQUBO, SatEven, SatApo, Earth};",
        "View.CoordinateSystem = EarthMJ2000Eq;",
        "View.DrawObject = [true true true true];",
        "View.ViewPointReference = Earth;",
        "View.ViewPointVector = [0 0 180000];",
        "View.ViewDirection = Earth;",
        "View.ViewScaleFactor = 1;",
        "View.ViewUpCoordinateSystem = EarthMJ2000Eq;",
        "View.ViewUpAxis = X;",
        "View.Axes = On;",
        "View.XYPlane = On;",
        "",
        "Create ReportFile Rep;",
        f"Rep.Filename = '{report_name}';",
        "Rep.Precision = 10;",
        "Rep.WriteHeaders = false;",
        "",
        "BeginMissionSequence;",
        "",
    ]

    for orbit in range(N_ORBITS):
        lines.append(f"% ==================== orbit {orbit} ====================")
        for name in ("QUBO", "Even", "Apo"):
            lines += _orbit_blocks(name, sched[name], slot_s)
    lines.append(
        "Report Rep SatQUBO.Earth.RadApo SatEven.Earth.RadApo SatApo.Earth.RadApo;")
    return "\n".join(lines) + "\n"


def main() -> None:
    console = find_gmat_console()
    if not console.exists():
        raise SystemExit(f"GmatConsole not found at {console}")

    report = "smart1_duty_viz_apogees.txt"
    script = build_viz_script(report)
    out = FIG_DIR / "smart1_duty_viz.script"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(script, encoding="ascii")
    print(f"  viz script written to {out}")

    # headless validation
    print("  validating headless ...", flush=True)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".script", delete=False, encoding="ascii"
    ) as fh:
        fh.write(script)
        tmp = Path(fh.name)
    try:
        proc = subprocess.run([str(console), str(tmp)], cwd=str(console.parent),
                              capture_output=True, text=True, timeout=600)
        raw = (proc.stdout or "") + (proc.stderr or "")
        if not re.search(r"Mission run completed", raw):
            raise SystemExit("GMAT failed:\n" + "\n".join(raw.splitlines()[-20:]))
        for d in (console.parent.parent / "output", console.parent / "output",
                  console.parent):
            p = d / report
            if p.exists():
                vals = [float(v) for v in p.read_text().split()]
                p.unlink(missing_ok=True)
                print(f"  after {N_ORBITS} orbits, apogee radius: "
                      f"QUBO {vals[0]:,.0f} km | Even {vals[1]:,.0f} km | "
                      f"Apo {vals[2]:,.0f} km")
                break
    finally:
        tmp.unlink(missing_ok=True)

    print("\n  open in the GMAT GUI to watch the three orbits climb:")
    print(f'    GMAT.exe --run "{out}"')


if __name__ == "__main__":
    main()
