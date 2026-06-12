"""Lunar-capture arrival metrics of the Phase-2 schedules, measured in GMAT.

The mission objective of Phase 2 is not to hit a reference state per se
but to arrive at the lunar SOI with Moon-relative energy and speed that
the Phase-3 anti-tangential sliding-window QUBO can stabilise. This
script measures exactly that, under real DE-series ephemerides, for the
three schedules of the GMAT-in-the-loop experiment:

* coast          - uncorrected injection (no burns)
* CR3BP design   - the paper's 3-burn schedule (designed in the CR3BP)
* GMAT QUBO      - the 2-burn schedule found with GMAT in the loop

Each schedule is flown in GMAT with Moon-relative RMAG / VMAG / energy
auto-logged at every integration step; Python then extracts

* the SOI crossing (first descent below 66,100 km): speed and energy,
* the perilune (minimum Moon distance): altitude, speed, two-body
  energy, v_inf, local circular/escape speeds, and the capture
  Delta-v deficit (v_peri - v_circ) that Phase 3 must dissipate.

The same metrics are computed along the in-house CR3BP propagation for
contrast, so each schedule gets a CR3BP-vs-GMAT row pair.

Run:  python -m scripts.run_gmat_capture_metrics
"""

from __future__ import annotations

import csv
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from qalunar.dynamics import PlanarCR3BP
from qalunar.dynamics.cr3bp import EARTH_MOON_MU
from qalunar.highfidelity import GmatOracleConfig, find_gmat_console
from qalunar.highfidelity.gmat_oracle import synodic_to_rotating_km
from qalunar.qubo.thrust_scheduling import ThrustSchedulingConfig, propagate_schedule
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

V_RATIO = 1.187
THRUST_MAG = 0.02
T_SPAN = (0.0, 4.0)
N_STEPS = 15
EPOCH = "08 Jan 2026 00:00:00.000"

MOON_RADIUS_KM = 1_737.4
MOON_SOI_KM = 66_100.0
GM_LUNA_KM3_S2 = 4_902.8005821478  # DE405 value used by GMAT

# Deterministic outputs of run_gmat_oracle_viz.py (brute-force sampler,
# fixed epoch); regenerate with that script if the scenario changes.
SCHEDULES = {
    "coast": np.zeros(N_STEPS, dtype=np.int64),
    "CR3BP design": np.array([int(b) for b in "001001100000000"], dtype=np.int64),
    "GMAT QUBO": np.array([int(b) for b in "010000001000000"], dtype=np.int64),
}


# ---------------------------------------------------------------------------
# GMAT script: fly all three schedules, auto-log Moon-relative quantities
# ---------------------------------------------------------------------------


def _sat_blocks(name: str, state_rot: np.ndarray, has_burns: bool,
                thrust_n: float, epoch: str = EPOCH) -> list[str]:
    lines = [
        f"Create Spacecraft {name};",
        f"{name}.DateFormat = UTCGregorian;",
        f"{name}.Epoch = '{epoch}';",
        f"{name}.CoordinateSystem = EarthMoonRot;",
        f"{name}.DisplayStateType = Cartesian;",
        f"{name}.X = {state_rot[0]:.12f};",
        f"{name}.Y = {state_rot[1]:.12f};",
        f"{name}.Z = 0.0;",
        f"{name}.VX = {state_rot[2]:.15f};",
        f"{name}.VY = {state_rot[3]:.15f};",
        f"{name}.VZ = 0.0;",
        f"{name}.DryMass = 95;",
    ]
    if has_burns:
        lines += [
            f"{name}.Tanks = {{Tank{name}}};",
            f"{name}.Thrusters = {{Thr{name}}};",
            f"{name}.PowerSystem = Pow{name};",
            "",
            f"Create ElectricTank Tank{name};",
            f"Tank{name}.AllowNegativeFuelMass = false;",
            f"Tank{name}.FuelMass = 5;",
            "",
            f"Create ElectricThruster Thr{name};",
            f"Thr{name}.CoordinateSystem = EarthMoonRot;",
            f"Thr{name}.ThrustDirection1 = 1;",
            f"Thr{name}.ThrustDirection2 = 0;",
            f"Thr{name}.ThrustDirection3 = 0;",
            f"Thr{name}.DecrementMass = false;",
            f"Thr{name}.Tank = {{Tank{name}}};",
            f"Thr{name}.ThrustModel = ConstantThrustAndIsp;",
            f"Thr{name}.ConstantThrust = {thrust_n:.12e};",
            f"Thr{name}.Isp = 1640;",
            f"Thr{name}.MaximumUsablePower = 10;",
            f"Thr{name}.MinimumUsablePower = 0.001;",
            "",
            f"Create SolarPowerSystem Pow{name};",
            f"Pow{name}.InitialMaxPower = 5;",
            f"Pow{name}.AnnualDecayRate = 0;",
            f"Pow{name}.Margin = 0;",
            f"Pow{name}.ShadowModel = 'None';",
            "",
            f"Create FiniteBurn Burn{name};",
            f"Burn{name}.Thrusters = {{Thr{name}}};",
        ]
    lines += [
        "",
        f"Create ReportFile Rep{name};",
        f"Rep{name}.Filename = 'qalunar_capture_{name}.txt';",
        f"Rep{name}.Precision = 12;",
        f"Rep{name}.WriteHeaders = false;",
        f"Rep{name}.Add = {{{name}.ElapsedSecs, {name}.Luna.RMAG, "
        f"{name}.LunaInertial.VMAG, {name}.Luna.Energy}};",
        "",
    ]
    return lines


def _flight_blocks(name: str, schedule: np.ndarray, dt_s: float) -> list[str]:
    lines: list[str] = []
    i, n = 0, schedule.size
    while i < n:
        if schedule[i] == 1:
            lines += [
                f"vx = {name}.EarthMoonRot.VX;",
                f"vy = {name}.EarthMoonRot.VY;",
                "vn = sqrt( vx*vx + vy*vy );",
                f"{name}.Thr{name}.ThrustDirection1 = vx / vn;",
                f"{name}.Thr{name}.ThrustDirection2 = vy / vn;",
                f"{name}.Thr{name}.ThrustDirection3 = 0;",
                f"BeginFiniteBurn Burn{name}({name});",
                f"Propagate Prop({name}) {{{name}.ElapsedSecs = {dt_s:.9f}}};",
                f"EndFiniteBurn Burn{name}({name});",
            ]
            i += 1
        else:
            j = i
            while j < n and schedule[j] == 0:
                j += 1
            lines.append(f"Propagate Prop({name}) "
                         f"{{{name}.ElapsedSecs = {(j - i) * dt_s:.9f}}};")
            i = j
    lines.append("")
    return lines


def build_metrics_script(state0: np.ndarray) -> tuple[str, dict[str, str]]:
    dt_s = (T_SPAN[1] - T_SPAN[0]) * TIME_S / N_STEPS
    thrust_n = THRUST_MAG * ACCELERATION_M_S2 * 100.0
    rot0 = synodic_to_rotating_km(state0)
    names = {"coast": "SatCoast", "CR3BP design": "SatCR3BP",
             "GMAT QUBO": "SatQUBO"}

    lines = [
        "% Auto-generated by scripts/run_gmat_capture_metrics.py",
        "",
        "Create CoordinateSystem EarthMoonRot;",
        "EarthMoonRot.Origin = Earth;",
        "EarthMoonRot.Axes = ObjectReferenced;",
        "EarthMoonRot.XAxis = R;",
        "EarthMoonRot.ZAxis = N;",
        "EarthMoonRot.Primary = Earth;",
        "EarthMoonRot.Secondary = Luna;",
        "",
        "Create CoordinateSystem LunaInertial;",
        "LunaInertial.Origin = Luna;",
        "LunaInertial.Axes = MJ2000Eq;",
        "",
    ]
    for label, sat in names.items():
        lines += _sat_blocks(sat, rot0, bool(SCHEDULES[label].any()), thrust_n)

    lines += [
        "Create ForceModel FM;",
        "FM.CentralBody = Earth;",
        "FM.PointMasses = {Earth, Luna};",
        "",
        "Create Propagator Prop;",
        "Prop.FM = FM;",
        "Prop.Type = RungeKutta89;",
        "Prop.InitialStepSize = 60;",
        "Prop.Accuracy = 1e-12;",
        "Prop.MinStep = 0;",
        "Prop.MaxStep = 600;",
        "",
        "Create Variable vx vy vn;",
        "",
        "BeginMissionSequence;",
        "",
    ]
    for label, sat in names.items():
        lines += _flight_blocks(sat, SCHEDULES[label], dt_s)

    reports = {label: f"qalunar_capture_{sat}.txt"
               for label, sat in names.items()}
    return "\n".join(lines) + "\n", reports


# ---------------------------------------------------------------------------
# Run + parse
# ---------------------------------------------------------------------------


def _run_gmat(script_text: str, console: Path, timeout: float = 600.0) -> str:
    with tempfile.NamedTemporaryFile(
        "w", suffix=".script", delete=False, encoding="ascii"
    ) as fh:
        fh.write(script_text)
        path = Path(fh.name)
    try:
        proc = subprocess.run(
            [str(console), str(path)], cwd=str(console.parent),
            capture_output=True, text=True, timeout=timeout,
        )
        raw = (proc.stdout or "") + (proc.stderr or "")
        if not re.search(r"Mission run completed", raw):
            raise RuntimeError("GMAT run failed:\n"
                               + "\n".join(raw.splitlines()[-15:]))
        return raw
    finally:
        path.unlink(missing_ok=True)


def _read_report(console: Path, name: str) -> np.ndarray:
    for d in (console.parent.parent / "output", console.parent / "output",
              console.parent):
        p = d / name
        if p.exists():
            rows = [
                [float(v) for v in ln.split()]
                for ln in p.read_text().strip().splitlines() if ln.strip()
            ]
            p.unlink(missing_ok=True)
            return np.asarray(rows)
    raise FileNotFoundError(name)


def _capture_metrics_from_log(log: np.ndarray) -> dict:
    """SOI-crossing and perilune metrics from an (t, rmag, vmag, energy) log."""
    t, rmag, vmag, energy = log.T
    out: dict = {}

    below = np.flatnonzero(rmag <= MOON_SOI_KM)
    if below.size:
        k = int(below[0])
        out["soi_v_kms"] = vmag[k]
        out["soi_energy"] = energy[k]
    else:
        out["soi_v_kms"] = float("nan")
        out["soi_energy"] = float("nan")

    i = int(np.argmin(rmag))
    r_p, v_p, e_p = rmag[i], vmag[i], energy[i]
    v_circ = float(np.sqrt(GM_LUNA_KM3_S2 / r_p))
    v_esc = float(np.sqrt(2.0) * v_circ)
    out.update(
        peri_alt_km=r_p - MOON_RADIUS_KM,
        peri_v_ms=v_p * 1e3,
        peri_energy=e_p,
        v_circ_ms=v_circ * 1e3,
        v_esc_ms=v_esc * 1e3,
        capture_dv_ms=(v_p - v_circ) * 1e3,
        bound=bool(e_p < 0.0),
        v_inf_ms=float(np.sqrt(2.0 * e_p)) * 1e3 if e_p > 0 else 0.0,
        t_peri_d=t[i] / 86_400.0,
    )
    return out


def _capture_metrics_cr3bp(schedule: np.ndarray, state0: np.ndarray,
                           sched_cfg: ThrustSchedulingConfig) -> dict:
    """Same metrics along the in-house CR3BP propagation (nondim -> SI)."""
    dyn = PlanarCR3BP(mu=_MU)
    times, traj = propagate_schedule(
        dyn, state0, T_SPAN, schedule, sched_cfg,
        n_integration_substeps=300, return_trajectory=True,
    )
    rel = np.array([moon_relative_inertial(s, mu=_MU) for s in traj],
                   dtype=object)
    rmag_km = np.array([np.linalg.norm(rv[0]) for rv in rel]) * LENGTH_KM
    vmag_kms = np.array([np.linalg.norm(rv[1]) for rv in rel]) * VELOCITY_M_S / 1e3
    # two-body energy wrt Moon in km^2/s^2 with GMAT's GM for comparability
    energy = 0.5 * vmag_kms**2 - GM_LUNA_KM3_S2 / rmag_km
    log = np.column_stack([times * TIME_S, rmag_km, vmag_kms, energy])
    return _capture_metrics_from_log(log)


def main() -> None:
    console = find_gmat_console()
    if not console.exists():
        raise SystemExit(f"GmatConsole not found at {console}")

    sched_cfg = ThrustSchedulingConfig(
        thrust_magnitude=THRUST_MAG, thrust_direction="tangential",
    )
    state0 = np.concatenate([
        _R_SYN, np.array([0.0, _V_CIRC * V_RATIO]) - _OMEGA_CROSS_R,
    ])

    print("=" * 98)
    print("  Lunar-SOI arrival / capture metrics of the Phase-2 schedules"
          "  (epoch " + EPOCH + ")")
    print("=" * 98)

    script, reports = build_metrics_script(state0)
    _run_gmat(script, console)

    rows = []
    hdr = (f"  {'schedule':<14} {'model':<6} {'peri alt':>10} {'v_peri':>8} "
           f"{'v_circ':>8} {'v_esc':>8} {'cap dV':>8} {'E_moon':>9} "
           f"{'bound':>6} {'v@SOI':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for label, schedule in SCHEDULES.items():
        for model in ("CR3BP", "GMAT"):
            if model == "GMAT":
                m = _capture_metrics_from_log(
                    _read_report(console, reports[label]))
            else:
                m = _capture_metrics_cr3bp(schedule, state0, sched_cfg)
            rows.append(dict(schedule=label, model=model, **{
                k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in m.items()
            }))
            print(f"  {label:<14} {model:<6} "
                  f"{m['peri_alt_km']:>8,.0f} km "
                  f"{m['peri_v_ms']:>6,.0f} "
                  f"{m['v_circ_ms']:>6,.0f} "
                  f"{m['v_esc_ms']:>6,.0f} "
                  f"{m['capture_dv_ms']:>6,.0f} "
                  f"{m['peri_energy']:>9.4f} "
                  f"{str(m['bound']):>6} "
                  f"{m['soi_v_kms'] * 1e3:>6,.0f}")

    out = Path(__file__).resolve().parent / "figures" / "gmat_capture_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  units: alt km; speeds m/s; E_moon km^2/s^2; "
          f"cap dV = v_peri - v_circ (Phase-3 burden)")
    print(f"  table written to {out}")


if __name__ == "__main__":
    main()
