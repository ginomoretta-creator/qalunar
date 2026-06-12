"""Generate a GMAT GUI visualisation of the GMAT-in-the-loop result.

Recomputes the two schedules of ``run_gmat_oracle_demo`` (both are
deterministic) and writes a single GMAT script that flies four
spacecraft through the Phase-2 arc with an OrbitView in the Earth-Moon
rotating frame:

* ``SatCoast``  (gray)  - uncorrected injection, no burns
* ``RefSat``    (blue)  - perfectly-injected rendezvous reference
* ``SatCR3BP``  (red)   - the CR3BP-designed 3-burn schedule (open loop)
* ``SatQUBO``   (green) - the GMAT-linearised QUBO schedule

Open the generated script in the GMAT GUI (File > Open Script > Run),
or launch it directly:

    GMAT.exe --run scripts\\figures\\gmat_oracle_viz.script

Run:  python -m scripts.run_gmat_oracle_viz
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from qalunar.dynamics import PlanarCR3BP
from qalunar.dynamics.cr3bp import EARTH_MOON_MU
from qalunar.highfidelity import GmatOracleConfig, propagate_schedule_gmat, solve_gmat_iterative
from qalunar.highfidelity.gmat_oracle import synodic_to_rotating_km
from qalunar.qubo.scheduling_samplers import sample_brute_force
from qalunar.qubo.thrust_scheduling import ThrustSchedulingConfig, solve_iterative
from qalunar.reference.edelbaum import ACCELERATION_M_S2, LENGTH_KM, TIME_S


_MU = EARTH_MOON_MU
_GM = 1.0 - _MU
_R_HEO = 200_000.0 / LENGTH_KM
_V_CIRC = float(np.sqrt(_GM / _R_HEO))
_R_SYN = np.array([-_MU + _R_HEO, 0.0])
_OMEGA_CROSS_R = np.array([-_R_SYN[1], _R_SYN[0]])

V_RATIO = 1.187
V_RATIO_PERFECT = 1.190
THRUST_MAG = 0.02
T_SPAN = (0.0, 4.0)
N_STEPS = 15
EPOCH = "08 Jan 2026 00:00:00.000"

SATS = [
    # (name, color, uses burn hardware)
    ("SatCoast", "DimGray", False),
    ("RefSat", "Blue", False),
    ("SatCR3BP", "Red", True),
    ("SatQUBO", "Green", True),
]


def _bf_sampler(qubo) -> np.ndarray:
    return sample_brute_force(qubo).schedule


def _sat_definition(name: str, color: str, state_rot: np.ndarray,
                    burns: bool, thrust_n: float) -> list[str]:
    lines = [
        f"Create Spacecraft {name};",
        f"{name}.DateFormat = UTCGregorian;",
        f"{name}.Epoch = '{EPOCH}';",
        f"{name}.CoordinateSystem = EarthMoonRot;",
        f"{name}.DisplayStateType = Cartesian;",
        f"{name}.X = {state_rot[0]:.12f};",
        f"{name}.Y = {state_rot[1]:.12f};",
        f"{name}.Z = 0.0;",
        f"{name}.VX = {state_rot[2]:.15f};",
        f"{name}.VY = {state_rot[3]:.15f};",
        f"{name}.VZ = 0.0;",
        f"{name}.DryMass = 95;",
        f"{name}.OrbitColor = {color};",
    ]
    if burns:
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
    lines.append("")
    return lines


def _flight_blocks(name: str, schedule: np.ndarray, dt_s: float) -> list[str]:
    """Mission-sequence blocks flying ``schedule`` on spacecraft ``name``."""
    lines: list[str] = [f"% ---- {name}: schedule "
                        f"{''.join(str(int(b)) for b in schedule)} ----"]
    # merge consecutive coasts; burns re-evaluate direction per interval
    i = 0
    n = schedule.size
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
            seg = (j - i) * dt_s
            lines.append(
                f"Propagate Prop({name}) {{{name}.ElapsedSecs = {seg:.9f}}};"
            )
            i = j
    lines.append("")
    return lines


def build_viz_script(
    state_coast: np.ndarray,
    state_ref: np.ndarray,
    q_cr3bp: np.ndarray,
    q_qubo: np.ndarray,
) -> str:
    dt_s = (T_SPAN[1] - T_SPAN[0]) * TIME_S / N_STEPS
    thrust_n = THRUST_MAG * ACCELERATION_M_S2 * 100.0
    rot_coast = synodic_to_rotating_km(state_coast)
    rot_ref = synodic_to_rotating_km(state_ref)

    lines: list[str] = [
        "% Auto-generated by scripts/run_gmat_oracle_viz.py",
        "% Four-trajectory comparison of the GMAT-in-the-loop binary QUBO",
        "% scheduling experiment, in the Earth-Moon rotating frame.",
        "%   SatCoast (gray)  : uncorrected injection, no burns",
        "%   RefSat   (blue)  : perfect-injection rendezvous reference",
        "%   SatCR3BP (red)   : CR3BP 3-burn design flown open loop",
        "%   SatQUBO  (green) : GMAT-linearised QUBO schedule",
        "",
        "Create CoordinateSystem EarthMoonRot;",
        "EarthMoonRot.Origin = Earth;",
        "EarthMoonRot.Axes = ObjectReferenced;",
        "EarthMoonRot.XAxis = R;",
        "EarthMoonRot.ZAxis = N;",
        "EarthMoonRot.Primary = Earth;",
        "EarthMoonRot.Secondary = Luna;",
        "",
    ]

    states = {
        "SatCoast": rot_coast, "RefSat": rot_ref,
        "SatCR3BP": rot_coast, "SatQUBO": rot_coast,
    }
    for name, color, burns in SATS:
        lines += _sat_definition(name, color, states[name], burns, thrust_n)

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
        "Prop.MaxStep = 1000;",
        "",
        "Create OrbitView RotatingView;",
        "RotatingView.SolverIterations = Current;",
        "RotatingView.UpperLeft = [0.02 0.02];",
        "RotatingView.Size = [0.96 0.92];",
        "RotatingView.Add = {SatCoast, RefSat, SatCR3BP, SatQUBO, Earth, Luna};",
        "RotatingView.CoordinateSystem = EarthMoonRot;",
        "RotatingView.DrawObject = [true true true true true true];",
        "RotatingView.ViewPointReference = Earth;",
        "RotatingView.ViewPointVector = [192000 0 900000];",
        "RotatingView.ViewDirection = Earth;",
        "RotatingView.ViewScaleFactor = 1;",
        "RotatingView.ViewUpCoordinateSystem = EarthMoonRot;",
        "RotatingView.ViewUpAxis = Y;",
        "RotatingView.Axes = On;",
        "RotatingView.XYPlane = On;",
        "",
        "Create ReportFile Rep;",
        "Rep.Filename = 'gmat_oracle_viz_final_states.txt';",
        "Rep.Precision = 12;",
        "Rep.WriteHeaders = false;",
        "",
        "Create Variable vx vy vn;",
        "",
        "BeginMissionSequence;",
        "",
    ]

    coast = np.zeros(N_STEPS, dtype=np.int64)
    lines += _flight_blocks("SatCoast", coast, dt_s)
    lines += _flight_blocks("RefSat", coast, dt_s)
    lines += _flight_blocks("SatCR3BP", q_cr3bp, dt_s)
    lines += _flight_blocks("SatQUBO", q_qubo, dt_s)

    lines += [
        "Report Rep SatCoast.EarthMoonRot.X SatCoast.EarthMoonRot.Y "
        "RefSat.EarthMoonRot.X RefSat.EarthMoonRot.Y "
        "SatCR3BP.EarthMoonRot.X SatCR3BP.EarthMoonRot.Y "
        "SatQUBO.EarthMoonRot.X SatQUBO.EarthMoonRot.Y;",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    dyn = PlanarCR3BP(mu=_MU)
    sched_cfg = ThrustSchedulingConfig(
        thrust_magnitude=THRUST_MAG, thrust_direction="tangential",
    )
    oracle_cfg = GmatOracleConfig()

    state_ref = np.concatenate([
        _R_SYN, np.array([0.0, _V_CIRC * V_RATIO_PERFECT]) - _OMEGA_CROSS_R,
    ])
    state0 = np.concatenate([
        _R_SYN, np.array([0.0, _V_CIRC * V_RATIO]) - _OMEGA_CROSS_R,
    ])
    coast = np.zeros(N_STEPS, dtype=np.int64)

    _, traj_perfect = dyn.propagate(state_ref, T_SPAN, n_steps=8000)
    target_cr3bp = traj_perfect[-1]
    target_gmat = propagate_schedule_gmat(
        state_ref, T_SPAN, coast, sched_cfg, oracle_cfg,
    )

    print("[1/3] recomputing CR3BP design schedule ...")
    t0 = time.perf_counter()
    res_cr3bp = solve_iterative(
        dyn, state0, target_cr3bp, T_SPAN, n_decision_steps=N_STEPS,
        sampler=_bf_sampler, config=sched_cfg, max_iters=8,
        n_integration_substeps=80, n_truth_substeps=300,
    )
    print(f"      schedule {''.join(map(str, res_cr3bp.schedule))} "
          f"({time.perf_counter() - t0:.1f} s)")

    print("[2/3] recomputing GMAT-linearised schedule ...")
    t0 = time.perf_counter()
    res_qubo = solve_gmat_iterative(
        state0, target_gmat, T_SPAN, n_decision_steps=N_STEPS,
        sampler=_bf_sampler, sched_config=sched_cfg,
        oracle_config=oracle_cfg, max_iters=8,
    )
    print(f"      schedule {''.join(map(str, res_qubo.schedule))} "
          f"({time.perf_counter() - t0:.1f} s)")

    print("[3/3] writing GMAT visualisation script ...")
    script = build_viz_script(
        state0, state_ref, res_cr3bp.schedule, res_qubo.schedule,
    )
    out = Path(__file__).resolve().parent / "figures" / "gmat_oracle_viz.script"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(script, encoding="ascii")
    print(f"      written to {out}")
    print("\nOpen it in the GMAT GUI (File > Open Script, then Run), or:")
    print(f'  GMAT.exe --run "{out}"')


if __name__ == "__main__":
    main()
