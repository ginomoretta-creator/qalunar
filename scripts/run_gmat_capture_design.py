"""Design the Phase-2 lunar approach IN GMAT: recalibrate, correct, capture.

``run_gmat_capture_metrics`` exposed that the CR3BP-calibrated injection
(``v/v_circ = 1.190``) does not survive real ephemerides: at the chosen
epoch its coast never enters the lunar SOI (closest approach >100,000 km
altitude). The mission objective -- arrive at the SOI with Moon-relative
energy and speed that Phase 3 can stabilise -- therefore requires the
*calibration itself* to be done in the truth model. This script runs the
full design pipeline in GMAT:

1. **Recalibrate the reference injection in GMAT.** Multi-probe coast
   scans over the injection ratio (one GMAT run per refinement level,
   all probes in a single script) find the ratio whose real-ephemeris
   coast reproduces the design perilune radius of the CR3BP reference.
2. **Correct the imperfect injection with the GMAT-linearised QUBO.**
   The chaser still starts at the mission's imperfect ratio 1.187; the
   binary schedule steers it onto the recalibrated reference endpoint
   (``solve_gmat_iterative``, all sensitivities and accept/reject in
   GMAT).
3. **Measure capture conditions of the corrected trajectory**: perilune
   altitude, Moon-relative speed and energy, capture Delta-v deficit,
   and the SOI-crossing speed -- the quantities Phase 3 actually needs.

Run:  python -m scripts.run_gmat_capture_design
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from qalunar.dynamics import PlanarCR3BP
from qalunar.highfidelity import GmatOracleConfig, find_gmat_console, solve_gmat_iterative
from qalunar.highfidelity.gmat_oracle import propagate_schedule_gmat, synodic_to_rotating_km
from qalunar.qubo.scheduling_samplers import sample_brute_force
from qalunar.qubo.thrust_scheduling import ThrustSchedulingConfig
from qalunar.reference.edelbaum import ACCELERATION_M_S2, TIME_S

from scripts.run_gmat_capture_metrics import (
    EPOCH,
    MOON_RADIUS_KM,
    N_STEPS,
    T_SPAN,
    THRUST_MAG,
    V_RATIO,
    _MU,
    _OMEGA_CROSS_R,
    _R_SYN,
    _V_CIRC,
    _capture_metrics_cr3bp,
    _capture_metrics_from_log,
    _flight_blocks,
    _read_report,
    _run_gmat,
    _sat_blocks,
)


V_RATIO_PERFECT_CR3BP = 1.190


def _bf_sampler(qubo) -> np.ndarray:
    return sample_brute_force(qubo).schedule


def _state_for_ratio(ratio: float) -> np.ndarray:
    return np.concatenate([
        _R_SYN, np.array([0.0, _V_CIRC * ratio]) - _OMEGA_CROSS_R,
    ])


def _header_blocks() -> list[str]:
    return [
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


def _force_prop_blocks() -> list[str]:
    return [
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
    ]


def _scan_level(ratios: list[float], console: Path) -> dict[float, np.ndarray]:
    """One multi-probe GMAT run: coast each ratio, return per-ratio logs."""
    total_s = (T_SPAN[1] - T_SPAN[0]) * TIME_S
    lines = ["% Auto-generated injection-ratio scan", ""] + _header_blocks()
    names = {}
    for k, ratio in enumerate(ratios):
        name = f"Probe{k}"
        names[ratio] = name
        rot = synodic_to_rotating_km(_state_for_ratio(ratio))
        lines += _sat_blocks(name, rot, False, 0.0)
    lines += _force_prop_blocks()
    lines += ["BeginMissionSequence;", ""]
    for ratio in ratios:
        lines.append(f"Propagate Prop({names[ratio]}) "
                     f"{{{names[ratio]}.ElapsedSecs = {total_s:.9f}}};")
    _run_gmat("\n".join(lines) + "\n", console)
    return {
        ratio: _read_report(console, f"qalunar_capture_{names[ratio]}.txt")
        for ratio in ratios
    }


def _recalibrate(console: Path, target_rmag_km: float,
                 verbose: bool = True) -> tuple[float, np.ndarray]:
    """Find the injection ratio whose GMAT coast matches the design perilune."""
    best_ratio, best_log = None, None
    centre, half, n_pts = 1.190, 0.010, 9
    for level in range(3):
        ratios = list(np.round(np.linspace(centre - half, centre + half, n_pts), 6))
        logs = _scan_level(ratios, console)
        scored = []
        for ratio, log in logs.items():
            rmin = float(log[:, 1].min())
            scored.append((abs(rmin - target_rmag_km), ratio, rmin, log))
        scored.sort()
        _, best_ratio, best_rmin, best_log = scored[0]
        if verbose:
            span = ", ".join(f"{r:.4f}:{float(l[:, 1].min()):,.0f}"
                             for r, l in sorted(logs.items()))
            print(f"    level {level}: ratio -> min RMAG km  [{span}]")
            print(f"    level {level} best: ratio={best_ratio:.6f} "
                  f"(min RMAG {best_rmin:,.0f} km, "
                  f"target {target_rmag_km:,.0f} km)")
        centre, half = best_ratio, half / (n_pts - 1) * 2.0
    return float(best_ratio), best_log


def _fly_and_log(schedule: np.ndarray, state0: np.ndarray,
                 console: Path) -> np.ndarray:
    """Fly one schedule in GMAT with Moon-relative logging, return the log."""
    dt_s = (T_SPAN[1] - T_SPAN[0]) * TIME_S / N_STEPS
    thrust_n = THRUST_MAG * ACCELERATION_M_S2 * 100.0
    name = "SatX"
    lines = ["% Auto-generated single-schedule capture log", ""]
    lines += _header_blocks()
    lines += _sat_blocks(name, synodic_to_rotating_km(state0),
                         bool(schedule.any()), thrust_n)
    lines += _force_prop_blocks()
    lines += ["Create Variable vx vy vn;", "", "BeginMissionSequence;", ""]
    lines += _flight_blocks(name, schedule, dt_s)
    _run_gmat("\n".join(lines) + "\n", console)
    return _read_report(console, f"qalunar_capture_{name}.txt")


def _print_metrics(label: str, m: dict) -> None:
    soi = (f"{m['soi_v_kms'] * 1e3:,.0f} m/s"
           if np.isfinite(m["soi_v_kms"]) else "never enters")
    print(f"  {label:<38} peri alt {m['peri_alt_km']:>9,.0f} km   "
          f"v_peri {m['peri_v_ms']:>6,.0f}   v_circ {m['v_circ_ms']:>5,.0f}   "
          f"cap dV {m['capture_dv_ms']:>6,.0f}   E {m['peri_energy']:>8.4f}   "
          f"SOI: {soi}")


def main() -> None:
    console = find_gmat_console()
    if not console.exists():
        raise SystemExit(f"GmatConsole not found at {console}")

    dyn = PlanarCR3BP(mu=_MU)
    sched_cfg = ThrustSchedulingConfig(
        thrust_magnitude=THRUST_MAG, thrust_direction="tangential",
    )
    oracle_cfg = GmatOracleConfig()
    coast = np.zeros(N_STEPS, dtype=np.int64)
    state_chaser = _state_for_ratio(V_RATIO)

    print("=" * 96)
    print("  Phase-2 lunar-approach design in GMAT: recalibrate -> "
          "QUBO-correct -> capture metrics")
    print(f"  epoch {EPOCH}, T = {T_SPAN[1]} nondim, N = {N_STEPS}, "
          f"a = {THRUST_MAG} nondim, chaser ratio = {V_RATIO}")
    print("=" * 96)

    # ---- design intent: the CR3BP reference perilune ----
    ref_cr3bp = _capture_metrics_cr3bp(
        coast, _state_for_ratio(V_RATIO_PERFECT_CR3BP), sched_cfg,
    )
    target_rmag = ref_cr3bp["peri_alt_km"] + MOON_RADIUS_KM
    print(f"\n[0] design intent (CR3BP reference, ratio "
          f"{V_RATIO_PERFECT_CR3BP}):")
    _print_metrics("CR3BP reference coast", ref_cr3bp)

    # ---- 1. recalibrate the injection in GMAT ----
    print("\n[1] recalibrating the injection ratio in GMAT "
          "(multi-probe coast scans):")
    ratio_star, ref_log = _recalibrate(console, target_rmag)
    ref_gmat = _capture_metrics_from_log(ref_log)
    print(f"\n    recalibrated ratio* = {ratio_star:.6f} "
          f"(CR3BP value was {V_RATIO_PERFECT_CR3BP})")
    _print_metrics("GMAT reference coast (ratio*)", ref_gmat)

    # ---- 2. QUBO-correct the imperfect injection onto the reference ----
    print("\n[2] GMAT-linearised QUBO: correct chaser (1.187) onto the "
          "recalibrated reference ...")
    state_ref = _state_for_ratio(ratio_star)
    target_state = propagate_schedule_gmat(
        state_ref, T_SPAN, coast, sched_cfg, oracle_cfg,
    )
    res = solve_gmat_iterative(
        state_chaser, target_state, T_SPAN, n_decision_steps=N_STEPS,
        sampler=_bf_sampler, sched_config=sched_cfg,
        oracle_config=oracle_cfg, max_iters=8, verbose=True,
    )
    schedule = res.schedule
    print(f"    schedule {''.join(map(str, schedule))} "
          f"({int(schedule.sum())} burns, {res.converged_reason})")

    # ---- 3. capture metrics of the corrected trajectory ----
    print("\n[3] capture conditions in GMAT (the mission objective):")
    log_uncorrected = _fly_and_log(coast, state_chaser, console)
    m_uncorrected = _capture_metrics_from_log(log_uncorrected)
    _print_metrics("chaser coast (uncorrected)", m_uncorrected)
    log_corrected = _fly_and_log(schedule, state_chaser, console)
    m_corrected = _capture_metrics_from_log(log_corrected)
    _print_metrics("chaser + GMAT-QUBO schedule", m_corrected)

    rows = [
        dict(case="CR3BP reference (design intent)", model="CR3BP",
             ratio=V_RATIO_PERFECT_CR3BP, **ref_cr3bp),
        dict(case="GMAT reference (recalibrated)", model="GMAT",
             ratio=ratio_star, **ref_gmat),
        dict(case="chaser coast (uncorrected)", model="GMAT",
             ratio=V_RATIO, **m_uncorrected),
        dict(case="chaser + GMAT-QUBO schedule", model="GMAT",
             ratio=V_RATIO, **m_corrected),
    ]
    out = Path(__file__).resolve().parent / "figures" / "gmat_capture_design.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  table written to {out}")

    if np.isfinite(m_corrected["soi_v_kms"]):
        print(f"\n  HEADLINE: under real ephemerides the corrected chaser "
              f"enters the lunar SOI at "
              f"{m_corrected['soi_v_kms'] * 1e3:,.0f} m/s and reaches "
              f"perilune at {m_corrected['peri_alt_km']:,.0f} km altitude "
              f"with a capture deficit of "
              f"{m_corrected['capture_dv_ms']:,.0f} m/s for Phase 3 -- "
              f"the design objective, delivered by "
              f"{int(schedule.sum())} binary burns.")
    else:
        print("\n  NOTE: corrected trajectory still does not enter the SOI; "
              "widen the QUBO authority or revisit the target.")


if __name__ == "__main__":
    main()
