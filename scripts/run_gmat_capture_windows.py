"""Phase-3 lunar capture in GMAT: sliding-window binary braking.

Continues the recalibrated Phase-2 scenario of
``run_gmat_capture_design``: the corrected 14-burn chaser arrives
hyperbolic at the lunar SOI (perilune ~4,000 km altitude, two-body
energy ~+0.12 km^2/s^2). This script chains short anti-tangential QUBO
windows -- the paper's sliding-window capture structure -- with **every
dynamical quantity supplied by GMAT**:

* per-window impulse-response b-vectors: single-bit-flip finite
  differences in GMAT (``solve_gmat_iterative``),
* per-window targets: local circular speed about the *real* Moon (the
  Earth-Moon distance at each window epoch is recovered from the GMAT
  probe itself, so the braking target tracks the true lunar position
  rather than the idealised CR3BP one),
* window chaining: each window's epoch advances along the real
  calendar (``epoch_plus_seconds``).

Capture is declared when the GMAT-reported Moon-relative two-body
energy goes negative; a final verification probe reads the energy
directly from GMAT.

Run:  python -m scripts.run_gmat_capture_windows
"""

from __future__ import annotations

import csv
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from qalunar.highfidelity import GmatOracleConfig, find_gmat_console, solve_gmat_iterative
from qalunar.highfidelity.gmat_oracle import (
    epoch_plus_seconds,
    rotating_km_to_synodic,
    synodic_to_rotating_km,
)
from qalunar.qubo.receding_horizon import CentralBody
from qalunar.qubo.scheduling_samplers import sample_brute_force
from qalunar.qubo.thrust_scheduling import ThrustSchedulingConfig
from qalunar.reference.edelbaum import (
    ACCELERATION_M_S2, LENGTH_KM, TIME_S, VELOCITY_M_S,
)

from scripts.run_gmat_capture_metrics import (
    EPOCH,
    GM_LUNA_KM3_S2,
    MOON_RADIUS_KM,
    N_STEPS,
    T_SPAN,
    THRUST_MAG,
    V_RATIO,
    _MU,
    _OMEGA_CROSS_R,
    _R_SYN,
    _V_CIRC,
    _flight_blocks,
    _read_report,
    _run_gmat,
    _sat_blocks,
)
from scripts.run_gmat_capture_design import _header_blocks, _force_prop_blocks


# Phase-2 result (deterministic output of run_gmat_capture_design.py).
Q_PHASE2 = np.array([int(b) for b in "111111111111110"], dtype=np.int64)

# Phase-3 window parameters (CR3BP mission-demo values adapted to the
# measured-linearisation setting: windows may be longer because the
# b-vectors are finite-differenced in the truth model itself).
ACCEL_MAIN_NONDIM = 1.286          # 350 mN on 100 kg
N_WIN = 10                         # binary slots per window
WINDOW_REVS = 0.5                  # fraction of local circular period
WINDOW_T_MAX = 0.04                # nondim cap (~4.2 h)
BRAKE_FACTOR = 1.0                 # target local circular speed
MAX_WINDOWS = 25
MAX_INNER_ITERS = 2
APPROACH_RMAG_KM = 20_000.0        # start Phase 3 here on the inbound leg
R_ESCAPE_KM = 0.25 * LENGTH_KM     # fail-safe: gave up if this far out

# Stop when the GMAT-reported Moon-relative energy is below this: a
# barely-bound orbit (E just under 0) has its apolune outside the SOI
# and is not operationally captured. -0.065 km^2/s^2 puts the apolune
# near the SOI boundary for a ~9,000 km perilune -- genuinely bound.
E_TARGET_KM2_S2 = -0.065

_LOG_COLS = ("ElapsedSecs", "Luna.RMAG", "LunaInertial.VMAG", "Luna.Energy",
             "EarthMoonRot.X", "EarthMoonRot.Y",
             "EarthMoonRot.VX", "EarthMoonRot.VY")


def _bf_sampler(qubo) -> np.ndarray:
    return sample_brute_force(qubo).schedule


def _add_list(name: str) -> str:
    """Comma-separated form for ``ReportFile.Add = {...}`` lists."""
    return ", ".join(f"{name}.{c}" for c in _LOG_COLS)


def _report_list(name: str) -> str:
    """Space-separated form for explicit ``Report`` commands."""
    return " ".join(f"{name}.{c}" for c in _LOG_COLS)


def _fly_log_states(schedule: np.ndarray, state0_syn: np.ndarray,
                    t_span_nd: tuple[float, float], cfg: GmatOracleConfig,
                    console: Path) -> np.ndarray:
    """Fly a schedule, auto-logging Moon-relative AND rotating-frame state."""
    n = schedule.size
    dt_s = (t_span_nd[1] - t_span_nd[0]) * TIME_S / n
    thrust_n = THRUST_MAG * ACCELERATION_M_S2 * 100.0
    name = "SatLog"
    lines = ["% Auto-generated approach log", ""]
    lines += _header_blocks()
    blocks = _sat_blocks(name, synodic_to_rotating_km(state0_syn),
                         bool(schedule.any()), thrust_n, epoch=cfg.epoch_utc)
    # widen the report column set
    blocks = [
        (f"Rep{name}.Add = {{{_add_list(name)}}};"
         if ln.startswith(f"Rep{name}.Add") else ln)
        for ln in blocks
    ]
    lines += blocks
    lines += _force_prop_blocks()
    lines += ["Create Variable vx vy vn;", "", "BeginMissionSequence;", ""]
    lines += _flight_blocks(name, schedule, dt_s)
    _run_gmat("\n".join(lines) + "\n", console)
    return _read_report(console, f"qalunar_capture_{name}.txt")


def _coast_probe(state_syn: np.ndarray, t_window_nd: float,
                 cfg: GmatOracleConfig, console: Path) -> np.ndarray:
    """Coast one window, reporting the wide column set at start and end."""
    name = "Probe"
    total_s = t_window_nd * TIME_S
    lines = ["% Auto-generated window probe", ""]
    lines += _header_blocks()
    blocks = _sat_blocks(name, synodic_to_rotating_km(state_syn), False, 0.0,
                         epoch=cfg.epoch_utc)
    blocks = [ln for ln in blocks if not ln.startswith(f"Rep{name}.Add")]
    lines += blocks
    lines += _force_prop_blocks()
    lines += [
        "BeginMissionSequence;",
        "",
        f"Report Rep{name} {_report_list(name)};",
        f"Propagate Prop({name}) {{{name}.ElapsedSecs = {total_s:.9f}}};",
        f"Report Rep{name} {_report_list(name)};",
    ]
    _run_gmat("\n".join(lines) + "\n", console)
    return _read_report(console, f"qalunar_capture_{name}.txt")


def _moon_distance_km(row: np.ndarray) -> float:
    """Earth-Moon distance from a log row (sat rotating X, Y + Luna RMAG).

    The Moon sits at (d, 0) in the Earth-centred rotating frame; with
    the sat at (X, Y) and |sat - Moon| = RMAG, d = X +/- sqrt(RMAG^2 -
    Y^2). The root inside the physical lunar-distance band is taken.
    """
    _, rmag, _, _, x, y, _, _ = row
    disc = max(rmag * rmag - y * y, 0.0)
    for d in (x + np.sqrt(disc), x - np.sqrt(disc)):
        if 340_000.0 < d < 420_000.0:
            return float(d)
    raise ValueError(f"no physical Earth-Moon distance root (row={row})")


def _moon_body(d_km: float) -> CentralBody:
    """CentralBody anchored at the real Moon's bridge-frame position."""
    return CentralBody(
        position=np.array([d_km / LENGTH_KM - _MU, 0.0]),
        gm=_MU, name="Moon",
    )


def _row_state_synodic(row: np.ndarray) -> np.ndarray:
    return rotating_km_to_synodic(np.array(row[4:8]))


def main() -> None:
    console = find_gmat_console()
    if not console.exists():
        raise SystemExit(f"GmatConsole not found at {console}")

    oracle0 = GmatOracleConfig()
    capture_cfg = ThrustSchedulingConfig(
        thrust_magnitude=ACCEL_MAIN_NONDIM,
        thrust_direction="antitangential",
        target_weights=np.array([0.0, 0.0, 1.0, 1.0]),
    )
    state_chaser = np.concatenate([
        _R_SYN, np.array([0.0, _V_CIRC * V_RATIO]) - _OMEGA_CROSS_R,
    ])

    print("=" * 96, flush=True)
    print("  Phase-3 lunar capture in GMAT: sliding-window anti-tangential "
          "binary braking", flush=True)
    print(f"  main thrust a = {ACCEL_MAIN_NONDIM} nondim (350 mN), "
          f"{N_WIN} slots/window, window cap {WINDOW_T_MAX} nondim, "
          f"brake factor {BRAKE_FACTOR}", flush=True)
    print("=" * 96, flush=True)

    # ---- locate the SOI approach point on the Phase-2 corrected arc ----
    print("\n[approach] flying the Phase-2 corrected schedule to find the "
          f"{APPROACH_RMAG_KM:,.0f} km inbound point ...", flush=True)
    log = _fly_log_states(Q_PHASE2, state_chaser, T_SPAN, oracle0, console)
    inbound = np.flatnonzero(log[:, 1] <= APPROACH_RMAG_KM)
    if not inbound.size:
        raise SystemExit("corrected arc never reaches the approach radius")
    k0 = int(inbound[0])
    row0 = log[k0]
    t_off_s = float(row0[0])
    state = _row_state_synodic(row0)
    d_now = _moon_distance_km(row0)
    print(f"  t = {t_off_s / 86400:.2f} d after Phase-2 start, "
          f"r_moon = {row0[1]:,.0f} km, v = {row0[2] * 1e3:,.0f} m/s, "
          f"E = {row0[3]:+.4f} km^2/s^2, d_EM = {d_now:,.0f} km", flush=True)

    rows: list[dict] = []
    total_dv = 0.0
    total_burns = 0
    captured = False

    for w in range(MAX_WINDOWS):
        cfg_w = replace(
            oracle0, epoch_utc=epoch_plus_seconds(EPOCH, t_off_s),
        )
        # Window sizing uses the bridge-frame Moon distance (adequate
        # for a timescale); all stop decisions use exact GMAT values
        # from the probe, because the bridge energy is unreliable near
        # E = 0 (a false CAPTURED was observed at E_gmat = +0.009).
        moon = _moon_body(d_now)
        t_w = min(WINDOW_REVS * moon.circular_period(moon.distance(state)),
                  WINDOW_T_MAX)

        probe = _coast_probe(state, t_w, cfg_w, console)
        r_km = float(probe[0][1])
        e_gmat = float(probe[0][3])

        if e_gmat < E_TARGET_KM2_S2:
            captured = True
            print(f"\n  window {w:2d}: E_moon = {e_gmat:+.5f} km^2/s^2 "
                  f"< {E_TARGET_KM2_S2} -> CAPTURED & BOUND "
                  f"(r = {r_km:,.0f} km)", flush=True)
            break
        if r_km > R_ESCAPE_KM:
            print(f"\n  window {w:2d}: receding at r = {r_km:,.0f} km "
                  "with E above target -> escape, capture failed", flush=True)
            break

        end_row = probe[-1]
        d_end = _moon_distance_km(end_row)
        end_state = _row_state_synodic(end_row)
        target = _moon_body(d_end).circular_speed_target(
            end_state, factor=BRAKE_FACTOR,
        )

        t0 = time.perf_counter()
        res = solve_gmat_iterative(
            state, target, (0.0, t_w), n_decision_steps=N_WIN,
            sampler=_bf_sampler, sched_config=capture_cfg,
            oracle_config=cfg_w, max_iters=MAX_INNER_ITERS, max_workers=6,
        )
        burns = int(res.schedule.sum())
        dv = burns * ACCEL_MAIN_NONDIM * (t_w / N_WIN) * VELOCITY_M_S
        total_dv += dv
        total_burns += burns

        state = res.true_final_state
        t_off_s += t_w * TIME_S
        d_now = d_end

        print(f"  window {w:2d}: r {r_km:>9,.0f} km  "
              f"E_gmat {e_gmat:+.5f}  "
              f"burns {burns:>2}/{N_WIN}  dV {dv:6.1f} m/s  "
              f"t_w {t_w * TIME_S / 3600:4.1f} h  "
              f"({time.perf_counter() - t0:4.0f} s)", flush=True)
        rows.append(dict(
            window=w, r_km=round(r_km, 1), energy_gmat=e_gmat,
            burns=burns, dv_ms=round(dv, 2),
            t_window_h=round(t_w * TIME_S / 3600, 3),
        ))

    # ---- authoritative final check straight from GMAT ----
    cfg_f = replace(oracle0, epoch_utc=epoch_plus_seconds(EPOCH, t_off_s))
    probe_f = _coast_probe(state, 60.0 / TIME_S, cfg_f, console)
    e_final = float(probe_f[0][3])
    r_final = float(probe_f[0][1])
    v_final = float(probe_f[0][2]) * 1e3
    sma_km = (-GM_LUNA_KM3_S2 / (2.0 * e_final)) if e_final < 0 else float("inf")

    print("\n" + "=" * 96, flush=True)
    print(f"  RESULT: {'CAPTURED' if captured and e_final < 0 else 'NOT captured'}"
          f"   E_moon = {e_final:+.5f} km^2/s^2 (GMAT)   "
          f"r = {r_final:,.0f} km ({r_final - MOON_RADIUS_KM:,.0f} km alt)   "
          f"v = {v_final:,.0f} m/s", flush=True)
    if e_final < 0:
        print(f"  bound orbit: sma = {sma_km:,.0f} km, "
              f"period = {2 * np.pi * np.sqrt(sma_km**3 / GM_LUNA_KM3_S2) / 3600:,.1f} h",
              flush=True)
    print(f"  Phase-3 totals: {len(rows)} active windows, "
          f"{total_burns} burns, dV = {total_dv:,.1f} m/s, "
          f"elapsed {(t_off_s - float(row0[0])) / 86400:.2f} d", flush=True)

    out = Path(__file__).resolve().parent / "figures" / "gmat_capture_windows.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out.open("w", newline="", encoding="utf-8") as fh:
            wcsv = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wcsv.writeheader()
            wcsv.writerows(rows)
        print(f"  window table written to {out}", flush=True)


if __name__ == "__main__":
    main()
