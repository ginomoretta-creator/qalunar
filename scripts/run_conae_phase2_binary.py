"""CONAE 12U CubeSat, Phase 2: apogee raising to lunar distance with a
duty-cycle-limited *binary* thrust schedule, GMAT-in-the-loop, one QUBO per
perigee pass (receding horizon).

Why this exists
---------------
The continuous-thrust spiral previously shown for Phase 2 (2187 m/s over 9.5
days at 100 % duty) violates both the thrust-window requirement (RQ-MIS-03,
<= 15 % of the period) and the orbit-averaged power budget (RQ-MIS-05, < 15 W)
by more than an order of magnitude. This script flies the phase the way the
vehicle actually can: on each perigee pass a thrust window of 15 % of the
period centred on perigee is split into N_WIN equal-time slots, the apogee
gain of every slot is measured in GMAT (finite differences in the truth
model), a QUBO with a cardinality budget K picks which slots fire, and the
chosen schedule is flown back in GMAT to advance the state. K is set by the
power budget: K * slot_duration <= (15 W / 493.5 W) * period.

Both requirements therefore hold on every pass by construction, and the
price -- many more revolutions -- is measured, not asserted.

Slot selection
--------------
Far from the target the objective is linear in the gains and the budget is
binding, so the QUBO is the cardinality form of :mod:`qalunar.qubo.cardinality`
(sampled with simulated annealing, cross-checked against top-K). On the final
passes, when the remaining apogee gap can be closed with fewer than K slots,
the objective becomes the squared miss (sum_j g_j q_j - d)^2 + alpha sum_j q_j
and is solved exactly over the <= K feasible set (N_WIN = 10 -> 1024 states),
so the phase ends at the target instead of overshooting it.

Run:  python -m scripts.run_conae_phase2_binary
      python -m scripts.run_conae_phase2_binary --perigee-alt 600 --fuel 3.45
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import subprocess
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from qalunar.highfidelity import find_gmat_console
from qalunar.highfidelity.gmat_oracle import epoch_plus_seconds
from qalunar.qubo.cardinality import solve_cardinality_qubo, top_k

MU_EARTH = 398_600.4418            # km^3/s^2
EARTH_RADIUS_KM = 6_378.137
G0 = 9.80665

# Vehicle (same as Phase 1)
THRUST_N = 0.040
ISP_S = 1200.0
DRY_MASS_KG = 13.0
XE_LOADED_KG = 3.5
OPERATING_POWER_W = 493.5
POWER_BUDGET_W = 15.0              # RQ-MIS-05 (orbit-averaged, strict <)
WINDOW_FRACTION = 0.15             # RQ-MIS-03 (thrust window, fraction of period)

# Orbit
APOGEE_RADIUS_KM = 76_922.0        # CONAE reference apogee (start of Phase 2)
INC_DEG, RAAN_DEG, AOP_DEG = 39.0, 0.0, 0.0
APO_TARGET_KM = 370_000.0          # lunar distance (Phase 3 handles the encounter)

# Scheduling
N_WIN = 10                         # slots inside the 15 % window
ALPHA = 1e-3                       # fuel tiebreaker in the final-pass objective (km^2 per burn)
MAX_PASSES = 120
MIN_FUEL_KG = 0.10

FIG_DIR = Path(__file__).resolve().parent / "figures"


# ---------------------------------------------------------------------------
# Orbital helpers (two-body, Earth-centred, km / s)
# ---------------------------------------------------------------------------


def keplerian_to_cartesian(a, e, i_deg, raan_deg, aop_deg, ta_deg):
    i, O, w, nu = np.deg2rad([i_deg, raan_deg, aop_deg, ta_deg])
    p = a * (1 - e * e)
    r = p / (1 + e * np.cos(nu))
    r_pf = np.array([r * np.cos(nu), r * np.sin(nu), 0.0])
    v_pf = np.sqrt(MU_EARTH / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])
    cO, sO, cw, sw, ci, si = np.cos(O), np.sin(O), np.cos(w), np.sin(w), np.cos(i), np.sin(i)
    R = np.array([
        [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
        [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
        [sw * si, cw * si, ci],
    ])
    return R @ r_pf, R @ v_pf


def elements_from_cartesian(r, v):
    rn, vn = np.linalg.norm(r), np.linalg.norm(v)
    a = 1.0 / (2.0 / rn - vn * vn / MU_EARTH)
    h = np.cross(r, v)
    e_vec = np.cross(v, h) / MU_EARTH - r / rn
    e = float(np.linalg.norm(e_vec))
    T = 2.0 * np.pi * np.sqrt(a ** 3 / MU_EARTH)
    return float(a), e, float(T)


def true_anomaly_after(e: float, T: float, dt: float) -> float:
    """True anomaly (deg) reached ``dt`` seconds after perigee (Kepler)."""
    M = 2.0 * np.pi * dt / T
    E = M if e < 0.8 else np.pi
    for _ in range(60):
        E -= (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    nu = 2.0 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    return float(np.rad2deg(nu) % 360.0)


# ---------------------------------------------------------------------------
# GMAT script generation
# ---------------------------------------------------------------------------


def _header(r, v, fuel_kg, epoch, has_burn):
    lines = [
        "Create Spacecraft Sat;",
        "Sat.DateFormat = UTCGregorian;",
        f"Sat.Epoch = '{epoch}';",
        "Sat.CoordinateSystem = EarthMJ2000Eq;",
        "Sat.DisplayStateType = Cartesian;",
        f"Sat.X = {r[0]:.9f};", f"Sat.Y = {r[1]:.9f};", f"Sat.Z = {r[2]:.9f};",
        f"Sat.VX = {v[0]:.12f};", f"Sat.VY = {v[1]:.12f};", f"Sat.VZ = {v[2]:.12f};",
        f"Sat.DryMass = {DRY_MASS_KG};",
        "Sat.Cd = 2.2;",
        "Sat.DragArea = 0.12;",
        "Sat.Cr = 1.8;",
        "Sat.SRPArea = 0.12;",
    ]
    if has_burn:
        lines += [
            "Sat.Tanks = {XeTank};",
            "Sat.Thrusters = {Hall};",
            "Sat.PowerSystem = SolarP;",
            "",
            "Create ElectricTank XeTank;",
            "XeTank.AllowNegativeFuelMass = false;",
            f"XeTank.FuelMass = {fuel_kg:.6f};",
            "",
            "Create ElectricThruster Hall;",
            "Hall.CoordinateSystem = Local;",
            "Hall.Origin = Earth;",
            "Hall.Axes = VNB;",
            "Hall.ThrustDirection1 = 1;",
            "Hall.ThrustDirection2 = 0;",
            "Hall.ThrustDirection3 = 0;",
            "Hall.DecrementMass = true;",
            "Hall.Tank = {XeTank};",
            "Hall.ThrustModel = ConstantThrustAndIsp;",
            f"Hall.ConstantThrust = {THRUST_N:.6f};",
            f"Hall.Isp = {ISP_S};",
            "Hall.MaximumUsablePower = 2;",
            "Hall.MinimumUsablePower = 0.01;",
            "",
            # Power system gates thrust on sunlight; the battery that feeds the
            # 493.5 W burn is book-kept analytically (orbit-averaged power).
            "Create SolarPowerSystem SolarP;",
            "SolarP.InitialMaxPower = 2;",
            "SolarP.AnnualDecayRate = 0;",
            "SolarP.Margin = 0;",
            "SolarP.ShadowModel = 'DualCone';",
            "SolarP.ShadowBodies = {Earth};",
            "",
            "Create FiniteBurn Burn;",
            "Burn.Thrusters = {Hall};",
        ]
    else:
        lines += ["Sat.Tanks = {XeTank};", "", "Create ElectricTank XeTank;",
                  f"XeTank.FuelMass = {fuel_kg:.6f};"]
    lines += [
        "",
        "Create ForceModel FM;",
        "FM.CentralBody = Earth;",
        "FM.PrimaryBodies = {Earth};",
        "FM.GravityField.Earth.Degree = 4;",
        "FM.GravityField.Earth.Order = 4;",
        "FM.PointMasses = {Luna, Sun};",
        "FM.SRP = On;",
        "FM.Drag.AtmosphereModel = 'MSISE90';",
        "FM.Drag.F107 = 150;",
        "FM.Drag.F107A = 150;",
        "FM.Drag.MagneticIndex = 3;",
        "",
        "Create Propagator Prop;",
        "Prop.FM = FM;",
        "Prop.Type = RungeKutta89;",
        "Prop.InitialStepSize = 30;",
        "Prop.Accuracy = 1e-11;",
        "Prop.MinStep = 0;",
        "Prop.MaxStep = 600;",
        "",
    ]
    return lines


REPORT_COLS = ("Sat.EarthMJ2000Eq.X Sat.EarthMJ2000Eq.Y Sat.EarthMJ2000Eq.Z "
               "Sat.EarthMJ2000Eq.VX Sat.EarthMJ2000Eq.VY Sat.EarthMJ2000Eq.VZ "
               "Sat.XeTank.FuelMass Sat.ElapsedSecs Sat.Earth.RadApo "
               "Sat.Earth.RadPer Sat.Earth.SMA Sat.Earth.ECC")


def _window_script(r, v, fuel, epoch, schedule, ta_start, slot_s, to_apoapsis, report):
    """Coast to the window start, fly ``schedule`` through the window, report;
    optionally continue to the next apoapsis and report again."""
    lines = ["% CONAE Phase 2 binary window", ""]
    lines += _header(r, v, fuel, epoch, bool(np.any(schedule)))
    lines += [f"Create ReportFile Rep;", f"Rep.Filename = '{report}';",
              "Rep.Precision = 12;", "Rep.WriteHeaders = false;", "",
              "BeginMissionSequence;", "",
              f"Propagate Prop(Sat) {{Sat.Earth.TA = {ta_start:.6f}}};"]
    i, n = 0, len(schedule)
    while i < n:
        if schedule[i]:
            lines += ["BeginFiniteBurn Burn(Sat);",
                      f"Propagate Prop(Sat) {{Sat.ElapsedSecs = {slot_s:.6f}}};",
                      "EndFiniteBurn Burn(Sat);"]
            i += 1
        else:
            j = i
            while j < n and not schedule[j]:
                j += 1
            lines.append(f"Propagate Prop(Sat) {{Sat.ElapsedSecs = {(j - i) * slot_s:.6f}}};")
            i = j
    lines.append(f"Report Rep {REPORT_COLS};")
    if to_apoapsis:
        lines += ["Propagate Prop(Sat) {Sat.Earth.Apoapsis};",
                  f"Report Rep {REPORT_COLS};"]
    return "\n".join(lines) + "\n"


def _run(script_text: str, report: str, console: Path) -> np.ndarray:
    with tempfile.NamedTemporaryFile("w", suffix=".script", delete=False,
                                     encoding="ascii") as fh:
        fh.write(script_text)
        path = Path(fh.name)
    log = path.with_suffix(".log")     # private log per run (parallel-safe)
    try:
        proc = subprocess.run([str(console), "-l", str(log), "-r", str(path)],
                              cwd=str(console.parent),
                              capture_output=True, text=True, timeout=1800)
        raw = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0 or not re.search(r"Mission run completed", raw):
            raise RuntimeError("GMAT failed:\n" + "\n".join(raw.splitlines()[-18:]))
        for d in (console.parent.parent / "output", console.parent / "output",
                  console.parent):
            p = d / report
            if p.exists():
                rows = [[float(x) for x in ln.split()]
                        for ln in p.read_text().strip().splitlines() if ln.strip()]
                p.unlink(missing_ok=True)
                return np.asarray(rows)
        raise FileNotFoundError(report)
    finally:
        path.unlink(missing_ok=True)
        log.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Slot selection
# ---------------------------------------------------------------------------


def select_slots(g: np.ndarray, gap_km: float, k: int, alpha: float):
    """Return (schedule, info). Cardinality QUBO while the budget is binding;
    exact squared-miss selection over the <= k feasible set once the gap can
    be closed with fewer slots."""
    ref = top_k(g, k)
    if float(g[ref == 1].sum()) < gap_km:
        q, info = solve_cardinality_qubo(g, k, num_reads=2000, seed=7)
        info["regime"] = "cardinality"
        return q, info
    n = g.size
    best_q, best_f = None, np.inf
    for bits in itertools.product((0, 1), repeat=n):
        q = np.asarray(bits, dtype=np.int64)
        if q.sum() > k:
            continue
        f = (float(g @ q) - gap_km) ** 2 + alpha * float(q.sum())
        if f < best_f:
            best_f, best_q = f, q
    return best_q, {"regime": "gap-closing (exact, <= k)", "objective": best_f,
                    "matches_top_k": bool(np.array_equal(best_q, ref))}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _phase1_handoff():
    """Perigee altitude, fuel and elapsed days at the end of Phase 1 from the
    cached climb, if available."""
    climb = FIG_DIR / "conae_climb.csv"
    if not climb.exists():
        return None
    rows = list(csv.DictReader(climb.open(newline="", encoding="utf-8")))
    if not rows or not rows[-1].get("fuel_kg"):
        return None
    last = rows[-1]
    return {"perigee_alt_km": float(last["perigee_radius_km"]) - EARTH_RADIUS_KM,
            "fuel_kg": float(last["fuel_kg"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perigee-alt", type=float, default=None,
                    help="perigee altitude at Phase-2 start (km); default: Phase-1 result")
    ap.add_argument("--fuel", type=float, default=None,
                    help="xenon on board at Phase-2 start (kg); default: Phase-1 result")
    ap.add_argument("--epoch", default="07 Jan 2026 00:00:00.000")
    ap.add_argument("--max-passes", type=int, default=MAX_PASSES)
    args = ap.parse_args()

    console = find_gmat_console()
    if not console.exists():
        raise SystemExit(f"GmatConsole not found at {console}")

    hand = _phase1_handoff()
    rp_alt = args.perigee_alt if args.perigee_alt is not None else (
        hand["perigee_alt_km"] if hand else 600.0)
    fuel = args.fuel if args.fuel is not None else (hand["fuel_kg"] if hand else 3.45)
    epoch = args.epoch

    rp = EARTH_RADIUS_KM + rp_alt
    a0 = 0.5 * (rp + APOGEE_RADIUS_KM)
    e0 = (APOGEE_RADIUS_KM - rp) / (APOGEE_RADIUS_KM + rp)
    r, v = keplerian_to_cartesian(a0, e0, INC_DEG, RAAN_DEG, AOP_DEG, 180.0)

    # Duty budget from the power requirement (strict <), rounded down.
    duty_max = POWER_BUDGET_W / OPERATING_POWER_W          # 0.0304
    slot_frac = WINDOW_FRACTION / N_WIN                    # 0.015
    K = int(np.floor(duty_max / slot_frac - 1e-12))        # 2

    print("=" * 92)
    print("  CONAE 12U CubeSat, Phase 2: binary duty-cycle-limited apogee raise (GMAT-in-the-loop)")
    print("=" * 92)
    print(f"  start: rp alt {rp_alt:.0f} km, ra {APOGEE_RADIUS_KM:.0f} km, a {a0:.0f} km, "
          f"e {e0:.4f}, fuel {fuel:.3f} kg, epoch {epoch}")
    print(f"  window {WINDOW_FRACTION:.0%} of period around perigee (RQ-MIS-03) in {N_WIN} slots; "
          f"budget K = {K} slots/pass = {K*slot_frac:.2%} duty = "
          f"{OPERATING_POWER_W*K*slot_frac:.1f} W orbit-averaged (RQ-MIS-05 < {POWER_BUDGET_W:.0f} W)")
    print(f"  target apogee {APO_TARGET_KM:.0f} km\n")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = FIG_DIR / "conae_phase2_binary.csv"
    fields = ["pass", "epoch", "elapsed_days", "a_km", "e", "period_h", "apogee_km",
              "perigee_alt_km", "fuel_kg", "dv_pass_m_s", "burn_s", "duty", "power_w",
              "regime", "schedule", "matches_top_k", "gain_pred_km", "gain_meas_km",
              "slot_burn_fraction_min"]
    rows_out = []
    elapsed_total = 0.0
    dv_total = 0.0
    a, e, T = elements_from_cartesian(r, v)
    apogee = a * (1 + e)

    for k in range(1, args.max_passes + 1):
        if apogee >= APO_TARGET_KM:
            break
        if fuel < MIN_FUEL_KG:
            print("  out of propellant")
            break
        a, e, T = elements_from_cartesian(r, v)
        theta_w = true_anomaly_after(e, T, 0.5 * WINDOW_FRACTION * T)
        ta_start = 360.0 - theta_w
        slot_s = WINDOW_FRACTION * T / N_WIN
        gap = APO_TARGET_KM - apogee

        def _one(j: int) -> np.ndarray:
            q = np.zeros(N_WIN, dtype=np.int64)
            if j >= 0:
                q[j] = 1
            rep = f"p2_{uuid.uuid4().hex[:10]}.txt"
            return _run(_window_script(r, v, fuel, epoch, q, ta_start, slot_s, False, rep),
                        rep, console)[-1]

        with ThreadPoolExecutor(max_workers=6) as pool:
            res = list(pool.map(_one, range(-1, N_WIN)))
        coast, slots = res[0], res[1:]
        apo_coast = coast[8]
        g = np.array([s[8] - apo_coast for s in slots])
        dm_full = THRUST_N * slot_s / (ISP_S * G0)
        burn_frac = np.array([(fuel - s[6]) / dm_full for s in slots])   # <1 => eclipsed

        q, info = select_slots(g, gap, K, ALPHA)
        rep = f"p2_{uuid.uuid4().hex[:10]}.txt"
        fly = _run(_window_script(r, v, fuel, epoch, q, ta_start, slot_s, True, rep),
                   rep, console)
        end_window, end_apo = fly[-2], fly[-1]
        gain_meas = end_window[8] - apo_coast
        gain_pred = float(g[q == 1].sum())
        fuel_new = float(end_apo[6])
        dv_pass = ISP_S * G0 * np.log((DRY_MASS_KG + fuel) / (DRY_MASS_KG + fuel_new))
        burn_s = float(q.sum()) * slot_s
        duty = burn_s / T
        r = end_apo[0:3].copy()
        v = end_apo[3:6].copy()
        dt = float(end_apo[7])
        epoch = epoch_plus_seconds(epoch, dt)
        elapsed_total += dt
        dv_total += dv_pass
        fuel = fuel_new
        a, e, T = elements_from_cartesian(r, v)
        apogee = a * (1 + e)

        row = dict(
            **{"pass": k}, epoch=epoch, elapsed_days=f"{elapsed_total/86400:.3f}",
            a_km=f"{a:.1f}", e=f"{e:.6f}", period_h=f"{T/3600:.3f}",
            apogee_km=f"{apogee:.1f}", perigee_alt_km=f"{a*(1-e)-EARTH_RADIUS_KM:.1f}",
            fuel_kg=f"{fuel:.6f}", dv_pass_m_s=f"{dv_pass:.3f}", burn_s=f"{burn_s:.1f}",
            duty=f"{duty:.5f}", power_w=f"{OPERATING_POWER_W*duty:.2f}",
            regime=info["regime"], schedule="".join(str(int(b)) for b in q),
            matches_top_k=info.get("matches_top_k"), gain_pred_km=f"{gain_pred:.2f}",
            gain_meas_km=f"{gain_meas:.2f}", slot_burn_fraction_min=f"{burn_frac.min():.3f}",
        )
        rows_out.append(row)
        print(f"  pass {k:3d}  {row['schedule']}  {info['regime'][:11]:11s} "
              f"apogee {apogee:9.0f} km  rp alt {float(row['perigee_alt_km']):6.0f} km  "
              f"T {T/3600:6.2f} h  dv {dv_pass:5.2f} m/s  duty {duty:.2%} "
              f"({OPERATING_POWER_W*duty:4.1f} W)  gain meas/pred {gain_meas:7.0f}/{gain_pred:7.0f} km"
              f"  fuel {fuel:.3f} kg  t {elapsed_total/86400:6.1f} d", flush=True)
        with out_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows_out)

    summary = {
        "passes": len(rows_out), "final_apogee_km": apogee,
        "reached_target": bool(apogee >= APO_TARGET_KM),
        "dv_total_m_s": dv_total, "elapsed_days": elapsed_total / 86400,
        "fuel_used_kg": (hand["fuel_kg"] if (hand and args.fuel is None) else
                         (args.fuel if args.fuel is not None else 3.45)) - fuel,
        "fuel_left_kg": fuel, "K": K, "N_WIN": N_WIN,
        "duty_per_pass": K * slot_frac, "power_w_per_pass": OPERATING_POWER_W * K * slot_frac,
        "final_epoch": epoch,
    }
    (FIG_DIR / "conae_phase2_binary_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print("\n  summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
