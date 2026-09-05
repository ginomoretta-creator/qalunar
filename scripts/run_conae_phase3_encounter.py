"""CONAE 12U CubeSat, Phase 3: lunar encounter and capture feasibility,
measured in GMAT from the end state of the binary Phase 2.

What it establishes
-------------------
1. **Encounter by waiting.** Phase 2 ends at apoapsis (~370,000 km) at an
   arbitrary lunar phase. The apogee direction is inertially fixed while the
   Moon sweeps past every 27.3 d, and the spacecraft returns to apogee every
   ~9.5 d, so the relative phase at successive apogees steps by ~125 deg and
   fills the circle in ~15 revolutions. Coasting (no propellant) until the
   first lunar-SOI entry therefore *is* the phasing strategy; its cost is
   waiting time, measured here.
2. **Arrival conditions.** At the first SOI entry the script coasts to
   periselene and reads, from GMAT, the perilune radius, Moon-relative speed
   and two-body energy, and derives the minimum braking dv to bind (speed to
   local escape speed).
3. **Brake capacity under the requirements.** The 40 mN thruster at the
   15 W orbit-averaged budget can fire ~3 % of the time; over a lunar pass of
   a few days that is tens of m/s. The ratio to the required dv is the
   feasibility verdict, stated as a number.
4. **Unconstrained binary capture (reference).** The SMART-1-style energy
   QUBO -- N slots over a window straddling periselene, per-slot Moon-relative
   energy change measured in GMAT, ``min (sum q dE - dE_needed)^2 + alpha sum q``
   -- is solved (exactly and by SA) and flown at 100 % duty, followed by a
   50-day coast, with the electrical energy of the schedule reported in kWh.
   This is what continuous-power hardware would do; the kWh figure says why a
   12U cannot.

Run:  python -m scripts.run_conae_phase3_encounter
"""

from __future__ import annotations

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

GM_LUNA = 4_902.8005821478
MOON_RADIUS_KM = 1_737.4
MOON_SOI_KM = 66_100.0
G0 = 9.80665

THRUST_N, ISP_S, DRY_MASS_KG = 0.040, 1200.0, 13.0
OPERATING_POWER_W, POWER_BUDGET_W = 493.5, 15.0
DUTY_MAX = POWER_BUDGET_W / OPERATING_POWER_W          # 0.0304

MAX_WAIT_DAYS = 150.0
CAPTURE_WINDOW_DAYS = 2.0
N_SLOTS = 16
E_TARGET = -0.010          # km^2/s^2, safely bound
FUEL_WEIGHT = 1e-5
POST_COAST_DAYS = 50.0

FIG_DIR = Path(__file__).resolve().parent / "figures"
CART = ("Sat.EarthMJ2000Eq.X Sat.EarthMJ2000Eq.Y Sat.EarthMJ2000Eq.Z "
        "Sat.EarthMJ2000Eq.VX Sat.EarthMJ2000Eq.VY Sat.EarthMJ2000Eq.VZ")


def _run(script: str, report: str, console: Path, timeout: float = 3600) -> np.ndarray:
    with tempfile.NamedTemporaryFile("w", suffix=".script", delete=False,
                                     encoding="ascii") as fh:
        fh.write(script)
        path = Path(fh.name)
    log = path.with_suffix(".log")
    try:
        proc = subprocess.run([str(console), "-l", str(log), "-r", str(path)],
                              cwd=str(console.parent), capture_output=True,
                              text=True, timeout=timeout)
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


def _header(state, epoch, fuel, brake: bool, report: str) -> str:
    thr = ""
    if brake:
        thr = f"""
Sat.Thrusters = {{Brk}};
Sat.PowerSystem = SolarP;

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
SolarP.InitialMaxPower = 2;
SolarP.AnnualDecayRate = 0;
SolarP.Margin = 0;
SolarP.ShadowModel = 'DualCone';
SolarP.ShadowBodies = {{Earth, Luna}};

Create FiniteBurn Brake;
Brake.Thrusters = {{Brk}};
"""
    return f"""Create Spacecraft Sat;
Sat.DateFormat = UTCGregorian;
Sat.Epoch = '{epoch}';
Sat.CoordinateSystem = EarthMJ2000Eq;
Sat.DisplayStateType = Cartesian;
Sat.X = {state[0]:.9f};
Sat.Y = {state[1]:.9f};
Sat.Z = {state[2]:.9f};
Sat.VX = {state[3]:.12f};
Sat.VY = {state[4]:.12f};
Sat.VZ = {state[5]:.12f};
Sat.DryMass = {DRY_MASS_KG};
Sat.Cr = 1.8;
Sat.SRPArea = 0.12;
Sat.Tanks = {{XeTank}};
{thr}
Create ElectricTank XeTank;
XeTank.AllowNegativeFuelMass = false;
XeTank.FuelMass = {fuel:.6f};

Create ForceModel FM;
FM.CentralBody = Earth;
FM.PrimaryBodies = {{Earth}};
FM.GravityField.Earth.Degree = 4;
FM.GravityField.Earth.Order = 4;
FM.PointMasses = {{Luna, Sun}};
FM.SRP = On;

Create Propagator Prop;
Prop.FM = FM;
Prop.Type = RungeKutta89;
Prop.InitialStepSize = 60;
Prop.Accuracy = 1e-11;
Prop.MinStep = 0;
Prop.MaxStep = 1800;

Create ReportFile Rep;
Rep.Filename = '{report}';
Rep.Precision = 12;
Rep.WriteHeaders = false;

Create Variable rmin tmin goflag niter;

BeginMissionSequence;
"""


def wait_for_encounter(state, epoch, fuel, console):
    rep = f"p3_wait_{uuid.uuid4().hex[:8]}.txt"
    script = _header(state, epoch, fuel, False, rep) + f"""
rmin = 1e9;
tmin = 0;
goflag = 1;
niter = 0;
While goflag > 0.5
   Propagate Prop(Sat) {{Sat.ElapsedSecs = 3600}};
   niter = niter + 1;
   If Sat.Luna.RMAG < rmin
      rmin = Sat.Luna.RMAG;
      tmin = Sat.ElapsedDays;
   EndIf
   goflag = 1;
   If Sat.Luna.RMAG < {MOON_SOI_KM:.1f}
      goflag = 0;
   EndIf
   If niter > {int(MAX_WAIT_DAYS * 24)}
      goflag = 0;
   EndIf
EndWhile
Report Rep Sat.ElapsedSecs {CART} Sat.XeTank.FuelMass Sat.Luna.RMAG Sat.Luna.Energy rmin tmin Sat.Earth.RadApo Sat.Earth.RadPer;
"""
    row = _run(script, rep, console)[-1]
    return {"elapsed_s": row[0], "state": row[1:7], "fuel": row[7], "rmag": row[8],
            "energy": row[9], "rmin": row[10], "tmin_days": row[11],
            "apogee": row[12], "perigee": row[13]}


def to_periselene(state, epoch, fuel, console):
    rep = f"p3_peri_{uuid.uuid4().hex[:8]}.txt"
    script = _header(state, epoch, fuel, False, rep) + f"""
Propagate Prop(Sat) {{Sat.Luna.Periapsis, Sat.ElapsedSecs = 600000}};
Report Rep Sat.ElapsedSecs {CART} Sat.XeTank.FuelMass Sat.Luna.RMAG Sat.Luna.VMAG Sat.Luna.Energy;
"""
    row = _run(script, rep, console)[-1]
    return {"elapsed_s": row[0], "state": row[1:7], "fuel": row[7], "rmag": row[8],
            "vmag": row[9], "energy": row[10]}


def _capture_script(state, epoch, fuel, schedule, window_s, post_coast_s, report):
    dt = window_s / len(schedule)
    seq, i, n = [], 0, len(schedule)
    while i < n:
        if schedule[i]:
            seq += ["BeginFiniteBurn Brake(Sat);",
                    f"Propagate Prop(Sat) {{Sat.ElapsedSecs = {dt:.4f}}};",
                    "EndFiniteBurn Brake(Sat);"]
            i += 1
        else:
            j = i
            while j < n and not schedule[j]:
                j += 1
            seq.append(f"Propagate Prop(Sat) {{Sat.ElapsedSecs = {(j - i) * dt:.4f}}};")
            i = j
    tail = f"Report Rep Sat.Luna.Energy Sat.Luna.RMAG Sat.XeTank.FuelMass Sat.ElapsedSecs;\n"
    if post_coast_s > 0:
        tail += (f"Propagate Prop(Sat) {{Sat.ElapsedSecs = {post_coast_s:.1f}}};\n"
                 f"Report Rep Sat.Luna.Energy Sat.Luna.RMAG Sat.XeTank.FuelMass Sat.ElapsedSecs;\n")
    return _header(state, epoch, fuel, True, report) + "\n".join(seq) + "\n" + tail


def fly_capture(state, epoch, fuel, schedule, window_s, console, post_coast_s=0.0):
    rep = f"p3_cap_{uuid.uuid4().hex[:8]}.txt"
    return _run(_capture_script(state, epoch, fuel, schedule, window_s, post_coast_s, rep),
                rep, console)


def main() -> None:
    console = find_gmat_console()
    if not console.exists():
        raise SystemExit(f"GmatConsole not found at {console}")
    summ = json.loads((FIG_DIR / "conae_phase2_binary_summary.json").read_text(encoding="utf-8"))
    if "final_state_eci_km_s" not in summ:
        raise SystemExit("Phase-2 summary lacks the final state; re-run run_conae_phase2_binary")
    state = np.asarray(summ["final_state_eci_km_s"], dtype=float)
    epoch = summ["final_epoch"]
    fuel = float(summ["final_fuel_kg"])
    out = {"phase2_end": {"epoch": epoch, "fuel_kg": fuel,
                          "apogee_km": summ["final_apogee_km"]}}

    print("=" * 92)
    print("  CONAE 12U CubeSat, Phase 3: lunar encounter and capture feasibility (GMAT)")
    print("=" * 92)
    print(f"  Phase-2 end: {epoch}, apogee {summ['final_apogee_km']:.0f} km, fuel {fuel:.3f} kg")

    # ---- 1. wait for the Moon --------------------------------------------
    print(f"\n[1] coasting up to {MAX_WAIT_DAYS:.0f} d for the first lunar-SOI entry ...", flush=True)
    w = wait_for_encounter(state, epoch, fuel, console)
    entered = w["rmag"] < MOON_SOI_KM
    print(f"    closest approach in the wait: {w['rmin']:,.0f} km at day {w['tmin_days']:.1f}; "
          f"SOI entry: {'yes' if entered else 'NO'} (final Moon distance {w['rmag']:,.0f} km "
          f"after {w['elapsed_s']/86400:.1f} d; Earth apogee {w['apogee']:,.0f} km)")
    out["wait"] = {"days": w["elapsed_s"] / 86400, "soi_entry": bool(entered),
                   "closest_km": w["rmin"], "closest_day": w["tmin_days"],
                   "energy_at_end": w["energy"]}
    if not entered:
        (FIG_DIR / "conae_phase3_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print("  no encounter within the wait budget; summary written")
        return

    state_soi = w["state"]
    epoch_soi = epoch_plus_seconds(epoch, w["elapsed_s"])
    fuel_soi = w["fuel"]

    # ---- 2. arrival conditions at periselene ---------------------------------
    p = to_periselene(state_soi, epoch_soi, fuel_soi, console)
    r_p, v_p, e_p = p["rmag"], p["vmag"], p["energy"]
    v_esc = np.sqrt(2 * GM_LUNA / r_p)
    v_circ = np.sqrt(GM_LUNA / r_p)
    dv_bind = max(v_p - v_esc, 0.0) * 1e3          # m/s to reach E = 0 at perilune
    dv_target = max(v_p - np.sqrt(2 * (E_TARGET + GM_LUNA / r_p)), 0.0) * 1e3
    soi_to_peri_s = p["elapsed_s"]
    print(f"\n[2] periselene {soi_to_peri_s/3600:.1f} h after SOI entry: r {r_p:,.0f} km "
          f"(alt {r_p-MOON_RADIUS_KM:,.0f} km), v {v_p:.3f} km/s, E {e_p:+.4f} km^2/s^2 "
          f"(v_esc {v_esc:.3f}, v_circ {v_circ:.3f})")
    print(f"    minimum braking to bind (E -> 0 at perilune): {dv_bind:.1f} m/s; "
          f"to E = {E_TARGET}: {dv_target:.1f} m/s")
    out["arrival"] = {"soi_epoch": epoch_soi, "soi_to_perilune_h": soi_to_peri_s / 3600,
                      "perilune_km": r_p, "perilune_alt_km": r_p - MOON_RADIUS_KM,
                      "speed_km_s": v_p, "energy_km2_s2": e_p, "v_esc_km_s": v_esc,
                      "dv_bind_m_s": dv_bind, "dv_target_m_s": dv_target}

    # ---- 3. brake capacity under the power budget --------------------------
    m = DRY_MASS_KG + fuel_soi
    pass_s = 2.0 * soi_to_peri_s                     # SOI in -> SOI out, roughly symmetric
    burn_s_allowed = DUTY_MAX * pass_s
    dv_allowed = THRUST_N * burn_s_allowed / m
    kwh_bind = OPERATING_POWER_W * (dv_bind * m / THRUST_N) / 3.6e6
    print(f"\n[3] SOI transit ~{pass_s/86400:.2f} d; at {DUTY_MAX:.2%} duty the thruster may fire "
          f"{burn_s_allowed/60:.0f} min -> {dv_allowed:.1f} m/s, i.e. {100*dv_allowed/max(dv_bind,1e-9):.1f} % "
          f"of the {dv_bind:.0f} m/s needed. Binding at 100 % duty would draw "
          f"{kwh_bind:.2f} kWh from storage.")
    out["capacity"] = {"soi_transit_days": pass_s / 86400, "burn_allowed_min": burn_s_allowed / 60,
                       "dv_allowed_m_s": dv_allowed, "fraction_of_required": dv_allowed / max(dv_bind, 1e-9),
                       "kwh_to_bind_at_full_duty": kwh_bind}

    # ---- 4. unconstrained binary capture (reference) ------------------------
    window_s = CAPTURE_WINDOW_DAYS * 86400.0
    # start the window so that periselene sits at its centre
    t_start = max(soi_to_peri_s - 0.5 * window_s, 0.0)
    if t_start > 0:
        rep = f"p3_pre_{uuid.uuid4().hex[:8]}.txt"
        pre = _run(_header(state_soi, epoch_soi, fuel_soi, False, rep) +
                   f"Propagate Prop(Sat) {{Sat.ElapsedSecs = {t_start:.1f}}};\n"
                   f"Report Rep Sat.ElapsedSecs {CART} Sat.XeTank.FuelMass;\n", rep, console)[-1]
        state_w, epoch_w, fuel_w = pre[1:7], epoch_plus_seconds(epoch_soi, t_start), pre[7]
    else:
        state_w, epoch_w, fuel_w = state_soi, epoch_soi, fuel_soi

    coast = fly_capture(state_w, epoch_w, fuel_w, np.zeros(N_SLOTS, int), window_s, console)[-1]
    e_coast = coast[0]
    print(f"\n[4] energy QUBO over a {CAPTURE_WINDOW_DAYS:.0f}-d window ({N_SLOTS} slots of "
          f"{window_s/N_SLOTS/3600:.1f} h) centred on periselene; coast end E {e_coast:+.4f}")
    print(f"    measuring per-slot dE in GMAT ({N_SLOTS} runs) ...", flush=True)

    def _slot(j):
        q = np.zeros(N_SLOTS, int); q[j] = 1
        return fly_capture(state_w, epoch_w, fuel_w, q, window_s, console)[-1][0] - e_coast
    with ThreadPoolExecutor(max_workers=6) as pool:
        dE = np.array(list(pool.map(_slot, range(N_SLOTS))))
    d_needed = E_TARGET - e_coast
    # exact ground state of  (sum q dE - d)^2 + alpha sum q  over 2^16 states
    best_q, best_f = None, np.inf
    for bits in itertools.product((0, 1), repeat=N_SLOTS):
        q = np.asarray(bits)
        f = (float(dE @ q) - d_needed) ** 2 + FUEL_WEIGHT * q.sum()
        if f < best_f:
            best_f, best_q = f, q
    # SA on the same QUBO
    import dimod, neal
    h = {i: float(dE[i] ** 2 - 2 * d_needed * dE[i] + FUEL_WEIGHT) for i in range(N_SLOTS)}
    J = {(i, j): float(2 * dE[i] * dE[j]) for i in range(N_SLOTS) for j in range(i + 1, N_SLOTS)}
    bqm = dimod.BinaryQuadraticModel(h, J, d_needed ** 2, dimod.BINARY)
    ss = neal.SimulatedAnnealingSampler().sample(bqm, num_reads=2000, seed=3)
    q_sa = np.array([int(ss.first.sample[i]) for i in range(N_SLOTS)])
    fly = fly_capture(state_w, epoch_w, fuel_w, best_q, window_s, console,
                      post_coast_s=POST_COAST_DAYS * 86400.0)
    e_cap, r_cap, fuel_cap = fly[-2][0], fly[-2][1], fly[-2][2]
    e_50, r_50 = fly[-1][0], fly[-1][1]
    dv_cap = ISP_S * G0 * np.log((DRY_MASS_KG + fuel_w) / (DRY_MASS_KG + fuel_cap))
    burn_s = best_q.sum() * window_s / N_SLOTS
    kwh = OPERATING_POWER_W * burn_s / 3.6e6
    print(f"    slots fired {''.join(map(str, best_q))} (exact) | SA {''.join(map(str, q_sa))} "
          f"{'= exact' if np.array_equal(q_sa, best_q) else '!= exact'}; predicted E "
          f"{e_coast + dE @ best_q:+.4f}, flown E {e_cap:+.4f} km^2/s^2, dv {dv_cap:.1f} m/s, "
          f"burn {burn_s/3600:.1f} h = {100*burn_s/window_s:.0f} % duty = {kwh:.1f} kWh")
    print(f"    after {POST_COAST_DAYS:.0f} d coast: E {e_50:+.4f}, Moon distance {r_50:,.0f} km "
          f"-> {'still bound' if (e_50 < 0 and r_50 < MOON_SOI_KM) else 'NOT captured'}")
    out["reference_capture_full_duty"] = {
        "window_days": CAPTURE_WINDOW_DAYS, "n_slots": N_SLOTS, "schedule": "".join(map(str, best_q)),
        "sa_matches_exact": bool(np.array_equal(q_sa, best_q)), "dE_per_slot": dE.tolist(),
        "energy_predicted": float(e_coast + dE @ best_q), "energy_flown": e_cap,
        "dv_m_s": dv_cap, "burn_hours": burn_s / 3600, "duty": burn_s / window_s, "kwh": kwh,
        "after_50d": {"energy": e_50, "moon_distance_km": r_50,
                      "bound_inside_soi": bool(e_50 < 0 and r_50 < MOON_SOI_KM)},
    }
    (FIG_DIR / "conae_phase3_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    with (FIG_DIR / "conae_phase3_slots.csv").open("w", newline="", encoding="utf-8") as fh:
        wtr = csv.writer(fh); wtr.writerow(["slot", "dE_km2_s2", "fires"])
        for j in range(N_SLOTS):
            wtr.writerow([j, f"{dE[j]:.6f}", int(best_q[j])])
    print("\n  summary written to scripts/figures/conae_phase3_summary.json")


if __name__ == "__main__":
    main()
