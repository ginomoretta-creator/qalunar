"""GMAT-in-the-loop demo: the binary QUBO scheduler judged by real ephemerides.

Reproduces the Phase-2 long-arc cislunar correction (the paper's
iterative-vs-single-pass scenario: T = 4 nondim, N = 15, cruise thrust
a = 0.02) and runs it three ways:

1. **CR3BP design, CR3BP truth** — the paper baseline: the iterative
   QUBO converges against the in-house planar CR3BP propagator.
2. **CR3BP design flown open-loop in GMAT** — the schedule from (1) is
   flown once in NASA GMAT (DE-series ephemerides, real Moon). The
   gap between its CR3BP miss and its GMAT miss is the *fidelity gap*
   of the idealised design.
3. **GMAT in the loop, anchor only** — the trust-region accept/reject
   of ``solve_iterative`` evaluates candidates in GMAT and the QUBO's
   effective gap is anchored at the oracle's nominal endpoint, but the
   impulse-response sensitivities remain CR3BP. Expected to stall on
   this arc (ablation row).
4. **GMAT-linearised QUBO** — ``solve_gmat_iterative``: the b-vectors
   are measured by single-bit-flip finite differences in GMAT itself,
   so the QUBO is assembled entirely from high-fidelity data and the
   annealer only does the combinatorial search.

Each truth model gets its own rendezvous target: the propagation of
the *perfectly injected* reference under that same model. A CR3BP
target under ephemeris dynamics would absorb the entire model drift
(~700 m/s) and exceed the schedule's control authority by an order of
magnitude; flying reference and chaser in the same model makes the
problem the mission actually poses — correct the injection error.

The headline comparison is (2) vs (4): the CR3BP design flown
open-loop misses the real rendezvous by O(100,000) km, while the same
QUBO machinery with GMAT supplying both the sensitivities and the
trust-region judge recovers the correction.

Requires GmatConsole (R2025a); set QALUNAR_GMAT_CONSOLE if it is not
at the default path.

Run:  python -m scripts.run_gmat_oracle_demo
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from qalunar.dynamics import PlanarCR3BP
from qalunar.dynamics.cr3bp import EARTH_MOON_MU
from qalunar.highfidelity import (
    GmatOracleConfig,
    find_gmat_console,
    make_gmat_truth_propagator,
    propagate_schedule_gmat,
    solve_gmat_iterative,
)
from qalunar.qubo.scheduling_samplers import sample_brute_force
from qalunar.qubo.thrust_scheduling import (
    ThrustSchedulingConfig,
    propagate_schedule,
    solve_iterative,
)
from qalunar.reference.edelbaum import LENGTH_KM, TIME_S, VELOCITY_M_S


# ---- Phase-2 long-arc scenario (matches run_iterative_vs_singlepass) ----
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
MAX_ITERS = 8


def _bf_sampler(qubo) -> np.ndarray:
    return sample_brute_force(qubo).schedule


def _miss_breakdown(x_f: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """(position miss in km, velocity miss in m/s)."""
    miss = x_f - target
    pos_km = float(np.linalg.norm(miss[:2]) * LENGTH_KM)
    vel_ms = float(np.linalg.norm(miss[2:]) * VELOCITY_M_S)
    return pos_km, vel_ms


def _delta_v_ms(schedule: np.ndarray) -> float:
    dt_nd = (T_SPAN[1] - T_SPAN[0]) / N_STEPS
    return float(THRUST_MAG * dt_nd * int(np.sum(schedule)) * VELOCITY_M_S)


def main() -> None:
    console = find_gmat_console()
    if not console.exists():
        raise SystemExit(
            f"GmatConsole not found at {console}; set QALUNAR_GMAT_CONSOLE."
        )

    dyn = PlanarCR3BP(mu=_MU)
    sched_cfg = ThrustSchedulingConfig(
        thrust_magnitude=THRUST_MAG, thrust_direction="tangential",
    )
    oracle_cfg = GmatOracleConfig()
    gmat_truth = make_gmat_truth_propagator(oracle_cfg)

    state_perfect = np.concatenate([
        _R_SYN, np.array([0.0, _V_CIRC * V_RATIO_PERFECT]) - _OMEGA_CROSS_R,
    ])
    state0 = np.concatenate([
        _R_SYN, np.array([0.0, _V_CIRC * V_RATIO]) - _OMEGA_CROSS_R,
    ])
    coast = np.zeros(N_STEPS, dtype=np.int64)

    # Two rendezvous targets, one per truth model. The mission is
    # "rendezvous with the perfectly-injected reference after T": the
    # reference must fly the SAME dynamics as the chaser, otherwise the
    # target absorbs the entire ephemeris drift (~700 m/s of velocity
    # gap) and no schedule within the control authority can reach it.
    _, traj_perfect = dyn.propagate(state_perfect, T_SPAN, n_steps=8000)
    target_cr3bp = traj_perfect[-1]
    target_gmat = propagate_schedule_gmat(
        state_perfect, T_SPAN, coast, sched_cfg, oracle_cfg,
    )

    rows: list[dict] = []

    def add_row(
        label: str, schedule: np.ndarray, x_f: np.ndarray,
        target: np.ndarray,
    ) -> None:
        pos_km, vel_ms = _miss_breakdown(x_f, target)
        rows.append(dict(
            case=label,
            burns=int(np.sum(schedule)),
            delta_v_ms=round(_delta_v_ms(schedule), 2),
            pos_miss_km=round(pos_km, 1),
            vel_miss_ms=round(vel_ms, 2),
        ))
        print(f"  {label:<42} burns={int(np.sum(schedule)):>2}  "
              f"dV={_delta_v_ms(schedule):7.2f} m/s  "
              f"pos miss={pos_km:>11,.1f} km  vel miss={vel_ms:8.2f} m/s")

    print("=" * 78)
    print("  GMAT-in-the-loop binary scheduling (Phase-2 long-arc scenario)")
    print(f"  T = {T_SPAN[1]} nondim ({T_SPAN[1] * TIME_S / 86400:.1f} d), "
          f"N = {N_STEPS}, a = {THRUST_MAG} nondim, epoch {oracle_cfg.epoch_utc}")
    print("=" * 78)

    # ---- baselines: coast under both dynamics, each vs its own target ----
    print("\n[baselines]")
    x_coast_cr3bp = propagate_schedule(
        dyn, state0, T_SPAN, coast, sched_cfg, n_integration_substeps=300,
    )
    add_row("coast / CR3BP truth", coast, x_coast_cr3bp, target_cr3bp)
    x_coast_gmat = propagate_schedule_gmat(
        state0, T_SPAN, coast, sched_cfg, oracle_cfg,
    )
    add_row("coast / GMAT truth", coast, x_coast_gmat, target_gmat)

    # ---- (1) CR3BP design ----
    print("\n[1] iterative QUBO, CR3BP truth (paper baseline)")
    t0 = time.perf_counter()
    res_cr3bp = solve_iterative(
        dyn, state0, target_cr3bp, T_SPAN, n_decision_steps=N_STEPS,
        sampler=_bf_sampler, config=sched_cfg, max_iters=MAX_ITERS,
        n_integration_substeps=80, n_truth_substeps=300,
    )
    print(f"    {res_cr3bp.iterations} accepted iterations, "
          f"{time.perf_counter() - t0:.1f} s  ({res_cr3bp.converged_reason})")
    add_row("CR3BP design / CR3BP truth", res_cr3bp.schedule,
            res_cr3bp.true_final_state, target_cr3bp)

    # ---- (2) the same design flown open-loop in GMAT ----
    print("\n[2] CR3BP design flown OPEN LOOP in GMAT (fidelity gap)")
    x_open = propagate_schedule_gmat(
        state0, T_SPAN, res_cr3bp.schedule, sched_cfg, oracle_cfg,
    )
    add_row("CR3BP design / GMAT truth (open loop)", res_cr3bp.schedule,
            x_open, target_gmat)

    # ---- (3) GMAT in the loop, CR3BP sensitivities + truth anchor ----
    print("\n[3a] GMAT-in-loop, anchor only (CR3BP sensitivities), cold start")
    t0 = time.perf_counter()
    res_cold = solve_iterative(
        dyn, state0, target_gmat, T_SPAN, n_decision_steps=N_STEPS,
        sampler=_bf_sampler, config=sched_cfg, max_iters=MAX_ITERS,
        n_integration_substeps=80,
        truth_propagator=gmat_truth,
    )
    print(f"    {res_cold.iterations} accepted iterations, "
          f"{time.perf_counter() - t0:.1f} s  ({res_cold.converged_reason})")
    add_row("GMAT-in-loop anchor-only / cold", res_cold.schedule,
            res_cold.true_final_state, target_gmat)

    print("\n[3b] GMAT-in-loop, anchor only, warm start (CR3BP design seed)")
    t0 = time.perf_counter()
    res_warm = solve_iterative(
        dyn, state0, target_gmat, T_SPAN, n_decision_steps=N_STEPS,
        sampler=_bf_sampler, config=sched_cfg, max_iters=MAX_ITERS,
        n_integration_substeps=80,
        truth_propagator=gmat_truth,
        initial_schedule=res_cr3bp.schedule,
    )
    print(f"    {res_warm.iterations} accepted iterations, "
          f"{time.perf_counter() - t0:.1f} s  ({res_warm.converged_reason})")
    add_row("GMAT-in-loop anchor-only / warm", res_warm.schedule,
            res_warm.true_final_state, target_gmat)

    # ---- (4) fully GMAT-linearised QUBO ----
    print("\n[4a] GMAT-linearised QUBO (finite-difference b-vectors), cold start")
    t0 = time.perf_counter()
    res_fd_cold = solve_gmat_iterative(
        state0, target_gmat, T_SPAN, n_decision_steps=N_STEPS,
        sampler=_bf_sampler, sched_config=sched_cfg,
        oracle_config=oracle_cfg, max_iters=MAX_ITERS, verbose=True,
    )
    print(f"    {res_fd_cold.iterations} accepted iterations, "
          f"{time.perf_counter() - t0:.1f} s  ({res_fd_cold.converged_reason})")
    add_row("GMAT-in-loop GMAT-linearised / cold", res_fd_cold.schedule,
            res_fd_cold.true_final_state, target_gmat)

    print("\n[4b] GMAT-linearised QUBO, warm start (CR3BP design seed)")
    t0 = time.perf_counter()
    res_fd_warm = solve_gmat_iterative(
        state0, target_gmat, T_SPAN, n_decision_steps=N_STEPS,
        sampler=_bf_sampler, sched_config=sched_cfg,
        oracle_config=oracle_cfg, max_iters=MAX_ITERS,
        initial_schedule=res_cr3bp.schedule, verbose=True,
    )
    print(f"    {res_fd_warm.iterations} accepted iterations, "
          f"{time.perf_counter() - t0:.1f} s  ({res_fd_warm.converged_reason})")
    add_row("GMAT-in-loop GMAT-linearised / warm", res_fd_warm.schedule,
            res_fd_warm.true_final_state, target_gmat)

    # ---- summary ----
    out = Path(__file__).resolve().parent / "figures" / "gmat_oracle_demo.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  table written to {out}")

    best = min(
        (r for r in rows if "GMAT-in-loop" in r["case"]),
        key=lambda r: r["pos_miss_km"],
    )
    open_loop = next(r for r in rows if "open loop" in r["case"])
    print("\n  HEADLINE: open-loop CR3BP design lands "
          f"{open_loop['pos_miss_km']:,.0f} km from the target under real "
          f"ephemerides; the same QUBO machinery with GMAT in the loop lands "
          f"{best['pos_miss_km']:,.0f} km away "
          f"({best['case'].split('/')[-1].strip()}, "
          f"dV = {best['delta_v_ms']} m/s).")


if __name__ == "__main__":
    main()
