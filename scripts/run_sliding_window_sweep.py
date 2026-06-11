"""Parametric sensitivity sweep for the sliding-window QUBO solver.

For a fixed bound-eccentric initial state at the Moon (perilune
\\sim 4000 km altitude, 1.20 v_circ), sweep the three principal
sliding-window parameters:

    * ``window_t_max``       -- per-window time cap (nondim)
    * ``brake_factor``       -- target speed ratio (target = brake * v_circ)
    * ``moon_action_radius`` -- gating radius around the Moon

For each combination the script runs the solver, monitors capture
status, total dV, number of burns, final orbital energy, and final
apolune. The output is both a console table and a CSV file written
to ``scripts/sliding_window_sweep.csv``.

Run::

    python -m scripts.run_sliding_window_sweep
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

from qalunar.dynamics.cr3bp import EARTH_MOON_MU, PlanarCR3BP
from qalunar.qubo.lunar_capture import (
    moon_orbit_apolune_perilune,
    moon_two_body_energy,
    solve_lunar_capture_sliding_window,
)
from qalunar.qubo.scheduling_samplers import (
    sample_brute_force,
    sample_simulated_annealing,
)
from qalunar.reference.edelbaum import LENGTH_KM, VELOCITY_M_S


_MU = EARTH_MOON_MU
_MOON_POS = np.array([1.0 - _MU, 0.0])

# Fixed initial state: perilune at 4000 km altitude, 1.20 v_circ.
# Bound but eccentric (e ~ 0.69). Stable testbed for the sweep.
PERILUNE_ALT_KM = 4_000.0
EXCESS_FACTOR = 1.20

# Reference solver settings (mid of the sweep range, used for any param
# that is not currently being swept).
DEFAULT_THRUST = 0.02
DEFAULT_N_DECISION = 12
DEFAULT_N_WINDOWS = 60
DEFAULT_WINDOW_T_MAX = 0.04
DEFAULT_BRAKE = 1.00
DEFAULT_ACTION_R = 0.10


def build_initial_state(
    perilune_alt_km: float, excess_factor: float,
) -> np.ndarray:
    r = (1737.0 + perilune_alt_km) / LENGTH_KM
    pos = _MOON_POS + np.array([r, 0.0])
    v_circ = float(np.sqrt(_MU / r))
    v_rel = np.array([0.0, excess_factor * v_circ])
    v_moon_inertial = np.array([-_MOON_POS[1], _MOON_POS[0]])
    v_inertial = v_rel + v_moon_inertial
    omega_cross_r = np.array([-pos[1], pos[0]])
    return np.concatenate([pos, v_inertial - omega_cross_r])


def _sampler(qubo):
    if qubo.n_vars <= 18:
        return sample_brute_force(qubo).schedule
    return sample_simulated_annealing(
        qubo, num_reads=400, seed=42,
    ).schedule


@dataclass(frozen=True)
class SweepRow:
    label: str
    window_t_max: float
    brake_factor: float
    moon_action_radius: float
    captured: bool
    dv_m_s: float
    n_burns: int
    n_windows: int
    e_init: float
    e_final: float
    apo_init_km: float
    apo_final_km: float
    wall_time_s: float


def run_one(
    dyn: PlanarCR3BP, state0: np.ndarray, *,
    window_t_max: float, brake_factor: float,
    moon_action_radius: float, label: str,
) -> SweepRow:
    e0 = moon_two_body_energy(state0)
    apo0, _, _ = moon_orbit_apolune_perilune(state0)

    t0 = time.perf_counter()
    res = solve_lunar_capture_sliding_window(
        dyn, state0, sampler=_sampler,
        n_windows=DEFAULT_N_WINDOWS,
        window_revs=0.5,
        window_t_max=window_t_max,
        moon_action_radius=moon_action_radius,
        drift_t_max=0.3,
        n_decision_steps=DEFAULT_N_DECISION,
        thrust_magnitude=DEFAULT_THRUST,
        target_position_weight=0.0,
        target_velocity_weight=1.0,
        fuel_weight=0.0,
        brake_factor=brake_factor,
        target_radius=None,
        n_integration_substeps=60,
        n_truth_substeps=200,
        max_inner_iters=3,
        verbose=False,
    )
    wall = time.perf_counter() - t0

    e_final = moon_two_body_energy(res.final_state)
    apo_final, _, _ = moon_orbit_apolune_perilune(res.final_state)
    apo_final_km = (apo_final * LENGTH_KM
                    if np.isfinite(apo_final) else float("inf"))
    apo0_km = apo0 * LENGTH_KM if np.isfinite(apo0) else float("inf")

    return SweepRow(
        label=label,
        window_t_max=window_t_max,
        brake_factor=brake_factor,
        moon_action_radius=moon_action_radius,
        captured=res.captured,
        dv_m_s=res.total_delta_v * VELOCITY_M_S,
        n_burns=res.total_burns,
        n_windows=len(res.windows),
        e_init=e0,
        e_final=e_final,
        apo_init_km=apo0_km,
        apo_final_km=apo_final_km,
        wall_time_s=wall,
    )


def print_table(rows: list[SweepRow], group_label: str) -> None:
    print()
    print("=" * 78)
    print(f"  Sweep group: {group_label}")
    print("=" * 78)
    hdr = (
        f"{'param':>10} {'capt?':>6} {'dV (m/s)':>10} "
        f"{'burns':>6} {'win':>5} {'E_init':>10} {'E_fin':>10} "
        f"{'apo_init':>10} {'apo_fin':>10} {'wall(s)':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        # Pick the swept parameter for display
        if "window_t_max" in r.label:
            param = f"{r.window_t_max:.4f}"
        elif "brake" in r.label:
            param = f"{r.brake_factor:.2f}"
        elif "action" in r.label:
            param = f"{r.moon_action_radius:.3f}"
        else:
            param = "-"
        apo_init_str = (f"{r.apo_init_km:8.0f}"
                        if np.isfinite(r.apo_init_km) else "     inf")
        apo_fin_str = (f"{r.apo_final_km:8.0f}"
                       if np.isfinite(r.apo_final_km) else "     inf")
        print(
            f"{param:>10} "
            f"{('y' if r.captured else 'n'):>6} "
            f"{r.dv_m_s:>10.1f} "
            f"{r.n_burns:>6d} "
            f"{r.n_windows:>5d} "
            f"{r.e_init:>+10.3e} "
            f"{r.e_final:>+10.3e} "
            f"{apo_init_str:>10} "
            f"{apo_fin_str:>10} "
            f"{r.wall_time_s:>8.1f}"
        )


def write_csv(rows: list[SweepRow], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "group", "window_t_max", "brake_factor", "moon_action_radius",
            "captured", "dV_m_s", "n_burns", "n_windows",
            "e_init", "e_final", "apo_init_km", "apo_final_km",
            "wall_time_s",
        ])
        for r in rows:
            writer.writerow([
                r.label, r.window_t_max, r.brake_factor, r.moon_action_radius,
                int(r.captured), f"{r.dv_m_s:.2f}", r.n_burns, r.n_windows,
                f"{r.e_init:.6e}", f"{r.e_final:.6e}",
                f"{r.apo_init_km:.1f}", f"{r.apo_final_km:.1f}",
                f"{r.wall_time_s:.2f}",
            ])


def main() -> None:
    print("Initial state: perilune 4000 km, 1.20 v_circ (bound-eccentric)")
    print(f"Solver baseline: thrust=0.02 nondim, N_steps={DEFAULT_N_DECISION},"
          f" max_windows={DEFAULT_N_WINDOWS}")

    dyn = PlanarCR3BP()
    state0 = build_initial_state(PERILUNE_ALT_KM, EXCESS_FACTOR)

    all_rows: list[SweepRow] = []

    # ---- Sweep 1: window_t_max ---------------------------------
    rows_w = []
    for w in (0.005, 0.01, 0.02, 0.04, 0.08):
        r = run_one(
            dyn, state0,
            window_t_max=w,
            brake_factor=DEFAULT_BRAKE,
            moon_action_radius=DEFAULT_ACTION_R,
            label=f"window_t_max={w}",
        )
        rows_w.append(r)
    print_table(rows_w, "window_t_max")
    all_rows.extend(rows_w)

    # ---- Sweep 2: brake_factor ---------------------------------
    rows_b = []
    for b in (0.80, 0.90, 1.00, 1.10, 1.20):
        r = run_one(
            dyn, state0,
            window_t_max=DEFAULT_WINDOW_T_MAX,
            brake_factor=b,
            moon_action_radius=DEFAULT_ACTION_R,
            label=f"brake={b}",
        )
        rows_b.append(r)
    print_table(rows_b, "brake_factor")
    all_rows.extend(rows_b)

    # ---- Sweep 3: moon_action_radius ---------------------------
    rows_a = []
    for a in (0.04, 0.08, 0.10, 0.20, 0.50):
        r = run_one(
            dyn, state0,
            window_t_max=DEFAULT_WINDOW_T_MAX,
            brake_factor=DEFAULT_BRAKE,
            moon_action_radius=a,
            label=f"action_r={a}",
        )
        rows_a.append(r)
    print_table(rows_a, "moon_action_radius")
    all_rows.extend(rows_a)

    # ---- CSV --------------------------------------------------
    out_path = Path(__file__).resolve().parent / "figures" / "sliding_window_sweep.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(all_rows, out_path)
    print()
    print(f"CSV written to {out_path}")


if __name__ == "__main__":
    main()
