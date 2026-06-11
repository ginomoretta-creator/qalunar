"""Parametric robustness sweep for the iterative scheduling solver.

Holds thrust magnitude fixed at the nominal mission value and sweeps
over (injection-velocity ratio, time-of-flight). For each cell the
iterative re-linearization solver is run with a brute-force sampler
inside, and the achieved final-state miss, Δv, and burn count are
recorded.

Three heatmaps are produced:

* Final miss after iteration (log scale)
* Δv in m/s
* Burn count

The point: show that the iterative QUBO method is robust across a
realistic parameter region, and that the "feasibility frontier" (where
iteration cannot close the gap) is identifiable from the heatmap.

Outputs ``scheduling_robustness.png``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from qalunar.dynamics import PlanarCR3BP
from qalunar.dynamics.cr3bp import EARTH_MOON_MU
from qalunar.qubo.thrust_scheduling import (
    ThrustSchedulingConfig,
    solve_iterative,
)
from qalunar.reference.edelbaum import LENGTH_KM


_MU = EARTH_MOON_MU
_GM = 1.0 - _MU
_R_HEO = 200_000.0 / LENGTH_KM
_V_CIRC = float(np.sqrt(_GM / _R_HEO))

_R_SYN = np.array([-_MU + _R_HEO, 0.0])
_OMEGA_CROSS_R = np.array([-_R_SYN[1], _R_SYN[0]])

THRUST_MAG = 0.02
N = 15


def _state_for_ratio(v_ratio: float) -> np.ndarray:
    v_syn = np.array([0.0, _V_CIRC * v_ratio]) - _OMEGA_CROSS_R
    return np.concatenate([_R_SYN, v_syn])


def _bf_sampler(qubo) -> np.ndarray:
    return qubo.brute_force()[0]


def main() -> None:
    dyn = PlanarCR3BP()
    cfg = ThrustSchedulingConfig(
        thrust_magnitude=THRUST_MAG,
        thrust_direction="tangential",
        fuel_weight=0.0,
    )

    v_ratios = np.array([1.185, 1.187, 1.189, 1.191, 1.193])
    T_values = np.array([2.0, 2.5, 3.0, 3.5, 4.0])

    miss_grid = np.full((len(T_values), len(v_ratios)), np.nan)
    dv_grid = np.full_like(miss_grid, np.nan)
    burn_grid = np.full_like(miss_grid, np.nan)
    iter_grid = np.full_like(miss_grid, np.nan)

    print(f"Sweeping {len(v_ratios)} v_ratios x {len(T_values)} T values "
          f"= {len(v_ratios) * len(T_values)} cells, N={N}")
    print(f"{'v_ratio':>9} {'T':>6} {'miss(km)':>12} {'dV(m/s)':>10} "
          f"{'burns':>6} {'iters':>6}")
    print("-" * 60)

    for i_T, T in enumerate(T_values):
        for i_v, v_ratio in enumerate(v_ratios):
            state0_imp = _state_for_ratio(float(v_ratio))
            state0_perfect = _state_for_ratio(1.190)

            _, traj_perfect = dyn.propagate(
                state0_perfect, (0.0, float(T)), n_steps=8000,
            )
            target = traj_perfect[-1]

            t_span = (0.0, float(T))
            result = solve_iterative(
                dyn, state0_imp, target, t_span,
                n_decision_steps=N, sampler=_bf_sampler, config=cfg,
                n_integration_substeps=80, n_truth_substeps=300,
                max_iters=6,
            )
            miss = float(np.linalg.norm(result.true_miss))
            dv_m_s = result.final_qubo.delta_v_m_s(result.schedule)
            n_b = int(result.schedule.sum())

            miss_grid[i_T, i_v] = miss
            dv_grid[i_T, i_v] = dv_m_s
            burn_grid[i_T, i_v] = n_b
            iter_grid[i_T, i_v] = result.iterations

            print(
                f"{v_ratio:9.4f} {T:6.2f} "
                f"{miss * LENGTH_KM:12.1f} {dv_m_s:10.1f} "
                f"{n_b:6d} {result.iterations:6d}"
            )

    # ---- Plot ----
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)

    miss_km = miss_grid * LENGTH_KM
    extent = (
        v_ratios[0] - 0.001, v_ratios[-1] + 0.001,
        T_values[0] - 0.25, T_values[-1] + 0.25,
    )

    # Miss
    ax = axes[0]
    im = ax.imshow(
        miss_km,
        norm=LogNorm(vmin=max(miss_km.min(), 1.0), vmax=miss_km.max()),
        origin="lower", aspect="auto", extent=extent, cmap="magma_r",
    )
    fig.colorbar(im, ax=ax, label="miss (km)")
    ax.set_xticks(v_ratios)
    ax.set_yticks(T_values)
    ax.set_xlabel(r"$v_\mathrm{inj} / v_\mathrm{circ}$")
    ax.set_ylabel("Time of flight $T$ (nondim)")
    ax.set_title("Final-state miss (km, log scale)")
    for i_T, T in enumerate(T_values):
        for i_v, v in enumerate(v_ratios):
            ax.text(v, T, f"{miss_km[i_T, i_v]:,.0f}",
                    ha="center", va="center", fontsize=11,
                    fontweight="bold",
                    color="white" if miss_km[i_T, i_v] > miss_km.mean() else "black")

    # Δv
    ax = axes[1]
    im = ax.imshow(
        dv_grid, origin="lower", aspect="auto", extent=extent, cmap="viridis",
    )
    fig.colorbar(im, ax=ax, label="$\\Delta v$ (m/s)")
    ax.set_xticks(v_ratios)
    ax.set_yticks(T_values)
    ax.set_xlabel(r"$v_\mathrm{inj} / v_\mathrm{circ}$")
    ax.set_ylabel("Time of flight $T$ (nondim)")
    ax.set_title(r"$\Delta v$ expended (m/s)")
    for i_T, T in enumerate(T_values):
        for i_v, v in enumerate(v_ratios):
            ax.text(v, T, f"{dv_grid[i_T, i_v]:.1f}",
                    ha="center", va="center", fontsize=11,
                    fontweight="bold",
                    color="white" if dv_grid[i_T, i_v] < dv_grid.max() * 0.6 else "black")

    # Burns
    ax = axes[2]
    im = ax.imshow(
        burn_grid, origin="lower", aspect="auto", extent=extent,
        cmap="cividis", vmin=0, vmax=N,
    )
    fig.colorbar(im, ax=ax, label="burns")
    ax.set_xticks(v_ratios)
    ax.set_yticks(T_values)
    ax.set_xlabel(r"$v_\mathrm{inj} / v_\mathrm{circ}$")
    ax.set_ylabel("Time of flight $T$ (nondim)")
    ax.set_title(f"Active burns (out of {N})")
    for i_T, T in enumerate(T_values):
        for i_v, v in enumerate(v_ratios):
            ax.text(v, T, f"{int(burn_grid[i_T, i_v])}",
                    ha="center", va="center", fontsize=11,
                    fontweight="bold",
                    color="black" if burn_grid[i_T, i_v] >= N * 0.4 else "white")

    out_path = Path(__file__).resolve().parent / "figures" / "scheduling_robustness.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
