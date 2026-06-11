"""Linearization-error study for the on/off scheduling QUBO.

For a fixed problem (state0, target, thrust direction, N), sweep over
time-of-flight T and report:

* Predicted miss (linearized objective at the QUBO optimum)
* True nonlinear miss after a single-pass solve
* True nonlinear miss after iterative re-linearization
* Number of outer iterations needed to converge

Larger T means a longer arc and more linearization drift. The figure
shows when single-pass is sufficient and when the iterative loop is
necessary, justifying the methodology.

Outputs ``scheduling_linearization_error.png`` next to this script.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qalunar.dynamics import PlanarCR3BP
from qalunar.dynamics.cr3bp import EARTH_MOON_MU
from qalunar.qubo.thrust_scheduling import (
    ThrustSchedulingConfig,
    build_thrust_scheduling_qubo,
    propagate_schedule,
    solve_iterative,
)
from qalunar.reference.edelbaum import LENGTH_KM


# ---------------------------------------------------------------------------
# Scenario — same cislunar setup as run_thrust_scheduling.py
# ---------------------------------------------------------------------------

_MU = EARTH_MOON_MU
_GM = 1.0 - _MU
_R_HEO = 200_000.0 / LENGTH_KM
_V_CIRC = float(np.sqrt(_GM / _R_HEO))

_R_SYN = np.array([-_MU + _R_HEO, 0.0])
_OMEGA_CROSS_R = np.array([-_R_SYN[1], _R_SYN[0]])
_V_SYN_PERFECT = np.array([0.0, _V_CIRC * 1.190]) - _OMEGA_CROSS_R
_V_SYN_IMPERFECT = np.array([0.0, _V_CIRC * 1.187]) - _OMEGA_CROSS_R

STATE0 = np.concatenate([_R_SYN, _V_SYN_IMPERFECT])
STATE0_PERFECT = np.concatenate([_R_SYN, _V_SYN_PERFECT])
THRUST_MAG = 0.02
N = 15


def _bf_sampler(qubo) -> np.ndarray:
    return qubo.brute_force()[0]


def _target_for_T(dyn: PlanarCR3BP, T: float) -> np.ndarray:
    _, traj = dyn.propagate(STATE0_PERFECT, (0.0, T), n_steps=8000)
    return traj[-1]


def main() -> None:
    dyn = PlanarCR3BP()
    cfg = ThrustSchedulingConfig(
        thrust_magnitude=THRUST_MAG,
        thrust_direction="tangential",
        fuel_weight=0.0,
    )

    T_values = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

    rows = []
    print(f"{'T':>5} {'pred':>12} {'true_1pass':>12} {'true_iter':>12} {'iters':>5}")
    print("-" * 55)
    for T in T_values:
        target = _target_for_T(dyn, float(T))
        t_span = (0.0, float(T))

        # Single pass
        qubo = build_thrust_scheduling_qubo(
            dyn, STATE0, target, t_span,
            n_decision_steps=N, config=cfg,
            n_integration_substeps=100,
        )
        q_single, _ = qubo.brute_force()
        pred_miss = float(np.linalg.norm(qubo.miss_distance(q_single)))
        x_true = propagate_schedule(
            dyn, STATE0, t_span, q_single, cfg,
            n_integration_substeps=300,
        )
        true_single = float(np.linalg.norm(x_true - target))

        # Iterative
        result = solve_iterative(
            dyn, STATE0, target, t_span,
            n_decision_steps=N, sampler=_bf_sampler, config=cfg,
            n_integration_substeps=100, n_truth_substeps=300,
            max_iters=10,
        )
        true_iter = float(np.linalg.norm(result.true_miss))

        print(
            f"{T:5.2f} {pred_miss:12.4e} {true_single:12.4e} "
            f"{true_iter:12.4e} {result.iterations:5d}"
        )
        rows.append((T, pred_miss, true_single, true_iter, result.iterations))

    Ts, pred, single, it, iters = zip(*rows)
    Ts = np.array(Ts)
    pred = np.array(pred)
    single = np.array(single)
    it = np.array(it)
    iters = np.array(iters)

    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    # ---- Panel 1: miss vs T ----
    ax = axes[0]
    ax.semilogy(Ts, pred, "o--", color="tab:gray",
                lw=2.0, ms=10, label="Predicted (linearised)")
    ax.semilogy(Ts, single, "o-", color="tab:red",
                lw=2.4, ms=10, label="True miss, single pass")
    ax.semilogy(Ts, it, "D-", color="tab:green",
                lw=2.4, ms=11, label="True miss, iterative")

    ax.set_xlabel("Time of flight $T$ (nondim)")
    ax.set_ylabel(r"$\|$ final-state miss $\|$  (nondim)")
    ax.set_title("Linearisation gap and iterative correction")
    ax.grid(alpha=0.35, which="both")
    ax.legend(fontsize=12, loc="lower right")

    # Annotate the gap that motivates the iterative loop
    idx_worst = int(np.argmax(single / np.maximum(it, 1e-15)))
    if single[idx_worst] / max(it[idx_worst], 1e-15) >= 5.0:
        ax.annotate(
            f"$\\times {single[idx_worst] / max(it[idx_worst], 1e-15):.0f}$ reduction\n"
            f"by iteration",
            xy=(Ts[idx_worst], it[idx_worst]),
            xytext=(Ts[idx_worst] - 0.6, it[idx_worst] * 8.0),
            fontsize=12,
            arrowprops=dict(arrowstyle="->", color="tab:green", lw=1.6,
                            alpha=0.85),
            ha="center", fontweight="bold",
        )

    # ---- Panel 2: iterations required ----
    ax = axes[1]
    ax.bar(Ts, iters, width=0.32, color="tab:blue", alpha=0.85,
           edgecolor="white", linewidth=0.8)
    for t, n in zip(Ts, iters):
        if n > 0:
            ax.text(t, n + 0.06, f"{int(n)}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")
    ax.set_xlabel("Time of flight $T$ (nondim)")
    ax.set_ylabel("Outer iterations to convergence")
    ax.set_title("Cost of the iterative loop")
    ax.grid(alpha=0.35, axis="y")
    ax.set_ylim(0, max(int(iters.max()) + 1, 3))

    out_path = Path(__file__).resolve().parent / "figures" / "scheduling_linearization_error.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
