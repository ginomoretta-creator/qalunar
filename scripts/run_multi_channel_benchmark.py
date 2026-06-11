"""Multi-channel scheduling benchmark.

Compares single-channel (tangential only) and multi-channel
(tangential + normal + anti-tangential) iterative scheduling on
the same cislunar correction scenarios. The multi-channel solver
should achieve smaller miss when the gap is not aligned with the
spacecraft velocity.

Outputs:
* table to stdout
* ``scheduling_multi_channel_benchmark.png``
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qalunar.dynamics import PlanarCR3BP
from qalunar.dynamics.cr3bp import EARTH_MOON_MU
from qalunar.qubo.scheduling_samplers import (
    adapt_for_iterative,
    sample_brute_force,
    sample_simulated_annealing,
)
from qalunar.qubo.thrust_scheduling import (
    ThrustSchedulingConfig,
    propagate_schedule,
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
N_STEPS = 10   # same N for both modes for an apples-to-apples comparison


def _state_for_v(v_in_plane: np.ndarray) -> np.ndarray:
    return np.concatenate([_R_SYN, v_in_plane - _OMEGA_CROSS_R])


def _sampler(qubo) -> np.ndarray:
    """Use brute-force when feasible, else SA with high read count
    (so multi-channel quality is not bottlenecked by sampling)."""
    if qubo.n_vars <= 18:
        return sample_brute_force(qubo).schedule
    return sample_simulated_annealing(qubo, num_reads=4000, seed=42).schedule


def _scenarios() -> list[dict]:
    """Return three scenarios with progressively more
    velocity-misaligned correction gaps."""
    v_perfect = np.array([0.0, _V_CIRC * 1.190])

    scenarios = []

    # Scenario A: pure speed error (under-injection by 0.3%).
    v_a = np.array([0.0, _V_CIRC * 1.187])
    scenarios.append(dict(
        name="speed-only",
        v_in_plane=v_a,
        v_target=v_perfect,
        description="Under-injection (speed error only)",
    ))

    # Scenario B: lateral component error (perpendicular to nominal).
    v_b = np.array([_V_CIRC * 0.0015, _V_CIRC * 1.190])
    scenarios.append(dict(
        name="lateral",
        v_in_plane=v_b,
        v_target=v_perfect,
        description="Lateral velocity error (perpendicular to nominal)",
    ))

    # Scenario C: mixed.
    v_c = np.array([_V_CIRC * 0.0010, _V_CIRC * 1.187])
    scenarios.append(dict(
        name="mixed",
        v_in_plane=v_c,
        v_target=v_perfect,
        description="Mixed speed + lateral error",
    ))

    return scenarios


def _solve_one(
    dyn: PlanarCR3BP,
    state0: np.ndarray,
    target: np.ndarray,
    t_span: tuple[float, float],
    channels: tuple[str, ...],
    n_steps: int,
) -> dict:
    cfg = ThrustSchedulingConfig(
        thrust_magnitude=THRUST_MAG,
        thrust_channels=channels,
        fuel_weight=0.0,
    )
    result = solve_iterative(
        dyn, state0, target, t_span,
        n_decision_steps=n_steps, sampler=_sampler, config=cfg,
        n_integration_substeps=80, n_truth_substeps=300,
        max_iters=6,
    )
    miss = float(np.linalg.norm(result.true_miss))
    dv_m_s = result.final_qubo.delta_v_m_s(result.schedule)
    return dict(
        miss_km=miss * LENGTH_KM,
        miss_nondim=miss,
        dv_m_s=dv_m_s,
        burns=int(result.schedule.sum()),
        iters=result.iterations,
        n_vars=result.final_qubo.n_vars,
    )


def main() -> None:
    dyn = PlanarCR3BP()
    T = 3.0
    t_span = (0.0, T)

    scenarios = _scenarios()

    rows = []
    print(f"{'scenario':>14} {'mode':>16} {'qubits':>7} "
          f"{'miss(km)':>12} {'dV(m/s)':>10} {'burns':>6} {'iters':>5}")
    print("-" * 80)
    for sc in scenarios:
        state0 = _state_for_v(sc["v_in_plane"])
        state_perfect = _state_for_v(sc["v_target"])
        _, traj = dyn.propagate(state_perfect, t_span, n_steps=8000)
        target = traj[-1]

        # Single-channel: tangential only (10 vars, BF)
        single = _solve_one(
            dyn, state0, target, t_span, ("tangential",), N_STEPS,
        )
        # Multi-channel: 4 channels x N_STEPS = 40 vars (SA)
        multi = _solve_one(
            dyn, state0, target, t_span,
            ("tangential", "antitangential", "normal", "antinormal"),
            N_STEPS,
        )

        rows.append(dict(scenario=sc["name"], single=single, multi=multi))
        print(f"{sc['name']:>14} {'tangential':>16} {single['n_vars']:>7d} "
              f"{single['miss_km']:>12.1f} {single['dv_m_s']:>10.1f} "
              f"{single['burns']:>6d} {single['iters']:>5d}")
        print(f"{sc['name']:>14} {'4-channel':>16} {multi['n_vars']:>7d} "
              f"{multi['miss_km']:>12.1f} {multi['dv_m_s']:>10.1f} "
              f"{multi['burns']:>6d} {multi['iters']:>5d}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    names = [r["scenario"] for r in rows]
    miss_single = [r["single"]["miss_km"] for r in rows]
    miss_multi = [r["multi"]["miss_km"] for r in rows]
    dv_single = [r["single"]["dv_m_s"] for r in rows]
    dv_multi = [r["multi"]["dv_m_s"] for r in rows]

    x = np.arange(len(names))
    w = 0.35

    ax = axes[0]
    ax.bar(x - w/2, miss_single, w, label="Tangential only", color="tab:red", alpha=0.85)
    ax.bar(x + w/2, miss_multi, w, label="Tan + Nor + AntiTan", color="tab:green", alpha=0.85)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([rows[i]["scenario"] for i in range(len(rows))])
    ax.set_ylabel("final-state miss (km, log scale)")
    ax.set_title("Multi-channel reduces miss when gap is mis-aligned with velocity")
    ax.grid(alpha=0.3, axis="y", which="both")
    ax.legend(fontsize=9)

    # Annotate ratios
    for i, (s, m) in enumerate(zip(miss_single, miss_multi)):
        if s > 0 and m > 0:
            ratio = s / m
            if ratio >= 1.5:
                ax.annotate(f"{ratio:.0f}x", xy=(i, max(s, m) * 1.4),
                            ha="center", fontsize=10, color="tab:gray")

    ax = axes[1]
    ax.bar(x - w/2, dv_single, w, label="Tangential only", color="tab:red", alpha=0.85)
    ax.bar(x + w/2, dv_multi, w, label="Tan + Nor + AntiTan", color="tab:green", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([rows[i]["scenario"] for i in range(len(rows))])
    ax.set_ylabel("Δv expended (m/s)")
    ax.set_title("Δv cost across scenarios")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9)

    out_path = Path(__file__).resolve().parent / "figures" / "scheduling_multi_channel_benchmark.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
