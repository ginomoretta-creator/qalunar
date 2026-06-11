"""Compare iterative QUBO scheduling against a continuous-thrust
direct-collocation NLP on the same cislunar correction scenarios.

The NLP (Hermite-Simpson direct transcription, the standard
aerospace-engineering approach to low-thrust optimal control) is
the appropriate "what would I do classically?" benchmark for the
paper, complementing the brute-force / MILP / SA classical solvers
that the methodology section already covers.

Reports for each scenario:
* QUBO (iterative): final miss (km), Δv (m/s), wall time
* NLP (Hermite-Simpson): final miss, Δv (m/s), wall time

The QUBO is a constrained-action solver (engine has fixed magnitude,
discrete switching); the NLP is unconstrained continuous thrust. The
expected pattern is that the NLP produces a slightly smaller miss at
slightly lower Δv but is bound to a continuous-thrust mission profile
that may not be physically realisable.

Outputs ``nlp_vs_qubo_comparison.png`` and a stdout table.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qalunar.dynamics import PlanarCR3BP
from qalunar.dynamics.cr3bp import EARTH_MOON_MU
from qalunar.qubo.scheduling_samplers import sample_brute_force
from qalunar.qubo.thrust_scheduling import (
    ThrustSchedulingConfig,
    propagate_schedule,
    solve_iterative,
)
from qalunar.reference.direct_collocation import (
    DirectCollocationConfig,
    solve_energy_optimal_cr3bp,
)
from qalunar.reference.edelbaum import LENGTH_KM, VELOCITY_M_S


_MU = EARTH_MOON_MU
_GM = 1.0 - _MU
_R_HEO = 200_000.0 / LENGTH_KM
_V_CIRC = float(np.sqrt(_GM / _R_HEO))
_R_SYN = np.array([-_MU + _R_HEO, 0.0])
_OMEGA_CROSS_R = np.array([-_R_SYN[1], _R_SYN[0]])

THRUST_MAG = 0.02


def _state_for_ratio(v_ratio: float) -> np.ndarray:
    return np.concatenate([
        _R_SYN,
        np.array([0.0, _V_CIRC * v_ratio]) - _OMEGA_CROSS_R,
    ])


def _bf_sampler(qubo) -> np.ndarray:
    return sample_brute_force(qubo).schedule


def _qubo_dv_m_s(N: int, T: float, state0: np.ndarray, target: np.ndarray) -> tuple[float, float, float, int]:
    """Run iterative QUBO scheduling. Returns (miss_km, dv_m_s, time_s, burns)."""
    cfg = ThrustSchedulingConfig(
        thrust_magnitude=THRUST_MAG,
        thrust_direction="tangential",
        fuel_weight=0.0,
    )
    t0 = time.perf_counter()
    result = solve_iterative(
        dyn, state0, target, (0.0, T),
        n_decision_steps=N, sampler=_bf_sampler, config=cfg,
        n_integration_substeps=80, n_truth_substeps=300,
        max_iters=6,
    )
    elapsed = time.perf_counter() - t0
    miss = float(np.linalg.norm(result.true_miss)) * LENGTH_KM
    dv = result.final_qubo.delta_v_m_s(result.schedule)
    burns = int(result.schedule.sum())
    return miss, dv, elapsed, burns


def _nlp_dv_m_s(T: float, state0: np.ndarray, target: np.ndarray) -> tuple[float, float, float, bool]:
    """Run Hermite-Simpson direct collocation. Returns (miss_km, dv_m_s, time_s, success).

    Δv = trapezoidal integral of ||u|| dt at the collocation nodes.
    Note that the NLP's *objective* is energy ((1/2)∫||u||² dt), not
    Δv, so it is not optimising the same cost as the QUBO; the Δv is
    reported for direct comparison.
    """
    cfg = DirectCollocationConfig(n_intervals=40, maxiter=400, tol=1e-9)
    r0 = state0[:2]
    v0 = state0[2:]
    rf = target[:2]
    vf = target[2:]
    t0 = time.perf_counter()
    res = solve_energy_optimal_cr3bp(dyn, r0, v0, rf, vf, T, cfg)
    elapsed = time.perf_counter() - t0
    if not res.success:
        return float("nan"), float("nan"), elapsed, False

    # Final-state miss is enforced exactly by the BC; max_bc_error tells us
    # how well that was achieved. Convert to km via the length unit.
    miss_km = float(res.max_bc_error) * LENGTH_KM

    # Δv = ∫|u| dt by trapezoidal rule on the node values.
    u_mag = np.sqrt(res.ux**2 + res.uy**2)
    h = T / cfg.n_intervals
    dv_nondim = float(np.trapezoid(u_mag, dx=h))
    dv_m_s = dv_nondim * VELOCITY_M_S
    return miss_km, dv_m_s, elapsed, True


if __name__ == "__main__":
    dyn = PlanarCR3BP()
    scenarios = [
        dict(v_ratio=1.187, T=2.0, label="under-injection, T=2"),
        dict(v_ratio=1.187, T=3.0, label="under-injection, T=3"),
        dict(v_ratio=1.185, T=3.0, label="strong under-injection, T=3"),
    ]
    N_QUBO = 15

    rows = []
    print(f"{'scenario':>30} {'method':>15} {'miss(km)':>10} "
          f"{'dV(m/s)':>10} {'time(s)':>10} {'extra':>20}")
    print("-" * 100)
    for sc in scenarios:
        state0 = _state_for_ratio(sc["v_ratio"])
        state_perfect = _state_for_ratio(1.190)
        _, traj = dyn.propagate(state_perfect, (0.0, sc["T"]), n_steps=8000)
        target = traj[-1]

        # QUBO
        miss_q, dv_q, t_q, burns = _qubo_dv_m_s(N_QUBO, sc["T"], state0, target)
        print(f"{sc['label']:>30} {'QUBO iterative':>15} "
              f"{miss_q:>10.1f} {dv_q:>10.1f} {t_q:>10.2f} "
              f"{f'{burns} burns/{N_QUBO}':>20}")

        # NLP
        miss_n, dv_n, t_n, ok = _nlp_dv_m_s(sc["T"], state0, target)
        print(f"{sc['label']:>30} {'NLP HS-direct':>15} "
              f"{miss_n:>10.4f} {dv_n:>10.1f} {t_n:>10.2f} "
              f"{'SLSQP' if ok else 'failed':>20}")

        rows.append(dict(scenario=sc["label"], qubo=(miss_q, dv_q, t_q, burns),
                         nlp=(miss_n, dv_n, t_n, ok)))

    # Plot: bar pairs for miss and Δv
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    names = [r["scenario"].replace(", ", "\n") for r in rows]
    miss_q = [r["qubo"][0] for r in rows]
    miss_n = [r["nlp"][0] for r in rows]
    dv_q = [r["qubo"][1] for r in rows]
    dv_n = [r["nlp"][1] for r in rows]
    x = np.arange(len(names))
    w = 0.35

    ax = axes[0]
    ax.bar(x - w/2, miss_q, w, label="QUBO iterative (binary)", color="tab:green", alpha=0.85)
    ax.bar(x + w/2, miss_n, w, label="NLP HS-direct (continuous)", color="tab:blue", alpha=0.85)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("final-state miss (km, log scale)")
    ax.set_title("Final miss: discrete-action vs continuous-control optimum")
    ax.grid(alpha=0.3, axis="y", which="both")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.bar(x - w/2, dv_q, w, label="QUBO iterative", color="tab:green", alpha=0.85)
    ax.bar(x + w/2, dv_n, w, label="NLP HS-direct", color="tab:blue", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Δv expended (m/s)")
    ax.set_title("Δv comparison")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9)

    out_path = Path(__file__).resolve().parent / "figures" / "nlp_vs_qubo_comparison.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nsaved: {out_path}")
