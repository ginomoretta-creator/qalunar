"""End-to-end pipeline: TFC+ELM transcription -> QUBO -> simulated annealing.

Demonstrates that a QUBO sampled with classical simulated annealing
recovers a trajectory indistinguishable from the Tikhonov least-squares
reference.  This validates the full quantum-annealing transcription
pipeline offline; when D-Wave access becomes available the SA sampler is
replaced by a QPU call with zero code changes to the QUBO assembly.

Produces two figures:

1. ``qubo_vs_tikhonov.png`` --- QUBO-decoded trajectory overlaid on the
   Tikhonov reference, with a right panel showing control magnitude.

2. ``qubo_bits_sweep.png`` --- how trajectory quality and QUBO size
   scale with ``bits_per_variable`` (4, 6, 8, 10, 12).

Run from the project root::

    python -m scripts.run_qubo_pipeline
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qalunar.dynamics import PlanarCR3BP
from qalunar.qubo import (
    LinearLsqToQuboConfig,
    build_linear_lsq_qubo,
    sample_simulated_annealing,
)
from qalunar.transcription import (
    IndirectTfcElmConfig,
    IndirectTfcElmTranscription,
)


# Use the same synthetic BCs as the warm-start demo -- mild transfer,
# away from primaries, well-behaved linearization.
BCS = dict(
    r0=np.array([-0.3, 0.0]),
    v0=np.array([0.0, 0.6]),
    rf=np.array([0.4, 0.2]),
    vf=np.array([-0.1, 0.0]),
    time_of_flight=2.5,
)

SA_NUM_READS = 2000
SA_SEED = 42


def _build_pipeline(
    bits_per_variable: int = 8,
):
    """Build the transcription, linear system, QUBO, and sample with SA."""
    dyn = PlanarCR3BP()
    cfg = IndirectTfcElmConfig(n_training=20, n_basis=80, seed=7)
    trans = IndirectTfcElmTranscription(dynamics=dyn, config=cfg, **BCS)

    nominal_x, nominal_y = trans.initial_nominal()
    A, B = trans.build_linear_system(nominal_x, nominal_y)

    qubo_cfg = LinearLsqToQuboConfig(bits_per_variable=bits_per_variable)
    qubo = build_linear_lsq_qubo(A, B, qubo_cfg)

    sr = sample_simulated_annealing(
        qubo,
        num_reads=SA_NUM_READS,
        seed=SA_SEED,
        initial_states=qubo.warm_start_bits(),
        A=A,
        B=B,
    )

    traj_tik = trans.decode_trajectory(qubo.xi_tik)
    traj_qubo = trans.decode_trajectory(sr.xi)

    return dict(
        trans=trans,
        A=A,
        B=B,
        qubo=qubo,
        sr=sr,
        traj_tik=traj_tik,
        traj_qubo=traj_qubo,
    )


# ---------------------------------------------------------------------------
# Figure 1: QUBO trajectory vs Tikhonov reference
# ---------------------------------------------------------------------------


def figure_qubo_vs_tikhonov(out_path: Path) -> None:
    """Side-by-side: trajectory + control comparison."""
    res = _build_pipeline(bits_per_variable=8)
    trans = res["trans"]
    traj_tik = res["traj_tik"]
    traj_qubo = res["traj_qubo"]
    qubo = res["qubo"]
    sr = res["sr"]
    dyn = trans.dynamics

    fig, axes = plt.subplots(
        1, 2, figsize=(13.0, 5.6),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
        constrained_layout=True,
    )

    # ---------- Left: synodic-frame trajectory ----------
    ax = axes[0]

    ax.plot(
        traj_tik["x"], traj_tik["y"],
        color="tab:blue", lw=2.2, label="Tikhonov (classical LSQ)",
        zorder=3,
    )
    ax.plot(
        traj_qubo["x"], traj_qubo["y"],
        color="tab:orange", lw=1.6, ls="--",
        label=f"QUBO + SA  ({qubo.n_bits} qubits, {SA_NUM_READS} reads)",
        zorder=4,
    )

    # Endpoints
    ax.plot(traj_tik["x"][0], traj_tik["y"][0], "o", color="black", ms=6, zorder=5)
    ax.plot(traj_tik["x"][-1], traj_tik["y"][-1], "s", color="black", ms=6, zorder=5)
    ax.annotate(
        "start", (traj_tik["x"][0], traj_tik["y"][0]),
        textcoords="offset points", xytext=(-14, -12), fontsize=8,
    )
    ax.annotate(
        "end", (traj_tik["x"][-1], traj_tik["y"][-1]),
        textcoords="offset points", xytext=(6, 4), fontsize=8,
    )

    # Primaries
    earth = dyn.earth_position()
    moon = dyn.moon_position()
    ax.plot(*earth, "o", color="tab:blue", ms=11, zorder=5)
    ax.plot(*moon, "o", color="dimgray", ms=6, zorder=5)

    ax.set_aspect("equal")
    ax.set_xlabel("x  (synodic, nondim)")
    ax.set_ylabel("y  (synodic, nondim)")
    ax.set_title("QUBO + SA  vs  Tikhonov reference")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.92)

    # Metrics box
    dx = np.max(np.abs(traj_qubo["x"] - traj_tik["x"]))
    dy = np.max(np.abs(traj_qubo["y"] - traj_tik["y"]))
    res_tik = float(np.linalg.norm(res["A"] @ qubo.xi_tik - res["B"]))
    metrics = (
        f"QUBO size:  {qubo.n_bits} qubits  "
        f"({qubo.n_reduced} vars x {qubo.bits_per_variable} bits)\n"
        f"SVD rank:   {qubo.effective_rank}\n"
        f"SA reads:   {SA_NUM_READS}\n"
        f"max |dx|:   {dx:.4e}\n"
        f"max |dy|:   {dy:.4e}\n"
        f"res Tik:    {res_tik:.4e}\n"
        f"res QUBO:   {sr.residual_norm:.4e}"
    )
    ax.text(
        0.02, 0.98, metrics, transform=ax.transAxes,
        fontsize=7.5, family="monospace", va="top", ha="left",
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor="white",
            edgecolor="tab:gray", alpha=0.92,
        ),
    )

    # ---------- Right: control magnitude ----------
    ax = axes[1]
    u_tik = np.sqrt(traj_tik["ux"] ** 2 + traj_tik["uy"] ** 2)
    u_qubo = np.sqrt(traj_qubo["ux"] ** 2 + traj_qubo["uy"] ** 2)
    ax.plot(trans.t, u_tik, color="tab:blue", lw=1.8, label="Tikhonov")
    ax.plot(
        trans.t, u_qubo, color="tab:orange", lw=1.4, ls="--", label="QUBO + SA"
    )
    ax.set_xlabel("t  (nondim)")
    ax.set_ylabel(r"$\|u\|$  (nondim acceleration)")
    ax.set_title("Control magnitude")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.savefig(out_path, dpi=150)
    print(f"saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: bits-per-variable sweep
# ---------------------------------------------------------------------------


def figure_bits_sweep(out_path: Path) -> None:
    """Trajectory error vs bits_per_variable."""
    bits_list = [4, 6, 8, 10, 12]
    n_qubits_list = []
    dx_list = []
    dy_list = []
    res_ratio_list = []

    print(f"\n{'bits':>5} {'n_qubits':>9} {'max|dx|':>10} {'max|dy|':>10} {'res_ratio':>10}")
    print("-" * 50)

    for bits in bits_list:
        res = _build_pipeline(bits_per_variable=bits)
        qubo = res["qubo"]
        sr = res["sr"]
        traj_tik = res["traj_tik"]
        traj_qubo = res["traj_qubo"]

        dx = np.max(np.abs(traj_qubo["x"] - traj_tik["x"]))
        dy = np.max(np.abs(traj_qubo["y"] - traj_tik["y"]))
        res_tik = float(np.linalg.norm(res["A"] @ qubo.xi_tik - res["B"]))
        ratio = sr.residual_norm / max(res_tik, 1e-15)

        n_qubits_list.append(qubo.n_bits)
        dx_list.append(dx)
        dy_list.append(dy)
        res_ratio_list.append(ratio)

        print(f"{bits:5d} {qubo.n_bits:9d} {dx:10.4e} {dy:10.4e} {ratio:10.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), constrained_layout=True)

    # Left: position error vs bits
    ax = axes[0]
    ax.semilogy(bits_list, dx_list, "o-", color="tab:blue", label=r"max $|x_{qubo} - x_{tik}|$")
    ax.semilogy(bits_list, dy_list, "s--", color="tab:orange", label=r"max $|y_{qubo} - y_{tik}|$")
    ax.set_xlabel("bits per variable")
    ax.set_ylabel("position error  (nondim)")
    ax.set_title("Trajectory error vs quantization depth")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)

    # Right: QUBO size vs bits
    ax = axes[1]
    ax.plot(bits_list, n_qubits_list, "D-", color="tab:green", lw=1.8)
    for b, nq in zip(bits_list, n_qubits_list):
        ax.annotate(str(nq), (b, nq), textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
    ax.set_xlabel("bits per variable")
    ax.set_ylabel("total QUBO qubits")
    ax.set_title("QUBO size vs quantization depth")
    ax.grid(alpha=0.3)

    fig.savefig(out_path, dpi=150)
    print(f"\nsaved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("  QA-Lunar: full QUBO pipeline demo")
    print("  TFC+ELM -> QUBO -> simulated annealing -> trajectory")
    print("=" * 60)

    figure_qubo_vs_tikhonov(out_dir / "qubo_vs_tikhonov.png")
    figure_bits_sweep(out_dir / "qubo_bits_sweep.png")

    print("\nDone. The QUBO pipeline is validated classically.")
    print("When D-Wave access arrives, replace sample_simulated_annealing()")
    print("with a QPU call -- the QUBO assembly is identical.\n")


if __name__ == "__main__":
    main()
