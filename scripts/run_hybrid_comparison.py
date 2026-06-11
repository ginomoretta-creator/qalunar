"""Compare classical vs hybrid QUBO convergence on the synthetic benchmark.

Runs three solvers on the same BCs and starting nominal:

1. **Classical** --- ``solve_sequential`` with Armijo line search (lstsq)
2. **Hybrid ground-state** --- QUBO + SA, always use lowest-energy sample
3. **Hybrid reranked** --- QUBO + SA, rerank top-k by nonlinear residual

Produces four figures:

1. ``hybrid_convergence.png`` --- nonlinear TPBVP residual vs iteration
   for all three solvers (cold start from Hermite interpolant).
2. ``hybrid_rerank_stats.png`` --- per-iteration rank of the reranking
   winner and the NL-residual improvement over the ground state.
3. ``hybrid_scaling.png`` --- wall-clock time vs problem size (Path A),
   comparing classical lstsq solve time vs QUBO build+SA sample time.
4. ``hybrid_warm_convergence.png`` --- warm-started convergence from
   the direct-NLP reference, where the classical solver hits a plateau
   and reranking is expected to break through.

Run from the project root::

    python -m scripts.run_hybrid_comparison
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qalunar.dynamics import PlanarCR3BP
from qalunar.hybrid import HybridConfig, solve_hybrid
from qalunar.qubo import LinearLsqToQuboConfig, build_linear_lsq_qubo
from qalunar.qubo.samplers import sample_simulated_annealing
from qalunar.reference.direct_collocation import (
    DirectCollocationConfig,
    solve_energy_optimal_cr3bp,
)
from qalunar.transcription import (
    IndirectTfcElmConfig,
    IndirectTfcElmTranscription,
    OuterLoopConfig,
)


SYNTHETIC_BCS = dict(
    r0=np.array([-0.3, 0.0]),
    v0=np.array([0.0, 0.6]),
    rf=np.array([0.4, 0.2]),
    vf=np.array([-0.1, 0.0]),
    time_of_flight=2.5,
)

MAX_ITER = 20
NUM_READS = 200
BITS = 8
RERANK_TOP_K = 30
N_TRAINING = 12
N_BASIS = 40
DAMPING = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_transcription(
    n_training: int = N_TRAINING, n_basis: int = N_BASIS
) -> IndirectTfcElmTranscription:
    cfg = IndirectTfcElmConfig(n_training=n_training, n_basis=n_basis, seed=7)
    return IndirectTfcElmTranscription(
        dynamics=PlanarCR3BP(), config=cfg, **SYNTHETIC_BCS
    )


# ---------------------------------------------------------------------------
# Figure 1: Convergence comparison
# ---------------------------------------------------------------------------


def figure_convergence(out_path: Path) -> None:
    print("\n--- Running classical solver ---")
    trans = _make_transcription()
    t0 = time.perf_counter()
    sol_classical = trans.solve_sequential(
        OuterLoopConfig(
            max_iter=MAX_ITER, tol=1e-7, damping=1.0, line_search=True
        )
    )
    t_classical = time.perf_counter() - t0
    print(
        f"  {sol_classical.n_iterations} iters, "
        f"final NL res = {sol_classical.history['nonlinear_residual'][-1]:.4e}, "
        f"{t_classical:.1f}s"
    )

    print("\n--- Running hybrid ground-state only ---")
    trans2 = _make_transcription()
    t0 = time.perf_counter()
    sol_ground = solve_hybrid(
        trans2,
        config=HybridConfig(
            qubo=LinearLsqToQuboConfig(bits_per_variable=BITS),
            num_reads=NUM_READS,
            rerank_top_k=RERANK_TOP_K,
            rerank=False,
            max_iter=MAX_ITER,
            tol=1e-7,
            damping=DAMPING,
            seed=42,
        ),
    )
    t_ground = time.perf_counter() - t0
    print(
        f"  {sol_ground.n_iterations} iters, "
        f"final NL res = {sol_ground.history['nonlinear_residual'][-1]:.4e}, "
        f"{t_ground:.1f}s"
    )

    print("\n--- Running hybrid with reranking ---")
    trans3 = _make_transcription()
    t0 = time.perf_counter()
    sol_reranked = solve_hybrid(
        trans3,
        config=HybridConfig(
            qubo=LinearLsqToQuboConfig(bits_per_variable=BITS),
            num_reads=NUM_READS,
            rerank_top_k=RERANK_TOP_K,
            rerank=True,
            max_iter=MAX_ITER,
            tol=1e-7,
            damping=DAMPING,
            seed=42,
        ),
    )
    t_reranked = time.perf_counter() - t0
    print(
        f"  {sol_reranked.n_iterations} iters, "
        f"final NL res = {sol_reranked.history['nonlinear_residual'][-1]:.4e}, "
        f"{t_reranked:.1f}s"
    )

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5.5))

    iters_c = np.arange(1, sol_classical.n_iterations + 1)
    iters_g = np.arange(1, sol_ground.n_iterations + 1)
    iters_r = np.arange(1, sol_reranked.n_iterations + 1)

    ax.semilogy(
        iters_c,
        sol_classical.history["nonlinear_residual"],
        "o-",
        color="tab:blue",
        ms=5,
        lw=1.4,
        label=f"Classical lstsq + Armijo  ({t_classical:.1f}s)",
    )
    ax.semilogy(
        iters_g,
        sol_ground.history["nonlinear_residual"],
        "s--",
        color="tab:red",
        ms=5,
        lw=1.4,
        label=f"Hybrid QUBO ground-state  ({t_ground:.1f}s)",
    )
    ax.semilogy(
        iters_r,
        sol_reranked.history["nonlinear_residual"],
        "D-",
        color="tab:green",
        ms=5,
        lw=1.6,
        label=f"Hybrid QUBO + reranking  ({t_reranked:.1f}s)",
    )

    ax.set_xlabel("outer iteration $k$")
    ax.set_ylabel(r"$\|$nonlinear TPBVP residual$\|_\infty$")
    ax.set_title(
        f"Cold-start convergence  "
        f"({BITS} bits, {NUM_READS} SA reads, "
        f"top-{RERANK_TOP_K} rerank, damping={DAMPING})"
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nsaved: {out_path}")
    plt.close(fig)

    return sol_reranked


# ---------------------------------------------------------------------------
# Figure 2: Reranking diagnostics
# ---------------------------------------------------------------------------


def figure_rerank_stats(out_path: Path, sol: object) -> None:
    hist = sol.history
    iters = np.arange(1, sol.n_iterations + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    # Left: rank of reranking winner
    ax = axes[0]
    ranks = hist["best_qubo_rank"]
    colors = ["tab:green" if r > 1 else "tab:gray" for r in ranks]
    ax.bar(iters, ranks, color=colors, edgecolor="none", width=0.8)
    ax.set_xlabel("iteration")
    ax.set_ylabel("QUBO energy rank of NL-best sample")
    ax.set_title("Reranking winner rank")
    ax.axhline(1, color="black", ls=":", lw=0.8)
    n_improved = sum(1 for r in ranks if r > 1)
    ax.text(
        0.98, 0.95,
        f"excited state chosen\n{n_improved}/{len(ranks)} iterations",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
    )

    # Center: NL residual improvement
    ax = axes[1]
    ground = np.array(hist["ground_state_nl_res"])
    reranked = np.array(hist["reranked_nl_res"])
    improvement = (ground - reranked) / np.maximum(ground, 1e-15) * 100
    ax.bar(iters, improvement, color="tab:green", edgecolor="none", width=0.8)
    ax.set_xlabel("iteration")
    ax.set_ylabel("NL residual improvement (%)")
    ax.set_title("Reranking improvement over ground state")
    ax.axhline(0, color="black", ls=":", lw=0.8)

    # Right: unique samples in top-k
    ax = axes[2]
    ax.bar(
        iters, hist["n_unique_samples"],
        color="tab:purple", edgecolor="none", width=0.8,
    )
    ax.set_xlabel("iteration")
    ax.set_ylabel("unique samples in top-k")
    ax.set_title(f"Sample diversity (top-{RERANK_TOP_K})")

    fig.savefig(out_path, dpi=150)
    print(f"saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Scaling (Path A) -- wall-clock time vs problem size
# ---------------------------------------------------------------------------


def figure_scaling(out_path: Path) -> None:
    print("\n--- Path A: Scaling analysis ---")
    n_training_values = [6, 8, 10, 12, 16, 20]
    n_basis = 40
    num_reads = 200

    classical_times = []
    qubo_build_times = []
    sa_sample_times = []
    qubo_total_times = []
    n_qubits_list = []

    print(f"{'n_train':>8} {'n_eq':>6} {'n_qub':>7} "
          f"{'lstsq(ms)':>10} {'QUBO+SA(ms)':>12} {'ratio':>7}")
    print("-" * 55)

    for n_train in n_training_values:
        cfg = IndirectTfcElmConfig(n_training=n_train, n_basis=n_basis, seed=7)
        trans = IndirectTfcElmTranscription(
            dynamics=PlanarCR3BP(), config=cfg, **SYNTHETIC_BCS
        )
        nominal_x, nominal_y = trans.initial_nominal()
        A, B = trans.build_linear_system(nominal_x, nominal_y)

        # Time classical lstsq (average over 50 runs)
        times_c = []
        for _ in range(50):
            t0 = time.perf_counter()
            np.linalg.lstsq(A, B, rcond=None)
            times_c.append(time.perf_counter() - t0)
        t_lstsq = np.median(times_c)

        # Time QUBO build + SA sample
        t0 = time.perf_counter()
        qubo_cfg = LinearLsqToQuboConfig(bits_per_variable=BITS)
        qubo = build_linear_lsq_qubo(A, B, qubo_cfg)
        t_build = time.perf_counter() - t0

        t0 = time.perf_counter()
        sr = sample_simulated_annealing(
            qubo, num_reads=num_reads, seed=0,
            initial_states=qubo.warm_start_bits(),
        )
        t_sa = time.perf_counter() - t0

        classical_times.append(t_lstsq * 1000)
        qubo_build_times.append(t_build * 1000)
        sa_sample_times.append(t_sa * 1000)
        qubo_total_times.append((t_build + t_sa) * 1000)
        n_qubits_list.append(qubo.n_bits)

        ratio = (t_build + t_sa) / t_lstsq
        print(f"{n_train:8d} {6*n_train:6d} {qubo.n_bits:7d} "
              f"{t_lstsq*1000:10.2f} {(t_build+t_sa)*1000:12.1f} {ratio:7.0f}x")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    # Left: wall-clock time comparison
    ax = axes[0]
    ax.semilogy(
        n_training_values, classical_times,
        "o-", color="tab:blue", lw=1.6, label="Classical lstsq (median)",
    )
    ax.semilogy(
        n_training_values, qubo_total_times,
        "D-", color="tab:green", lw=1.6, label=f"QUBO build + SA ({num_reads} reads)",
    )
    ax.semilogy(
        n_training_values, qubo_build_times,
        "s--", color="tab:orange", lw=1.2, label="QUBO build only",
    )

    # QPU anneal time reference line (20 μs, constant)
    ax.axhline(
        0.02, color="tab:purple", ls=":", lw=1.5,
        label="D-Wave QPU anneal (~20 μs)",
    )

    ax.set_xlabel("n_training (collocation points)")
    ax.set_ylabel("wall-clock time per solve (ms)")
    ax.set_title("Scaling: classical lstsq vs QUBO pipeline")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper left")

    # Right: QUBO size vs n_training
    ax = axes[1]
    ax.plot(n_training_values, n_qubits_list, "D-", color="tab:green", lw=1.6)
    for nt, nq in zip(n_training_values, n_qubits_list):
        ax.annotate(str(nq), (nt, nq), textcoords="offset points",
                    xytext=(0, 10), fontsize=8, ha="center")
    ax.axhline(5000, color="tab:purple", ls=":", lw=1.5,
               label="Advantage2 logical limit (~5000)")
    ax.set_xlabel("n_training (collocation points)")
    ax.set_ylabel("QUBO logical qubits")
    ax.set_title(f"QUBO size ({BITS} bits/var)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    fig.savefig(out_path, dpi=150)
    print(f"\nsaved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Warm-started convergence (from direct-NLP reference)
# ---------------------------------------------------------------------------


def figure_warm_convergence(out_path: Path) -> None:
    """Convergence from the NLP reference initial nominal.

    The classical solver hits a nonlinear residual plateau around ~0.7
    because the Armijo line search damps toward a local minimum of the
    linearized residual. The hybrid reranker can potentially break through
    by choosing excited-state samples that are better in the nonlinear sense.
    """
    print("\n--- Solving direct-NLP reference ---")
    dyn = PlanarCR3BP()
    ref = solve_energy_optimal_cr3bp(
        dyn,
        config=DirectCollocationConfig(n_intervals=40, maxiter=500),
        **SYNTHETIC_BCS,
    )
    print(f"  NLP: success={ref.success}, max_defect={ref.max_defect:.2e}")

    # Build transcription and resample NLP solution onto TFC grid
    trans = _make_transcription()
    warm_x, warm_y = ref.sample_position(trans.t)

    print("\n--- Warm classical solver ---")
    trans_c = _make_transcription()
    t0 = time.perf_counter()
    sol_classical = trans_c.solve_sequential(
        OuterLoopConfig(
            max_iter=MAX_ITER, tol=1e-7, damping=1.0, line_search=True
        ),
        initial_nominal=(warm_x.copy(), warm_y.copy()),
    )
    t_classical = time.perf_counter() - t0
    print(
        f"  {sol_classical.n_iterations} iters, "
        f"final NL res = {sol_classical.history['nonlinear_residual'][-1]:.4e}, "
        f"{t_classical:.1f}s"
    )

    print("\n--- Warm hybrid ground-state only ---")
    trans_g = _make_transcription()
    t0 = time.perf_counter()
    sol_ground = solve_hybrid(
        trans_g,
        config=HybridConfig(
            qubo=LinearLsqToQuboConfig(bits_per_variable=BITS),
            num_reads=NUM_READS,
            rerank_top_k=RERANK_TOP_K,
            rerank=False,
            max_iter=MAX_ITER,
            tol=1e-7,
            damping=DAMPING,
            seed=42,
        ),
        initial_nominal=(warm_x.copy(), warm_y.copy()),
    )
    t_ground = time.perf_counter() - t0
    print(
        f"  {sol_ground.n_iterations} iters, "
        f"final NL res = {sol_ground.history['nonlinear_residual'][-1]:.4e}, "
        f"{t_ground:.1f}s"
    )

    print("\n--- Warm hybrid with reranking ---")
    trans_r = _make_transcription()
    t0 = time.perf_counter()
    sol_reranked = solve_hybrid(
        trans_r,
        config=HybridConfig(
            qubo=LinearLsqToQuboConfig(bits_per_variable=BITS),
            num_reads=NUM_READS,
            rerank_top_k=RERANK_TOP_K,
            rerank=True,
            max_iter=MAX_ITER,
            tol=1e-7,
            damping=DAMPING,
            seed=42,
        ),
        initial_nominal=(warm_x.copy(), warm_y.copy()),
    )
    t_reranked = time.perf_counter() - t0
    print(
        f"  {sol_reranked.n_iterations} iters, "
        f"final NL res = {sol_reranked.history['nonlinear_residual'][-1]:.4e}, "
        f"{t_reranked:.1f}s"
    )

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5.5))

    iters_c = np.arange(1, sol_classical.n_iterations + 1)
    iters_g = np.arange(1, sol_ground.n_iterations + 1)
    iters_r = np.arange(1, sol_reranked.n_iterations + 1)

    ax.semilogy(
        iters_c,
        sol_classical.history["nonlinear_residual"],
        "o-",
        color="tab:blue",
        ms=5,
        lw=1.4,
        label=f"Classical lstsq + Armijo  ({t_classical:.1f}s)",
    )
    ax.semilogy(
        iters_g,
        sol_ground.history["nonlinear_residual"],
        "s--",
        color="tab:red",
        ms=5,
        lw=1.4,
        label=f"Hybrid QUBO ground-state  ({t_ground:.1f}s)",
    )
    ax.semilogy(
        iters_r,
        sol_reranked.history["nonlinear_residual"],
        "D-",
        color="tab:green",
        ms=5,
        lw=1.6,
        label=f"Hybrid QUBO + reranking  ({t_reranked:.1f}s)",
    )

    ax.set_xlabel("outer iteration $k$")
    ax.set_ylabel(r"$\|$nonlinear TPBVP residual$\|_\infty$")
    ax.set_title(
        f"Warm-start convergence (NLP reference)  "
        f"({BITS} bits, damping={DAMPING})"
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nsaved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Hybrid comparison: classical vs QUBO + reranking")
    print("=" * 60)

    sol_reranked = figure_convergence(out_dir / "hybrid_convergence.png")
    figure_rerank_stats(out_dir / "hybrid_rerank_stats.png", sol_reranked)
    figure_scaling(out_dir / "hybrid_scaling.png")
    figure_warm_convergence(out_dir / "hybrid_warm_convergence.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
