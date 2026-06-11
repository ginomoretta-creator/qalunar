"""Multistart characterization of the transcribed CR3BP NLP (SLSQP + fmincon).

The single-seed cross-check in ``run_fmincon_comparison.py`` already showed that
scipy SLSQP and MATLAB fmincon, given the *same* linear-interpolation seed,
converge to *different* feasible local minima of the Hermite-Simpson
energy-optimal CR3BP transcription -- i.e. the NLP is nonconvex and the
"classical continuous optimum" is solver/seed-dependent.

This script quantifies *how often* and *how widely* that happens. For each
boundary-condition case it draws ``n_starts`` random seeds (Gaussian
perturbations of the linear-interp state/control), solves each from that seed,
and reports the distribution of feasible converged objectives. The number of
distinct objective clusters is a direct, reproducible measure of nonconvexity.

By default only Python SLSQP is run (fast, hundreds of solves). With
``--with-matlab`` a subset of the *same* seeds is also handed to fmincon, so the
two optimizers are compared on identical starting points -- confirming the
multistart spread is a property of the problem, not of one optimizer.

Outputs ``scripts/fmincon_multistart.png`` (objective spread per case + the
primary-case histogram with the fmincon overlay) and ``fmincon_multistart.csv``.

Run::

    python -m scripts.run_fmincon_multistart                # SLSQP only
    python -m scripts.run_fmincon_multistart --with-matlab  # + fmincon subset
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qalunar.dynamics import PlanarCR3BP
from qalunar.dynamics.cr3bp import EARTH_MOON_MU
from qalunar.reference.direct_collocation import (
    DirectCollocationConfig,
    solve_energy_optimal_cr3bp,
)


# Boundary-condition cases (all mild, away from the primaries, so a clean
# feasible manifold exists; the first mirrors run_fmincon_comparison.py).
CASES: dict[str, dict] = {
    "A (baseline)": dict(
        r0=np.array([-0.3, 0.0]), v0=np.array([0.0, 0.6]),
        rf=np.array([0.4, 0.2]), vf=np.array([-0.1, 0.0]),
        time_of_flight=2.5,
    ),
    "B (longer T)": dict(
        r0=np.array([-0.4, 0.1]), v0=np.array([0.0, 0.5]),
        rf=np.array([0.5, -0.1]), vf=np.array([0.0, 0.3]),
        time_of_flight=3.5,
    ),
    "C (wide swing)": dict(
        r0=np.array([-0.2, -0.3]), v0=np.array([0.3, 0.4]),
        rf=np.array([0.6, 0.3]), vf=np.array([-0.2, 0.1]),
        time_of_flight=3.0,
    ),
}
PRIMARY_CASE = "A (baseline)"

N_INTERVALS = 40
MAXITER = 300
TOL = 1e-8
N_STARTS = 32
SIGMA_STATE = 0.15        # Gaussian std on state-node perturbation
SIGMA_CONTROL = 0.20      # Gaussian std on control-node perturbation
FEASIBLE_TOL = 1e-6       # max(defect, bc) below this == feasible
CLUSTER_REL_TOL = 1e-3    # objectives within this rel. gap == same minimum
N_MATLAB_SUBSET = 6       # seeds also handed to fmincon under --with-matlab

REPO = Path(__file__).resolve().parents[1]
MATLAB_DIR = REPO / "matlab"
IO_DIR = MATLAB_DIR / "_io"


@dataclass
class StartResult:
    seed_index: int
    objective: float
    feasible: bool
    success: bool
    max_defect: float
    max_bc_error: float
    n_iterations: int


def _linear_seed(case: dict, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Linear-interpolation state seed and zero control (the default seed)."""
    s0 = np.concatenate([case["r0"], case["v0"]])
    sf = np.concatenate([case["rf"], case["vf"]])
    alphas = np.linspace(0.0, 1.0, n_nodes)
    state = (1.0 - alphas)[:, None] * s0 + alphas[:, None] * sf
    control = np.zeros((n_nodes, 2))
    return state, control


def _make_seeds(
    case: dict, n_nodes: int, n_starts: int, rng: np.random.Generator,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Linear seed (index 0) + Gaussian-perturbed variants.

    The endpoints of the state seed are left unperturbed (they sit on the
    BCs anyway), so every seed is an equally plausible trajectory guess.
    """
    state0, control0 = _linear_seed(case, n_nodes)
    seeds = [(state0.copy(), control0.copy())]
    for _ in range(n_starts - 1):
        ds = rng.normal(0.0, SIGMA_STATE, size=state0.shape)
        ds[0] = 0.0
        ds[-1] = 0.0
        du = rng.normal(0.0, SIGMA_CONTROL, size=control0.shape)
        seeds.append((state0 + ds, control0 + du))
    return seeds


def _pack_c(state: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Pack in C order, matching direct_collocation._pack (for SLSQP)."""
    return np.concatenate([state.ravel(), control.ravel()])


def _pack_fortran(state: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Pack in Fortran order, matching MATLAB's [S(:); U(:)] / reshape.

    MATLAB fills ``reshape(z, nn, 4)`` column-major, so the seed it expects
    is ``[state[:,0]; state[:,1]; ...; control[:,0]; control[:,1]]``.
    """
    return np.concatenate([state.flatten("F"), control.flatten("F")])


def _solve_slsqp(
    dyn: PlanarCR3BP, case: dict, cfg: DirectCollocationConfig,
    seeds: list[tuple[np.ndarray, np.ndarray]],
) -> list[StartResult]:
    out: list[StartResult] = []
    for i, (state, control) in enumerate(seeds):
        res = solve_energy_optimal_cr3bp(
            dyn, case["r0"], case["v0"], case["rf"], case["vf"],
            case["time_of_flight"], config=cfg,
            initial_guess=_pack_c(state, control),
        )
        feasible = res.max_defect < FEASIBLE_TOL and res.max_bc_error < FEASIBLE_TOL
        out.append(StartResult(
            seed_index=i, objective=res.objective, feasible=feasible,
            success=res.success, max_defect=res.max_defect,
            max_bc_error=res.max_bc_error, n_iterations=res.n_iterations,
        ))
    return out


def _solve_fmincon(
    case: dict, seeds: list[tuple[np.ndarray, np.ndarray]],
) -> list[StartResult]:
    matlab = shutil.which("matlab")
    if matlab is None:
        sys.exit("MATLAB not found on PATH; cannot run --with-matlab.")
    IO_DIR.mkdir(parents=True, exist_ok=True)
    params_path = IO_DIR / "ms_params.json"
    results_path = IO_DIR / "ms_results.json"
    out: list[StartResult] = []
    for i, (state, control) in enumerate(seeds):
        params = dict(
            mu=EARTH_MOON_MU,
            r0=case["r0"].tolist(), v0=case["v0"].tolist(),
            rf=case["rf"].tolist(), vf=case["vf"].tolist(),
            T=case["time_of_flight"],
            n_intervals=N_INTERVALS, maxiter=MAXITER, tol=TOL,
            control_bound=None, algorithm="sqp",
            z0=_pack_fortran(state, control).tolist(),
        )
        params_path.write_text(json.dumps(params), encoding="utf-8")
        cmd = (f"cr3bp_fmincon('{params_path.as_posix()}', "
               f"'{results_path.as_posix()}')")
        print(f"    fmincon seed {i} ...", flush=True)
        proc = subprocess.run(
            [matlab, "-batch", cmd], cwd=str(MATLAB_DIR),
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print("    [matlab stderr]", proc.stderr.strip())
            sys.exit(f"MATLAB exited with code {proc.returncode}")
        ml = json.loads(results_path.read_text(encoding="utf-8"))
        feasible = ml["max_defect"] < FEASIBLE_TOL and ml["max_bc_error"] < FEASIBLE_TOL
        out.append(StartResult(
            seed_index=i, objective=float(ml["objective"]), feasible=feasible,
            success=bool(ml["success"]), max_defect=float(ml["max_defect"]),
            max_bc_error=float(ml["max_bc_error"]),
            n_iterations=int(ml["n_iterations"]),
        ))
    return out


def _cluster(objectives: list[float], rel_tol: float) -> list[float]:
    """Greedy 1-D clustering: return sorted cluster representatives (minima)."""
    if not objectives:
        return []
    vals = sorted(objectives)
    clusters = [[vals[0]]]
    for v in vals[1:]:
        ref = clusters[-1][0]
        if abs(v - ref) / max(abs(ref), 1e-30) <= rel_tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [c[0] for c in clusters]


def _summarize(name: str, results: list[StartResult]) -> dict:
    feas = [r.objective for r in results if r.feasible]
    clusters = _cluster(feas, CLUSTER_REL_TOL)
    best = min(feas) if feas else float("nan")
    worst = max(feas) if feas else float("nan")
    spread = (worst - best) / abs(best) if feas and best != 0 else float("nan")
    return dict(
        name=name, n_starts=len(results), n_feasible=len(feas),
        n_minima=len(clusters), best=best, worst=worst, spread=spread,
        minima=clusters,
    )


def _plot(
    summaries: dict[str, dict],
    slsqp_by_case: dict[str, list[StartResult]],
    matlab_primary: list[StartResult] | None,
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "legend.fontsize": 8, "xtick.labelsize": 9, "ytick.labelsize": 9,
    })

    fig, (ax_strip, ax_hist) = plt.subplots(
        1, 2, figsize=(13, 5.2), constrained_layout=True,
    )

    # ---- (a) per-case strip plot of feasible objectives ----
    names = list(slsqp_by_case.keys())
    for j, name in enumerate(names):
        feas = [r.objective for r in slsqp_by_case[name] if r.feasible]
        x = np.full(len(feas), j) + np.random.default_rng(0).normal(
            0, 0.04, size=len(feas))
        ax_strip.scatter(x, feas, s=28, alpha=0.7, color="#1f77b4",
                         edgecolors="none", zorder=3)
        for m in summaries[name]["minima"]:
            ax_strip.hlines(m, j - 0.25, j + 0.25, color="#d62728",
                            lw=1.6, zorder=4)
        ax_strip.text(j, max(feas) if feas else 0,
                      f"  {summaries[name]['n_minima']} minima",
                      fontsize=8, rotation=90, va="bottom", ha="center")
    ax_strip.set_xticks(range(len(names)))
    ax_strip.set_xticklabels(names, fontsize=8)
    ax_strip.set_ylabel("Feasible converged objective J")
    ax_strip.set_title("(a) Multistart objective spread (SLSQP)\n"
                       "red bars = distinct local minima")
    ax_strip.grid(axis="y", alpha=0.3)

    # ---- (b) primary-case histogram with fmincon overlay ----
    primary = slsqp_by_case[PRIMARY_CASE]
    feas = [r.objective for r in primary if r.feasible]
    ax_hist.hist(feas, bins=18, color="#1f77b4", alpha=0.75,
                 label=f"SLSQP feasible (n={len(feas)})")
    for m in summaries[PRIMARY_CASE]["minima"]:
        ax_hist.axvline(m, color="#d62728", lw=1.4, ls="--")
    if matlab_primary is not None:
        ml_feas = [r.objective for r in matlab_primary if r.feasible]
        for k, v in enumerate(ml_feas):
            ax_hist.scatter(v, 0.5, marker="v", s=90, color="#ff7f0e",
                            edgecolors="black", zorder=5,
                            label="fmincon (same seeds)" if k == 0 else None)
    ax_hist.set_xlabel("Feasible converged objective J")
    ax_hist.set_ylabel("count")
    ax_hist.set_title(f"(b) Primary case {PRIMARY_CASE}: "
                      "local-minima distribution")
    ax_hist.legend(loc="upper right")
    ax_hist.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "The Hermite-Simpson CR3BP NLP is nonconvex: multistart finds "
        "multiple feasible local minima",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"\n  figure written to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-matlab", action="store_true",
                    help="also run a subset of seeds through MATLAB fmincon")
    ap.add_argument("--n-starts", type=int, default=N_STARTS)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    dyn = PlanarCR3BP(mu=EARTH_MOON_MU)
    cfg = DirectCollocationConfig(n_intervals=N_INTERVALS, maxiter=MAXITER, tol=TOL)
    rng = np.random.default_rng(args.seed)
    n_nodes = N_INTERVALS + 1

    print("=" * 78)
    print("  Multistart nonconvexity characterization of the CR3BP NLP")
    print("=" * 78)
    print(f"  transcription: Hermite-Simpson, {N_INTERVALS} intervals; "
          f"{args.n_starts} seeds/case\n")

    slsqp_by_case: dict[str, list[StartResult]] = {}
    seeds_by_case: dict[str, list] = {}
    summaries: dict[str, dict] = {}
    for name, case in CASES.items():
        seeds = _make_seeds(case, n_nodes, args.n_starts, rng)
        seeds_by_case[name] = seeds
        print(f"  [{name}] solving {len(seeds)} SLSQP starts ...", flush=True)
        results = _solve_slsqp(dyn, case, cfg, seeds)
        slsqp_by_case[name] = results
        summaries[name] = _summarize(name, results)

    # ---- fmincon cross-check on the primary case (same seeds) ----
    matlab_primary: list[StartResult] | None = None
    if args.with_matlab:
        subset = seeds_by_case[PRIMARY_CASE][:N_MATLAB_SUBSET]
        print(f"\n  fmincon cross-check on {len(subset)} shared seeds "
              f"({PRIMARY_CASE}) ...", flush=True)
        matlab_primary = _solve_fmincon(CASES[PRIMARY_CASE], subset)

    # ---- report ----
    print()
    print("-" * 78)
    print(f"  {'case':<16}{'starts':>8}{'feasible':>10}{'#minima':>9}"
          f"{'J_best':>12}{'J_worst':>12}{'spread':>9}")
    print("-" * 78)
    for name in CASES:
        s = summaries[name]
        print(f"  {name:<16}{s['n_starts']:>8}{s['n_feasible']:>10}"
              f"{s['n_minima']:>9}{s['best']:>12.6f}{s['worst']:>12.6f}"
              f"{s['spread']:>8.1%}")
    print("-" * 78)

    if matlab_primary is not None:
        slsqp_primary = slsqp_by_case[PRIMARY_CASE]
        print(f"\n  Seed-by-seed (primary case {PRIMARY_CASE}):")
        print(f"  {'seed':>5}{'SLSQP J':>14}{'fmincon J':>14}"
              f"{'rel gap':>10}{'agree?':>9}")
        for i in range(len(matlab_primary)):
            sp = slsqp_primary[i]
            ml = matlab_primary[i]
            gap = abs(sp.objective - ml.objective) / max(abs(sp.objective), 1e-30)
            agree = "yes" if gap < CLUSTER_REL_TOL else "NO"
            print(f"  {i:>5}{sp.objective:>14.6f}{ml.objective:>14.6f}"
                  f"{gap:>10.1e}{agree:>9}")

    # ---- CSV ----
    csv_path = Path(__file__).resolve().parent / "figures" / "fmincon_multistart.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "solver", "seed_index", "objective", "feasible",
                    "success", "max_defect", "max_bc_error", "n_iterations"])
        for name in CASES:
            for r in slsqp_by_case[name]:
                w.writerow([name, "SLSQP", r.seed_index, f"{r.objective:.8f}",
                            r.feasible, r.success, f"{r.max_defect:.3e}",
                            f"{r.max_bc_error:.3e}", r.n_iterations])
        if matlab_primary is not None:
            for r in matlab_primary:
                w.writerow([PRIMARY_CASE, "fmincon", r.seed_index,
                            f"{r.objective:.8f}", r.feasible, r.success,
                            f"{r.max_defect:.3e}", f"{r.max_bc_error:.3e}",
                            r.n_iterations])

    _plot(summaries, slsqp_by_case, matlab_primary,
          Path(__file__).resolve().parent / "figures" / "fmincon_multistart.png")
    print(f"  table written to {csv_path}")


if __name__ == "__main__":
    main()
