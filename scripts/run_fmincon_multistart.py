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

Outputs ``scripts/figures/fmincon_multistart.png`` (panel a: per-case strip
plot of feasible converged objectives with the canonical seed marked;
panel b: seed-by-seed SLSQP-vs-fmincon pairing on the primary case) and
``scripts/figures/fmincon_multistart.csv``.

Run::

    python -m scripts.run_fmincon_multistart                # SLSQP only
    python -m scripts.run_fmincon_multistart --with-matlab  # + fmincon subset
    python -m scripts.run_fmincon_multistart --replot       # restyle from CSV
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
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
    })

    fig, (ax_strip, ax_pair) = plt.subplots(
        1, 2, figsize=(12.5, 5.0), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )

    blue = "#1f77b4"
    orange = "#ff7f0e"

    # ---- (a) per-case strip plot of feasible objectives ----
    #
    # One column per BC case. Every feasible converged objective is a dot;
    # the black diamond is the canonical linear-interpolation seed (seed 0).
    # The x tick label reports distinct minima / feasible starts, which is
    # the headline nonconvexity measure.
    names = list(slsqp_by_case.keys())
    rng_jitter = np.random.default_rng(0)
    for j, name in enumerate(names):
        results = slsqp_by_case[name]
        feas = [(r.seed_index, r.objective) for r in results if r.feasible]
        if not feas:
            continue
        objs = np.array([o for _, o in feas])
        # Min-to-max range bar behind the dots.
        ax_strip.plot([j, j], [objs.min(), objs.max()],
                      color="0.85", lw=5, solid_capstyle="round", zorder=1)
        x = j + rng_jitter.uniform(-0.07, 0.07, size=len(objs))
        ax_strip.scatter(x, objs, s=30, alpha=0.85, color=blue,
                         edgecolors="white", linewidths=0.5, zorder=3)
        # Canonical seed (index 0) drawn on top as an open diamond.
        for seed_idx, obj in feas:
            if seed_idx == 0:
                ax_strip.scatter([j], [obj], s=90, marker="D",
                                 facecolors="none", edgecolors="black",
                                 linewidths=1.4, zorder=4)
        # Best objective annotated under the column.
        ax_strip.annotate(f"best {objs.min():.2f}",
                          xy=(j, objs.min()), xytext=(0, -14),
                          textcoords="offset points",
                          ha="center", fontsize=8.5, color="0.25")

    counts = [
        f"{summaries[n]['n_minima']}/{summaries[n]['n_feasible']}"
        for n in names
    ]
    ax_strip.set_xticks(range(len(names)))
    ax_strip.set_xticklabels(
        [f"{n}\n{c} distinct minima" for n, c in zip(names, counts)],
        fontsize=8.5,
    )
    ax_strip.set_xlim(-0.5, len(names) - 0.5)
    ax_strip.set_ylim(bottom=0.0)
    ax_strip.set_ylabel("Converged objective $J$ (feasible runs)")
    ax_strip.set_title("(a) 32 random starts per case (SLSQP)")
    ax_strip.grid(axis="y", alpha=0.3)
    ax_strip.legend(handles=[
        Line2D([], [], marker="o", ls="none", color=blue,
               markeredgecolor="white", label="feasible local minimum"),
        Line2D([], [], marker="D", ls="none", markerfacecolor="none",
               markeredgecolor="black", label="canonical linear-interp seed"),
    ], loc="upper right")

    # ---- (b) same seeds, two optimisers (primary case) ----
    #
    # Seed-by-seed pairing of SLSQP and fmincon started from identical
    # iterates: vertical separation = different basins; coincident markers
    # = shared basin. Open markers are runs that ended infeasible.
    if matlab_primary is not None:
        slsqp_primary = {r.seed_index: r for r in slsqp_by_case[PRIMARY_CASE]}
        for ml in matlab_primary:
            sp = slsqp_primary.get(ml.seed_index)
            if sp is None:
                continue
            i = ml.seed_index
            ax_pair.plot([i, i], [sp.objective, ml.objective],
                         color="0.75", lw=1.2, zorder=1)
            ax_pair.scatter([i], [sp.objective], s=55, marker="o",
                            facecolors=blue if sp.feasible else "none",
                            edgecolors=blue, linewidths=1.4, zorder=3)
            ax_pair.scatter([i], [ml.objective], s=65, marker="v",
                            facecolors=orange if ml.feasible else "none",
                            edgecolors=orange, linewidths=1.4, zorder=3)
            gap = abs(sp.objective - ml.objective) / max(abs(sp.objective), 1e-30)
            if gap < CLUSTER_REL_TOL and sp.feasible and ml.feasible:
                ax_pair.annotate(
                    "same basin:\n$|\\Delta J| \\approx 10^{-9}$",
                    xy=(i, sp.objective), xytext=(i + 0.35, sp.objective + 4.0),
                    fontsize=8.5, ha="left",
                    arrowprops=dict(arrowstyle="->", color="0.3", lw=0.9),
                )
            elif i == 0:
                # Near-coincident at this scale but genuinely distinct
                # minima -- annotate so the pair is not misread as shared.
                ax_pair.annotate(
                    f"close but distinct:\n$\\Delta J = {gap * 100:.1f}\\%$",
                    xy=(i, max(sp.objective, ml.objective)),
                    xytext=(i - 0.45, sp.objective + 5.5),
                    fontsize=8.5, ha="left",
                    arrowprops=dict(arrowstyle="->", color="0.3", lw=0.9),
                )
        n_seeds = len(matlab_primary)
        ax_pair.set_xticks(range(n_seeds))
        ax_pair.set_xlim(-0.6, n_seeds - 0.4 + 0.8)
        ax_pair.set_xlabel("shared seed index")
        ax_pair.set_ylabel("Converged objective $J$")
        ax_pair.set_title(f"(b) Case {PRIMARY_CASE.split()[0]}: "
                          "same seed, two optimisers")
        ax_pair.grid(axis="y", alpha=0.3)
        ax_pair.legend(handles=[
            Line2D([], [], marker="o", ls="none", color=blue,
                   label="SLSQP (scipy)"),
            Line2D([], [], marker="v", ls="none", color=orange,
                   label="fmincon (MATLAB)"),
            Line2D([], [], marker="o", ls="none", markerfacecolor="none",
                   markeredgecolor="0.4", label="run ended infeasible"),
        ], loc="upper left")
    else:
        ax_pair.text(0.5, 0.5, "fmincon cross-check not run\n"
                     "(--with-matlab)", transform=ax_pair.transAxes,
                     ha="center", va="center", fontsize=10, color="0.4")
        ax_pair.set_axis_off()

    fig.suptitle(
        "The Hermite–Simpson energy-optimal CR3BP transcription "
        "is nonconvex",
        fontsize=12.5,
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\n  figure written to {out_path}")


def _load_results_csv(
    csv_path: Path,
) -> tuple[dict[str, list[StartResult]], list[StartResult] | None]:
    """Rebuild the solver results from a previously written CSV.

    Allows ``--replot`` to restyle the figure without re-running the
    SLSQP multistart (minutes) or the MATLAB fmincon subset.
    """
    slsqp_by_case: dict[str, list[StartResult]] = {}
    matlab_primary: list[StartResult] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            r = StartResult(
                seed_index=int(row["seed_index"]),
                objective=float(row["objective"]),
                feasible=row["feasible"] == "True",
                success=row["success"] == "True",
                max_defect=float(row["max_defect"]),
                max_bc_error=float(row["max_bc_error"]),
                n_iterations=int(row["n_iterations"]),
            )
            if row["solver"] == "SLSQP":
                slsqp_by_case.setdefault(row["case"], []).append(r)
            else:
                matlab_primary.append(r)
    return slsqp_by_case, (matlab_primary or None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-matlab", action="store_true",
                    help="also run a subset of seeds through MATLAB fmincon")
    ap.add_argument("--n-starts", type=int, default=N_STARTS)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--replot", action="store_true",
                    help="regenerate the figure from the saved CSV "
                         "without re-running any solver")
    args = ap.parse_args()

    fig_dir = Path(__file__).resolve().parent / "figures"
    if args.replot:
        csv_path = fig_dir / "fmincon_multistart.csv"
        if not csv_path.exists():
            sys.exit(f"--replot: {csv_path} not found; run the sweep first.")
        slsqp_by_case, matlab_primary = _load_results_csv(csv_path)
        summaries = {
            name: _summarize(name, results)
            for name, results in slsqp_by_case.items()
        }
        _plot(summaries, slsqp_by_case, matlab_primary,
              fig_dir / "fmincon_multistart.png")
        return

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
