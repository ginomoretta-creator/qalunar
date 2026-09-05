"""Solver benchmark on the cislunar correction QUBO with a time-to-solution
methodology.

Solvers: brute force (exact, N <= 20), MILP/HiGHS via a McCormick lift
(constraint-native reference), simulated annealing (dwave-neal), ballistic
simulated bifurcation (quantum-inspired tier, NumPy), and Kerberos
(dwave-hybrid; a *local classical* tabu + SA workflow, no QPU).

Why time-to-solution
--------------------
Wall time of one run says little about a stochastic heuristic: a fast run that
finds the optimum one time in ten is slower, in the sense that matters, than a
slow run that always finds it. Following Ronnow et al. (Science 345, 420,
2014), each stochastic solver is run ``R`` times with independent seeds and a
fixed per-run budget; ``p`` is the fraction of runs reaching the reference
energy, and

    TTS_99 = t_run * ln(1 - 0.99) / ln(1 - p)          (p in (0, 1))

is the expected time to hit the target with 99 % confidence (``t_run`` when
``p = 1``, infinite when ``p = 0``). "Hitting" means an energy within
``REL_TOL`` (1 %) of the reference -- a time-to-target, the form used when
no exact optimum is available (King et al., 2015). The reference is the
brute-force optimum for N <= 20 and the best energy found by any solver
otherwise (labelled "best-known", not "optimum").

The sweep reports every N and every solver; nothing is dropped. The
coefficient dynamic range of each instance (``max|c| / min|c|``) is recorded
because an analog annealer resolves coefficients to ~1-2 %.

Outputs ``scripts/figures/scaling_results.json`` (legacy single-shot entries
kept) and ``scaling_tts.csv``.

Run:  python -m scripts.run_scaling_sweep
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from qalunar.dynamics import PlanarCR3BP
from qalunar.dynamics.cr3bp import EARTH_MOON_MU
from qalunar.qubo.milp_baseline import solve_scheduling_milp
from qalunar.qubo.scheduling_samplers import (
    sample_brute_force,
    sample_kerberos,
    sample_simulated_annealing,
    sample_simulated_bifurcation,
    sample_simulated_bifurcation_torch,
)
from qalunar.qubo.thrust_scheduling import (
    ThrustSchedulingConfig,
    build_thrust_scheduling_qubo,
)
from qalunar.reference.edelbaum import LENGTH_KM


_MU = EARTH_MOON_MU
_GM = 1.0 - _MU
_R_HEO = 200_000.0 / LENGTH_KM
_V_CIRC = float(np.sqrt(_GM / _R_HEO))
_R_SYN = np.array([-_MU + _R_HEO, 0.0])
_OMEGA_CROSS_R = np.array([-_R_SYN[1], _R_SYN[0]])
_V_SYN_PERFECT = np.array([0.0, _V_CIRC * 1.190]) - _OMEGA_CROSS_R
_V_SYN_IMPERFECT = np.array([0.0, _V_CIRC * 1.187]) - _OMEGA_CROSS_R
STATE0 = np.concatenate([_R_SYN, _V_SYN_IMPERFECT])
STATE_PERFECT = np.concatenate([_R_SYN, _V_SYN_PERFECT])

THRUST_MAG = 0.02
T_SPAN = (0.0, 3.0)

NS = [10, 15, 20, 50, 100, 200]
REPEATS = 20                     # independent seeds per stochastic solver
SA_READS_PER_RUN = 100
SB_READS_PER_RUN = 100
KERBEROS_ITERS = 3
CONFIDENCE = 0.99
REL_TOL = 1e-2                   # time-to-target: within 1 % of the reference energy

FIG_DIR = Path(__file__).resolve().parent / "figures"


def _sb_backend():
    """Prefer the published simulated-bifurcation package (PyTorch); fall back
    to the in-house NumPy dSB, which is known to collapse on N >= 50."""
    try:
        import simulated_bifurcation  # noqa: F401
        return "sb_torch", lambda q, reads, seed: sample_simulated_bifurcation_torch(
            q, num_reads=reads, seed=seed)
    except ImportError:
        return "sb_numpy", lambda q, reads, seed: sample_simulated_bifurcation(
            q, num_reads=reads, seed=seed)


SB_NAME, SB_FN = _sb_backend()


def build_qubo(N: int):
    cfg = ThrustSchedulingConfig(
        thrust_magnitude=THRUST_MAG,
        thrust_direction="tangential",
        fuel_weight=0.0,
    )
    dyn = PlanarCR3BP()
    _, traj = dyn.propagate(STATE_PERFECT, T_SPAN, n_steps=8000)
    target = traj[-1]
    return build_thrust_scheduling_qubo(
        dyn, STATE0, target, T_SPAN,
        n_decision_steps=N, config=cfg, n_integration_substeps=80,
    )


def tts(times: np.ndarray, hits: np.ndarray) -> float:
    p = float(hits.mean())
    t_run = float(times.mean())
    if p >= 1.0:
        return t_run
    if p <= 0.0:
        return float("inf")
    return t_run * np.log(1.0 - CONFIDENCE) / np.log(1.0 - p)


def main() -> None:
    rows, tts_rows = [], []
    print(f"  SB backend: {SB_NAME}")
    print(f"{'N':>4} {'solver':>10} {'best E':>12} {'p':>6} {'t_run s':>9} {'TTS99 s':>10}  note")
    print("-" * 80)

    for N in NS:
        qubo = build_qubo(N)
        crange = qubo.coefficient_range()
        row: dict = {"N": N, "coefficient_range": crange, "results": {}, "tts": {}}
        runs: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        # ---- exact / deterministic references ------------------------------
        ref = None
        if N <= 20:
            bf = sample_brute_force(qubo)
            row["results"]["brute_force"] = {"energy": bf.energy, "time": bf.solve_time, "note": "exact"}
            ref = bf.energy
            print(f"{N:>4} {'BF':>10} {bf.energy:12.4e} {'1':>6} {bf.solve_time:9.3f} {bf.solve_time:10.3f}  exact")

        time_limit = 30.0 if N <= 20 else 60.0
        try:
            t0 = time.perf_counter()
            milp = solve_scheduling_milp(qubo, time_limit=time_limit, mip_rel_gap=1e-6)
            t_milp = time.perf_counter() - t0
            row["results"]["milp"] = {"energy": milp.energy, "time": t_milp,
                                      "is_optimal": milp.is_optimal, "n_aux": milp.n_aux_vars}
            print(f"{N:>4} {'MILP':>10} {milp.energy:12.4e} {'-':>6} {t_milp:9.3f} {'-':>10}  "
                  f"{'optimal' if milp.is_optimal else f'time limit {time_limit:.0f}s'}")
            milp_e = milp.energy
        except Exception as exc:
            row["results"]["milp"] = {"error": str(exc)}
            milp_e = np.inf
            print(f"{N:>4} {'MILP':>10} {'-':>12}  failed: {str(exc)[:50]}")

        # ---- legacy single-shot entries (continuity with the earlier table) ---
        sa1 = sample_simulated_annealing(qubo, num_reads=1000, seed=42)
        row["results"]["sa"] = {"energy": sa1.energy, "time": sa1.solve_time, "num_reads": 1000}
        try:
            k1 = sample_kerberos(qubo, max_iter=6, convergence=2, seed=42)
            row["results"]["kerberos"] = {"energy": k1.energy, "time": k1.solve_time,
                                          "backend": "local classical tabu+SA"}
        except Exception as exc:
            row["results"]["kerberos"] = {"error": str(exc)}
        sb1 = SB_FN(qubo, 1000, 42)
        row["results"]["sb"] = {"energy": sb1.energy, "time": sb1.solve_time, "num_reads": 1000,
                                "backend": SB_NAME}

        # ---- repeated runs for TTS -----------------------------------------
        for name, fn in (
            ("sa", lambda s: sample_simulated_annealing(qubo, num_reads=SA_READS_PER_RUN, seed=s)),
            ("sb", lambda s: SB_FN(qubo, SB_READS_PER_RUN, s)),
            ("kerberos", lambda s: sample_kerberos(qubo, max_iter=KERBEROS_ITERS, convergence=2, seed=s)),
        ):
            ts, es = [], []
            for r in range(REPEATS):
                try:
                    res = fn(1000 + r)
                except Exception as exc:
                    print(f"{N:>4} {name:>10}  run {r} failed: {str(exc)[:60]}")
                    continue
                ts.append(res.solve_time)
                es.append(res.energy)
            if ts:
                runs[name] = (np.asarray(ts), np.asarray(es))

        # reference: exact optimum, else best-known across everything measured
        candidates = [milp_e, sa1.energy, sb1.energy, row["results"].get("kerberos", {}).get("energy", np.inf)]
        candidates += [float(es.min()) for _, es in runs.values()]
        best_known = float(np.min(candidates))
        if ref is None:
            ref = best_known
            ref_note = "best-known"
        else:
            ref_note = "exact"
        row["reference"] = {"energy": ref, "kind": ref_note}

        for name, (ts, es) in runs.items():
            hits = es <= ref * (1 + REL_TOL) + 1e-15
            val = tts(ts, hits)
            row["tts"][name] = {"p": float(hits.mean()), "t_run": float(ts.mean()),
                                "tts99": val, "best": float(es.min()), "repeats": int(len(ts)),
                                "budget": {"sa": SA_READS_PER_RUN, "sb": SB_READS_PER_RUN,
                                           "kerberos": KERBEROS_ITERS}[name]}
            tts_rows.append({"N": N, "solver": name, "reference": ref_note, "p": float(hits.mean()),
                             "t_run_s": float(ts.mean()), "tts99_s": val,
                             "best_energy": float(es.min()), "coefficient_range": crange})
            print(f"{N:>4} {name:>10} {es.min():12.4e} {hits.mean():6.2f} {ts.mean():9.3f} "
                  f"{val:10.3f}  {ref_note} ref, {len(ts)} runs", flush=True)
        rows.append(row)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (FIG_DIR / "scaling_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (FIG_DIR / "scaling_tts.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(tts_rows[0].keys()))
        w.writeheader()
        w.writerows(tts_rows)
    print("\n  written: scripts/figures/scaling_results.json, scaling_tts.csv")


if __name__ == "__main__":
    main()
