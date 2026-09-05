"""Offline minor-embedding study of the dense scheduling QUBO on D-Wave
topologies (no QPU access needed).

The naturally-binary scheduling QUBO ``Q = B^T W B`` is dense (every pair of
slots couples through the shared final-state miss), so its logical graph is
the complete graph ``K_N``. Whether an instance fits a quantum annealer is a
property of that graph and of the hardware topology alone, and can be settled
on a laptop with ``minorminer``:

* **Clique embedding** (``minorminer.busclique``): the structured, provably
  minimal-chain embedding of ``K_N`` into Pegasus / Zephyr. This is what a
  dense QUBO actually uses on hardware.
* **Heuristic embedding** (``minorminer.find_embedding``): the generic
  path-search embedder, reported for comparison (longer chains, may fail).

Targets are the *ideal* graphs -- Pegasus P16 (Advantage, 5,640 qubits) and
Zephyr Z12 (Advantage2, 4,800 qubits). Real devices have a few percent of
inoperable qubits, which shrinks the largest embeddable clique slightly; the
ideal-graph numbers are upper bounds and are labelled as such.

Outputs ``scripts/figures/embedding_study.csv`` and ``.json``.

Run:  python -m scripts.run_embedding_study
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import dwave_networkx as dnx
import minorminer
import networkx as nx
import numpy as np
from minorminer import busclique

from scripts.run_scaling_sweep import build_qubo

FIG_DIR = Path(__file__).resolve().parent / "figures"
NS = [10, 15, 20, 50, 100, 150, 200]
HEURISTIC_TIMEOUT_S = 120


def _chain_stats(emb: dict) -> dict:
    lens = np.array([len(c) for c in emb.values()])
    return {"embedded": True, "max_chain": int(lens.max()),
            "mean_chain": float(lens.mean()), "physical_qubits": int(lens.sum())}


def main() -> None:
    targets = {
        "Pegasus P16 (Advantage)": dnx.pegasus_graph(16),
        "Zephyr Z12 (Advantage2)": dnx.zephyr_graph(12),
    }
    rows = []
    print(f"{'N':>4} {'density':>8} {'coef range':>11}  {'topology':<24} {'method':<10} "
          f"{'emb':>4} {'maxchain':>9} {'phys qubits':>12} {'time s':>8}")
    print("-" * 100)
    caches = {}
    for name, G in targets.items():
        t0 = time.perf_counter()
        caches[name] = busclique.busgraph_cache(G)
        print(f"  [{name}] clique cache built in {time.perf_counter()-t0:.1f} s "
              f"(largest clique {len(caches[name].largest_clique())})", flush=True)

    for N in NS:
        qubo = build_qubo(N)
        Q = qubo.Q
        off = np.abs(Q[np.triu_indices(N, 1)])
        density = float(np.mean(off > 0))
        crange = qubo.coefficient_range()
        K = nx.complete_graph(N)
        for name, G in targets.items():
            # clique embedding
            t0 = time.perf_counter()
            emb = caches[name].find_clique_embedding(N)
            t_clique = time.perf_counter() - t0
            st = _chain_stats(emb) if emb else {"embedded": False}
            rows.append({"N": N, "density": density, "coefficient_range": crange,
                         "topology": name, "method": "clique", "time_s": t_clique, **st})
            print(f"{N:>4} {density:>8.3f} {crange:>11.3g}  {name:<24} {'clique':<10} "
                  f"{str(st['embedded']):>4} {st.get('max_chain', '-'):>9} "
                  f"{st.get('physical_qubits', '-'):>12} {t_clique:>8.2f}", flush=True)
            # heuristic embedding (skip the very largest to bound runtime)
            t0 = time.perf_counter()
            try:
                emb_h = minorminer.find_embedding(list(K.edges), G, timeout=HEURISTIC_TIMEOUT_S,
                                                  random_seed=1)
            except Exception as exc:  # pragma: no cover
                emb_h = {}
                print("   heuristic error:", exc)
            t_h = time.perf_counter() - t0
            st_h = _chain_stats(emb_h) if emb_h else {"embedded": False}
            rows.append({"N": N, "density": density, "coefficient_range": crange,
                         "topology": name, "method": "heuristic", "time_s": t_h, **st_h})
            print(f"{'':>4} {'':>8} {'':>11}  {'':<24} {'heuristic':<10} "
                  f"{str(st_h['embedded']):>4} {st_h.get('max_chain', '-'):>9} "
                  f"{st_h.get('physical_qubits', '-'):>12} {t_h:>8.2f}", flush=True)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["N", "density", "coefficient_range", "topology", "method", "embedded",
              "max_chain", "mean_chain", "physical_qubits", "time_s"]
    with (FIG_DIR / "embedding_study.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    summary = {name: {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
                      "largest_clique_ideal": len(caches[name].largest_clique())}
               for name, G in targets.items()}
    (FIG_DIR / "embedding_study.json").write_text(
        json.dumps({"targets": summary, "rows": rows}, indent=2), encoding="utf-8")
    print("\n  written: scripts/figures/embedding_study.csv / .json")


if __name__ == "__main__":
    main()
