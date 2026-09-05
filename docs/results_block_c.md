# Block C results — benchmarks and hardware readiness (2026-09-05)

All numbers come from `scripts/figures/`; the manuscript tables are generated
from these files by `scripts/make_paper_tables.py`.

## Minor embedding (`run_embedding_study.py`, `embedding_study.csv`)

Ideal graphs: Pegasus P16 (Advantage, 5,640 qubits, 40,484 couplers) and
Zephyr Z12 (Advantage2, 4,800 qubits, 45,864 couplers). Largest ideal cliques:
K₁₈₀ (Pegasus) and K₁₈₄ (Zephyr). The scheduling QUBO is dense (K_N).

| N | coef. range | Pegasus clique (max chain / qubits) | Zephyr clique | Pegasus heuristic |
|---|---|---|---|---|
| 10 | 247 | 2 / 20 | 3 / 30 | 2 / 16 |
| 15 | 391 | 3 / 45 | 3 / 45 | 3 / 36 |
| 20 | 549 | 3 / 60 | 3 / 60 | 3 / 54 |
| 50 | 1,540 | 6 / 290 | 5 / 250 | 8 / 283 |
| 100 | 3,210 | 10 / 982 | 8 / 800 | 22 / 1,366 |
| 150 | 4,880 | 14 / 2,074 | 11 / 1,650 | fails |
| 200 | 6,550 | **fails** | **fails** | fails |

The earlier manuscript's "N = 200: Yes (hybrid + QPU)" was wrong; the dense
instance is processor-embeddable up to N ≈ 150–180. Real devices have a few
percent of inoperable qubits; ideal-graph numbers are upper bounds.

## Direct-collocation mesh study (`run_collocation_mesh_study.py`)

Benchmark case r0 = (−0.3, 0), v0 = (0, 0.6), rf = (0.4, 0.2), vf = (−0.1, 0),
T = 2.5; Hermite–Simpson defects, Simpson-consistent cost, warm starts between
meshes, feasibility-gated `success`.

| N | J | max defect | feasible | SLSQP iterations | time (s) |
|---|---|---|---|---|---|
| 20 | 4.899 | 1.2e-12 | yes | 119 | 3 |
| 40 | 4.108 | 2.3e-11 | yes | 139 | 12 |
| 80 | 3.884 | 2.6e-12 | yes | 400 | 98 |
| 160 | 2.603 | 1.2e-12 | yes | 580 | 468 |
| 320 | (running) | | | | |

The N = 40 reference the earlier papers quoted is ≥ 60 % above the N = 160
value and the sequence is not in the asymptotic regime. Objective-value
comparisons "QUBO vs classical NLP" are withdrawn; feasibility (miss)
comparisons stand.

## NLP-vs-QUBO (`run_nlp_vs_qubo_comparison.py`)

| scenario | method | miss (km) | Δv (m/s) | burns |
|---|---|---|---|---|
| under-injection, T=2 | QUBO iterative | 2053 | 10.9 | 4/15 |
| | NLP HS (SLSQP) | 0 | 7.7 | — |
| under-injection, T=3 | QUBO iterative | 1087 | 8.2 | 2/15 |
| | NLP HS (SLSQP) | 0 | 8.8 | — |
| strong under-injection, T=3 | QUBO iterative | 2626 | 20.5 | 5/15 |
| | NLP HS (SLSQP) | 0 | 14.6 | — |

## Robustness (`run_robustness_sweep.py`, `scheduling_robustness.png`)

Prograde-only single channel corrects under-injection (ratio < 1.190) with
2–5 burns; cannot act on over-injection (0 burns); linearisation breaks beyond
T ≈ 4 (miss ~3e5 km).

## Solver tiers

- **MILP/HiGHS**: objective must be rescaled to unit max coefficient; unscaled,
  HiGHS returned "optimal" 6.26e-6 against the exact 5.85e-6 at N = 15
  (energies are of the order of its absolute tolerances). Fixed in
  `milp_baseline.py`.
- **Simulated bifurcation**: the published `simulated-bifurcation` 2.0.0
  package (PyTorch), discrete mode, float32, positional `(Q, l, c)` polynomial.
  Exact at N ≤ 20; at N = 50/100 finds energies below SA-1000 (3.1e-6 vs
  7.8e-6; 5.0e-6 vs 7.6e-6). Ballistic/heated modes are worse. The in-house
  NumPy dSB collapses to the coast solution at N ≥ 50 and is kept only as a
  fallback.
- **Kerberos** (dwave-hybrid 0.6.16): no per-sampler seed; global RNGs seeded.
- **SA vs brute force**: 17/18 random instances (N ∈ {15,18,20}, 1000 reads)
  recover the optimum; N = 20 seed 2 stays 2× above at 3000 reads.

## Time-to-target sweep (`run_scaling_sweep.py`, `scaling_tts.csv`)

Twenty seeds per stochastic solver, per-run budgets SA 100 reads / SB 100
agents / Kerberos 3 iterations; "hit" = within 1 % of the exact optimum
(N ≤ 20) or the best-known energy. Penultimate run (Kerberos fixed, SB
published, MILP unscaled):

| N | best tier(s) at target | notes |
|---|---|---|
| 10–20 | all exact; SA fastest (TTT 0.03–0.07 s), Kerberos 0.32 s, SB 0.6–2.3 s | SA p = 0.8 at N = 15 |
| 50 | SB (3.13e-6, p = 0.10, TTT 63 s) | SA 6.4e-6, Kerberos 4.1e-6 never within 1 % |
| 100 | Kerberos (2.99e-6, p = 1.00, TTT 0.46 s) | SB 4.3e-6, SA 7.8e-6 |
| 200 | SB (3.80e-6, p = 0.05, TTT 547 s) | Kerberos 4.6e-6, SA 7.0e-6; MILP incumbent 1.2e-3 at 60 s |

No tier dominates; the final table (scaled MILP) is `tables/tab_tts.tex`.
