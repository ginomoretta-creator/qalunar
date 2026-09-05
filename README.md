# qalunar — binary thrust scheduling for cislunar transfers as a QUBO

Research code behind the manuscript on naturally-binary quantum-annealing
transcription of duty-cycle-limited electric propulsion for CubeSat cislunar
transfers (Moretta, Paolozzi, Ortore; Sapienza / UNC). The decision variables
are engine on/off commands per time slot; per-slot sensitivities are measured
either in the in-house planar CR3BP (state-transition matrices) or by finite
differences in NASA GMAT, the QUBO is assembled from them, and every accepted
candidate is re-flown in the truth model under a monotone trust-region rule.

## Layout

| Path | What it is |
|---|---|
| `qalunar/dynamics/cr3bp.py` | Planar CR3BP, analytic Jacobian, STM (variational equations), piecewise-control STM |
| `qalunar/qubo/thrust_scheduling.py` | The scheduling QUBO, `solve_iterative` (re-linearisation + trust region), `propagate_schedule` |
| `qalunar/qubo/cardinality.py` | Cardinality-`k` slot selection QUBO (duty-cycle budget) with a principled penalty |
| `qalunar/qubo/receding_horizon.py` | Window chaining (sliding window), records every flown segment |
| `qalunar/qubo/earth_escape.py`, `lunar_capture.py` | Phase adapters over the receding-horizon driver |
| `qalunar/qubo/scheduling_samplers.py` | Brute force, simulated annealing, Kerberos (local classical hybrid), Leap hybrid, D-Wave QPU |
| `qalunar/qubo/milp_baseline.py` | McCormick-lifted MILP (HiGHS) — constraint-native reference |
| `qalunar/highfidelity/gmat_oracle.py` | GMAT truth model: script templating, headless run, report parsing, finite-difference QUBO |
| `qalunar/reference/` | Classical baselines: Edelbaum, Hermite–Simpson direct collocation, primer vector |
| `scripts/run_conae_duty_cycle.py` | Phase 1: per-slot perigee gains in GMAT, cardinality QUBO, fly-back validation, 5-pass climb |
| `scripts/run_conae_phase2_binary.py` | Phase 2: receding-horizon binary apogee raise in GMAT, one QUBO per perigee pass |
| `scripts/run_conae_phase3_encounter.py` | Phase 3: encounter by waiting, arrival metrics, capture feasibility under the power budget |
| `scripts/run_scaling_sweep.py` | Solver benchmark with time-to-target statistics (BF, MILP, SA, SB, Kerberos) |
| `scripts/run_embedding_study.py` | Offline minor embedding of the dense QUBO into Pegasus P16 / Zephyr Z12 |
| `scripts/run_collocation_mesh_study.py` | Mesh refinement of the direct-collocation reference (warm-started, Richardson) |
| `scripts/make_paper_tables.py` | Generates the manuscript's tables from the cached artifacts |
| `scripts/run_*.py` (others) | Further experiments (see docstrings); `scripts/figures/` holds all cached outputs |
| `docs/results_block_b.md`, `docs/results_block_c.md` | Result tables of the re-flown mission and of the benchmarks |
| `tests/` | `pytest` suite; GMAT-dependent tests skip when `GmatConsole.exe` is absent |

## Install

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows
pip install -r requirements-lock.txt   # exact versions the results were produced with
pip install -e .
pip install simulated-bifurcation      # optional: the SB tier of the benchmark (PyTorch)
```

`requirements.txt` / `pyproject.toml` give the loose floors; `requirements-lock.txt`
is the frozen environment (Python 3.14, NumPy 2.4, dimod 0.12.22, dwave-hybrid 0.6.16).

GMAT R2025a is needed for the high-fidelity experiments. Point the code at the
console binary with

```bash
set QALUNAR_GMAT_CONSOLE=C:\path\to\GMAT_R2025a\bin\GmatConsole.exe
```

## Reproduce

```bash
pytest                                            # ~3 min with GMAT, ~1 min without
python -m scripts.run_conae_duty_cycle            # Phase-1 duty-cycle slot selection (GMAT)
python -m scripts.run_conae_duty_cycle --replot   # figure only, from scripts/figures/*.csv
python -m scripts.run_smart1_duty_cycle           # SMART-1 cross-check (GMAT)
python -m scripts.run_scaling_sweep               # solver benchmark (BF / MILP / SA / Kerberos)
```

Every entry point caches its GMAT-produced data under `scripts/figures/`
(tracked in git) so figures regenerate without GMAT.

## Force model used by the GMAT oracle

`GmatOracleConfig` defaults: Earth 8×8 spherical harmonics, Moon and Sun as
point masses (DE ephemerides), spherical SRP (`cr = 1.8`, 1 m²), thruster mass
depletion on, `DualCone` eclipse model on the power system, RK89 at 1e-12.
Out-of-plane state is projected out for the planar bridge but logged
(`out_of_plane_log`) so the size of the projection is reportable.

## Citing

See `CITATION.cff`.
