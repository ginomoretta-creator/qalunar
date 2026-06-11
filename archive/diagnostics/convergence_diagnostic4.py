"""Compare classical (undamped, no line search) vs hybrid on the now-converging problem."""
from __future__ import annotations

import time
import numpy as np

from qalunar.dynamics import PlanarCR3BP
from qalunar.hybrid import HybridConfig, solve_hybrid
from qalunar.qubo import LinearLsqToQuboConfig
from qalunar.transcription import (
    IndirectTfcElmConfig,
    IndirectTfcElmTranscription,
    OuterLoopConfig,
)

BCS = dict(
    r0=np.array([-0.3, 0.0]),
    v0=np.array([0.0, 0.6]),
    rf=np.array([0.4, 0.2]),
    vf=np.array([-0.1, 0.0]),
    time_of_flight=2.5,
)

cfg = IndirectTfcElmConfig(n_training=12, n_basis=40, seed=7)

# ---- Classical: undamped Picard (the one that works) ----
print("=== Classical: undamped Picard (no line search) ===")
trans1 = IndirectTfcElmTranscription(dynamics=PlanarCR3BP(), config=cfg, **BCS)
t0 = time.perf_counter()
sol1 = trans1.solve_sequential(
    OuterLoopConfig(max_iter=40, tol=1e-7, damping=1.0, line_search=False)
)
t1 = time.perf_counter() - t0
print(f"Converged: {sol1.converged}, {sol1.n_iterations} iters, {t1:.1f}s")
for i in range(sol1.n_iterations):
    print(f"  {i+1:3d}  NL={sol1.history['nonlinear_residual'][i]:.4e}  "
          f"step={sol1.history['trajectory_change_inf'][i]:.4e}")

# ---- Hybrid: damping=1.0, no reranking ----
print("\n=== Hybrid ground-state: damping=1.0, 8 bits ===")
trans2 = IndirectTfcElmTranscription(dynamics=PlanarCR3BP(), config=cfg, **BCS)
t0 = time.perf_counter()
sol2 = solve_hybrid(
    trans2,
    config=HybridConfig(
        qubo=LinearLsqToQuboConfig(bits_per_variable=8),
        num_reads=200,
        rerank_top_k=30,
        rerank=False,
        max_iter=40,
        tol=1e-7,
        damping=1.0,
        seed=42,
    ),
)
t2 = time.perf_counter() - t0
print(f"Converged: {sol2.converged}, {sol2.n_iterations} iters, {t2:.1f}s")
for i in range(sol2.n_iterations):
    print(f"  {i+1:3d}  NL={sol2.history['nonlinear_residual'][i]:.4e}  "
          f"step={sol2.history['trajectory_change_inf'][i]:.4e}")

# ---- Hybrid: damping=1.0, WITH reranking ----
print("\n=== Hybrid + reranking: damping=1.0, 8 bits ===")
trans3 = IndirectTfcElmTranscription(dynamics=PlanarCR3BP(), config=cfg, **BCS)
t0 = time.perf_counter()
sol3 = solve_hybrid(
    trans3,
    config=HybridConfig(
        qubo=LinearLsqToQuboConfig(bits_per_variable=8),
        num_reads=200,
        rerank_top_k=30,
        rerank=True,
        max_iter=40,
        tol=1e-7,
        damping=1.0,
        seed=42,
    ),
)
t3 = time.perf_counter() - t0
print(f"Converged: {sol3.converged}, {sol3.n_iterations} iters, {t3:.1f}s")
for i in range(sol3.n_iterations):
    print(f"  {i+1:3d}  NL={sol3.history['nonlinear_residual'][i]:.4e}  "
          f"step={sol3.history['trajectory_change_inf'][i]:.4e}  "
          f"rank={sol3.history['best_qubo_rank'][i]:.0f}")
