"""Deeper diagnostic: track both linearized and NL residuals per iteration."""
from __future__ import annotations

import numpy as np

from qalunar.dynamics import PlanarCR3BP
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

# Best-performing config from sweep
cfg = IndirectTfcElmConfig(n_training=12, n_basis=40, seed=7)
trans = IndirectTfcElmTranscription(dynamics=PlanarCR3BP(), config=cfg, **BCS)

sol = trans.solve_sequential(
    OuterLoopConfig(max_iter=30, tol=1e-7, damping=1.0, line_search=True)
)

print(f"{'iter':>5} {'step_norm':>12} {'lin_res':>12} {'NL_res':>12} {'step_size':>10}")
print("-" * 55)
for i in range(sol.n_iterations):
    print(f"{i+1:5d} "
          f"{sol.history['trajectory_change_inf'][i]:12.4e} "
          f"{sol.history['linearized_residual'][i]:12.4e} "
          f"{sol.history['nonlinear_residual'][i]:12.4e} "
          f"{sol.history['step_size'][i]:10.4f}")

print(f"\nConverged: {sol.converged}")
print(f"System: A is ({trans.n_equations} x {trans.n_unknowns})")
print(f"Condition number of final A:")

# Build the linear system at the final nominal to check conditioning
nominal_x = sol.trajectory["x"]
nominal_y = sol.trajectory["y"]
A, B = trans.build_linear_system(nominal_x, nominal_y)
print(f"  shape: {A.shape}")
print(f"  cond(A): {np.linalg.cond(A):.2e}")
print(f"  rank(A): {np.linalg.matrix_rank(A)}")
print(f"  ||A xi - B||: {np.linalg.norm(A @ sol.xi - B):.4e}")

# Check if lstsq residual is zero (underdetermined => should be)
xi_lstsq, residuals, rank, sv = np.linalg.lstsq(A, B, rcond=None)
print(f"  lstsq rank: {rank}")
print(f"  ||A xi_lstsq - B||: {np.linalg.norm(A @ xi_lstsq - B):.4e}")
print(f"  singular values (first 10): {sv[:10]}")
print(f"  singular values (last 10):  {sv[-10:]}")
