"""Identify which equation blocks dominate the NL residual and test
whether disabling line search (fixed damping=1.0) converges faster."""
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

cfg = IndirectTfcElmConfig(n_training=12, n_basis=40, seed=7)
trans = IndirectTfcElmTranscription(dynamics=PlanarCR3BP(), config=cfg, **BCS)

# ---- Run 1: with Armijo (what we had) ----
print("=== With Armijo line search ===")
sol1 = trans.solve_sequential(
    OuterLoopConfig(max_iter=30, tol=1e-7, damping=1.0, line_search=True)
)
nl_dict1 = trans.nonlinear_residual(sol1.xi)
print(f"Final NL residual: {sol1.history['nonlinear_residual'][-1]:.4e}")
print("Per-block max|residual|:")
for k, v in nl_dict1.items():
    print(f"  {k:6s}: {np.max(np.abs(v)):.4e}")

# ---- Run 2: NO line search, full steps ----
print("\n=== No line search, damping=1.0 (undamped Picard) ===")
trans2 = IndirectTfcElmTranscription(dynamics=PlanarCR3BP(), config=cfg, **BCS)
sol2 = trans2.solve_sequential(
    OuterLoopConfig(max_iter=30, tol=1e-7, damping=1.0, line_search=False)
)
print(f"Converged: {sol2.converged}")
print(f"{'iter':>5} {'step_norm':>12} {'NL_res':>12}")
for i in range(sol2.n_iterations):
    print(f"{i+1:5d} {sol2.history['trajectory_change_inf'][i]:12.4e} "
          f"{sol2.history['nonlinear_residual'][i]:12.4e}")

nl_dict2 = trans2.nonlinear_residual(sol2.xi)
print("Per-block max|residual|:")
for k, v in nl_dict2.items():
    print(f"  {k:6s}: {np.max(np.abs(v)):.4e}")

# ---- Run 3: NO line search, damping=0.5 ----
print("\n=== No line search, damping=0.5 ===")
trans3 = IndirectTfcElmTranscription(dynamics=PlanarCR3BP(), config=cfg, **BCS)
sol3 = trans3.solve_sequential(
    OuterLoopConfig(max_iter=60, tol=1e-7, damping=0.5, line_search=False)
)
print(f"Converged: {sol3.converged}")
print(f"{'iter':>5} {'step_norm':>12} {'NL_res':>12}")
for i in range(sol3.n_iterations):
    print(f"{i+1:5d} {sol3.history['trajectory_change_inf'][i]:12.4e} "
          f"{sol3.history['nonlinear_residual'][i]:12.4e}")

# ---- Run 4: Higher resolution with undamped Picard ----
print("\n=== Higher res: n_train=20, n_basis=80, undamped Picard ===")
cfg4 = IndirectTfcElmConfig(n_training=20, n_basis=80, seed=7)
trans4 = IndirectTfcElmTranscription(dynamics=PlanarCR3BP(), config=cfg4, **BCS)
sol4 = trans4.solve_sequential(
    OuterLoopConfig(max_iter=60, tol=1e-7, damping=1.0, line_search=False)
)
print(f"Converged: {sol4.converged}")
print(f"{'iter':>5} {'step_norm':>12} {'NL_res':>12}")
for i in range(min(sol4.n_iterations, 30)):
    print(f"{i+1:5d} {sol4.history['trajectory_change_inf'][i]:12.4e} "
          f"{sol4.history['nonlinear_residual'][i]:12.4e}")
if sol4.n_iterations > 30:
    print(f"  ... ({sol4.n_iterations} total iterations)")
    print(f"  last: {sol4.history['nonlinear_residual'][-1]:.4e}")
