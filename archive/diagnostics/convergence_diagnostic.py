"""Diagnose convergence vs resolution for the classical solver.

Sweeps n_training and n_basis to find where the NL residual actually
converges to a small value.
"""
from __future__ import annotations

import time

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

MAX_ITER = 40

print(f"{'n_train':>8} {'n_basis':>8} {'n_eq':>6} {'n_unk':>6} "
      f"{'ratio':>6} {'iters':>6} {'final_NL':>12} {'conv':>5} {'time':>7}")
print("-" * 75)

for n_train in [8, 12, 16, 20, 25, 30, 40]:
    for n_basis in [20, 40, 60, 80]:
        n_eq = 6 * n_train
        n_unk = 6 * n_basis
        if n_unk < n_eq:
            # overdetermined -- skip (lstsq handles it but not typical)
            continue

        cfg = IndirectTfcElmConfig(n_training=n_train, n_basis=n_basis, seed=7)
        trans = IndirectTfcElmTranscription(dynamics=PlanarCR3BP(), config=cfg, **BCS)

        t0 = time.perf_counter()
        sol = trans.solve_sequential(
            OuterLoopConfig(max_iter=MAX_ITER, tol=1e-7, damping=1.0, line_search=True)
        )
        elapsed = time.perf_counter() - t0

        final_nl = sol.history["nonlinear_residual"][-1]
        ratio = n_unk / n_eq

        print(f"{n_train:8d} {n_basis:8d} {n_eq:6d} {n_unk:6d} "
              f"{ratio:6.1f} {sol.n_iterations:6d} {final_nl:12.4e} "
              f"{'Y' if sol.converged else 'N':>5} {elapsed:7.1f}s")
