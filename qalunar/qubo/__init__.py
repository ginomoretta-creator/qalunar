"""QUBO assembly utilities for trajectory transcription problems.

The central primitive is :func:`build_linear_lsq_qubo`, a reusable port
of Gino Moretta's MATLAB ``Rendezvous_GIno.m`` v2 pipeline:

* Tikhonov regularization of the linear system ``A @ xi = B``
* SVD rank reduction in the augmented basis
* per-variable adaptive binary encoding centered on the Tikhonov estimate
* assembly of a quadratic unconstrained binary optimization (QUBO) whose
  ground state approximates the regularized least-squares solution

The module is intentionally dynamics-agnostic: any transcription that
produces a linear-residual collocation system (HCW rendezvous, direct
LGL, indirect TFC+ELM with a linearized inner solve, ...) can feed
its ``(A, B)`` pair in and get a QUBO plus a decode path back.
"""

from qalunar.qubo.linear_lsq import (
    LinearLsqToQuboConfig,
    LinearLsqToQuboResult,
    build_linear_lsq_qubo,
)
from qalunar.qubo.samplers import (
    QuboSampleResult,
    sample_exact,
    sample_simulated_annealing,
)
from qalunar.qubo.thrust_scheduling import (
    ThrustSchedulingConfig,
    ThrustSchedulingQubo,
    build_thrust_scheduling_qubo,
)
from qalunar.qubo.lunar_capture import (
    LunarCaptureResult,
    LunarCaptureWindow,
    moon_orbit_apolune_perilune,
    moon_two_body_energy,
    reconstruct_full_trajectory,
    solve_lunar_capture_sliding_window,
)

__all__ = [
    "LinearLsqToQuboConfig",
    "LinearLsqToQuboResult",
    "LunarCaptureResult",
    "LunarCaptureWindow",
    "QuboSampleResult",
    "ThrustSchedulingConfig",
    "ThrustSchedulingQubo",
    "build_linear_lsq_qubo",
    "build_thrust_scheduling_qubo",
    "moon_orbit_apolune_perilune",
    "moon_two_body_energy",
    "reconstruct_full_trajectory",
    "sample_exact",
    "sample_simulated_annealing",
    "solve_lunar_capture_sliding_window",
]
