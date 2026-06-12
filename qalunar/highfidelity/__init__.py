"""High-fidelity truth oracles for the binary scheduling loop.

The iterative re-linearisation driver only requires a *truth oracle* --
a function that flies a candidate binary schedule and returns the real
final state. By default that oracle is the in-house planar-CR3BP RK4
(:func:`qalunar.qubo.thrust_scheduling.propagate_schedule`); this
subpackage provides a drop-in replacement backed by NASA GMAT with real
DE-series ephemerides, so the trust-region accept/reject decisions are
made against flight-fidelity dynamics while the QUBO itself keeps being
assembled from the cheap CR3BP linearisation.
"""

from qalunar.highfidelity.gmat_oracle import (
    GmatOracleConfig,
    build_gmat_linearized_qubo,
    build_schedule_script,
    find_gmat_console,
    make_gmat_truth_propagator,
    propagate_schedule_gmat,
    rotating_km_to_synodic,
    solve_gmat_iterative,
    synodic_to_rotating_km,
)

__all__ = [
    "GmatOracleConfig",
    "build_gmat_linearized_qubo",
    "build_schedule_script",
    "find_gmat_console",
    "make_gmat_truth_propagator",
    "propagate_schedule_gmat",
    "rotating_km_to_synodic",
    "solve_gmat_iterative",
    "synodic_to_rotating_km",
]
