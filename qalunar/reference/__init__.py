"""Classical reference trajectories for the planar CR3BP.

Currently exposes:

* :class:`DirectCollocationConfig` / :func:`solve_energy_optimal_cr3bp`:
  Hermite-Simpson direct collocation for the energy-optimal planar
  CR3BP TPBVP. Produces a smooth state-and-control reference that can
  be fed to :meth:`qalunar.transcription.IndirectTfcElmTranscription.solve_sequential`
  as ``initial_nominal`` to warm-start the indirect iteration, and that
  doubles as a classical NLP baseline for the QUBO benchmark sweep.
* :class:`EdelbaumSpiral` / :func:`edelbaum_spiral`: closed-form
  analytical Phase-1 solution for the low-thrust Earth-escape spiral,
  used to hand off a ``(r, v)`` synodic state to the CR3BP Phase-2
  transcription.
"""

from qalunar.reference.direct_collocation import (
    DirectCollocationConfig,
    DirectCollocationResult,
    solve_energy_optimal_cr3bp,
)
from qalunar.reference.edelbaum import (
    EdelbaumSpiral,
    edelbaum_spiral,
    edelbaum_spiral_moon_descending,
)
from qalunar.reference.handoff import (
    PerilunePass,
    capture_delta_v,
    find_perilune,
    moon_relative_inertial,
)
from qalunar.reference.numeric_spiral import (
    NumericSpiralResult,
    phase1_spiral_numeric,
)
from qalunar.reference.primer_vector import (
    PrimerVectorTrace,
    reconstruct_primer_vector,
)

__all__ = [
    "DirectCollocationConfig",
    "DirectCollocationResult",
    "solve_energy_optimal_cr3bp",
    "EdelbaumSpiral",
    "edelbaum_spiral",
    "edelbaum_spiral_moon_descending",
    "PerilunePass",
    "capture_delta_v",
    "find_perilune",
    "moon_relative_inertial",
    "NumericSpiralResult",
    "phase1_spiral_numeric",
    "PrimerVectorTrace",
    "reconstruct_primer_vector",
]
