"""Transcription of continuous-time trajectory optimization problems.

Currently exposes:

* :class:`IndirectTfcElmTranscription` / :class:`IndirectTfcElmConfig`:
  indirect (Pontryagin) Theory-of-Functional-Connections + Extreme-
  Learning-Machine transcription of the planar CR3BP energy-optimal
  TPBVP. Produces a linear system ``A @ xi = B`` that can be fed
  directly into :func:`qalunar.qubo.build_linear_lsq_qubo`.
"""

from qalunar.transcription.indirect_tfc_elm import (
    IndirectTfcElmConfig,
    IndirectTfcElmTranscription,
    OuterLoopConfig,
    SequentialSolution,
)

__all__ = [
    "IndirectTfcElmConfig",
    "IndirectTfcElmTranscription",
    "OuterLoopConfig",
    "SequentialSolution",
]
