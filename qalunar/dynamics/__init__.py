"""Dynamics models for trajectory optimization.

Each dynamics model provides a uniform interface:
    rhs(state, control) -> d(state)/dt
    jacobian_state(state) -> df/dx   (analytical, (n_state, n_state))
    jacobian_control(state) -> df/du (analytical, (n_state, n_control))

so that the same transcription code (direct LGL pseudospectral, indirect
TFC+ELM, etc.) can be applied to different dynamical contexts.
"""

from qalunar.dynamics.cr3bp import EARTH_MOON_MU, PlanarCR3BP

__all__ = ["EARTH_MOON_MU", "PlanarCR3BP"]
