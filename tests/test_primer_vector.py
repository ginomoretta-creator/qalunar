"""Tests for the primer-vector reconstructor."""

from __future__ import annotations

import numpy as np
import pytest

from qalunar.dynamics import PlanarCR3BP
from qalunar.dynamics.cr3bp import EARTH_MOON_MU
from qalunar.reference.primer_vector import (
    PrimerVectorTrace,
    reconstruct_primer_vector,
)


def _coast_trajectory(T: float = 0.5, n_steps: int = 200):
    """Generate an unforced CR3BP trajectory near the L4 region."""
    dyn = PlanarCR3BP()
    state0 = np.array([-EARTH_MOON_MU + 0.5, 0.0, 0.0, 0.6])
    t, traj = dyn.propagate(state0, (0.0, T), n_steps=n_steps)
    return dyn, t, traj


def test_returns_correct_shapes():
    dyn, t, traj = _coast_trajectory()
    target = traj[-1] + np.array([0.01, 0.0, 0.0, 0.0])

    trace = reconstruct_primer_vector(dyn, t, traj, target)

    assert isinstance(trace, PrimerVectorTrace)
    assert trace.times.shape == t.shape
    assert trace.lambdas.shape == traj.shape
    assert trace.switching.shape == t.shape
    assert trace.primer_norm.shape == t.shape


def test_zero_miss_gives_zero_costate():
    """If x_f == target, lambda(t_f) = 0 => lambda(t) = 0 => S(t) = 0.

    This pins the terminal transversality and the linearity of the
    backward ODE in lambda.
    """
    dyn, t, traj = _coast_trajectory()
    target = traj[-1].copy()  # exact match

    trace = reconstruct_primer_vector(dyn, t, traj, target)

    assert np.allclose(trace.lambdas, 0.0, atol=1e-12)
    assert np.allclose(trace.switching, 0.0, atol=1e-12)
    assert np.allclose(trace.primer_norm, 0.0, atol=1e-12)


def test_terminal_costate_matches_transversality():
    """lambda(t_f) must equal 2 W (x_f - target) up to roundoff."""
    dyn, t, traj = _coast_trajectory()
    target = traj[-1] + np.array([0.02, -0.01, 0.005, 0.003])
    weights = np.array([1.0, 1.0, 0.5, 0.5])

    trace = reconstruct_primer_vector(
        dyn, t, traj, target, target_weights=weights
    )

    expected_lam_f = 2.0 * weights * (traj[-1] - target)
    assert np.allclose(trace.lambdas[-1], expected_lam_f, atol=1e-14)


def test_linearity_in_miss():
    """Doubling the miss doubles the costate and the switching function.

    The backward ODE is linear in lambda; the terminal condition is
    linear in (x_f - target). So scaling the miss by alpha must scale
    every reconstructed quantity by alpha.
    """
    dyn, t, traj = _coast_trajectory()
    base_dx = np.array([0.01, -0.005, 0.002, 0.001])
    target_a = traj[-1] - base_dx
    target_b = traj[-1] - 2.0 * base_dx

    trace_a = reconstruct_primer_vector(dyn, t, traj, target_a)
    trace_b = reconstruct_primer_vector(dyn, t, traj, target_b)

    assert np.allclose(trace_b.lambdas, 2.0 * trace_a.lambdas, atol=1e-12)
    assert np.allclose(
        trace_b.switching, 2.0 * trace_a.switching, atol=1e-12
    )


def test_validates_input_shapes():
    dyn = PlanarCR3BP()
    t = np.linspace(0.0, 1.0, 50)
    traj = np.zeros((50, 4))
    target = np.zeros(4)

    # Wrong state shape
    with pytest.raises(ValueError):
        reconstruct_primer_vector(dyn, t, traj[:, :3], target)

    # Time / state size mismatch
    with pytest.raises(ValueError):
        reconstruct_primer_vector(dyn, t[:10], traj, target)

    # Non-monotonic times
    t_bad = t.copy()
    t_bad[5] = t_bad[4]
    with pytest.raises(ValueError):
        reconstruct_primer_vector(dyn, t_bad, traj, target)

    # Wrong weights shape
    with pytest.raises(ValueError):
        reconstruct_primer_vector(
            dyn, t, traj, target, target_weights=np.ones(3)
        )
