"""Edelbaum analytical spiral for Phase-1 of a two-phase Earth--Moon transfer.

This module handles the ``boring but expensive`` first leg of a low-thrust
Earth-to-Moon mission: lifting the spacecraft from a low circular Earth
orbit to a high circular Earth orbit via continuous tangential thrust,
with Moon gravity ignored. In this regime the dynamics are pure two-body
Keplerian around Earth and the problem admits a closed-form answer --- no
optimization needed, tangential thrust is already Edelbaum-optimal for a
coplanar circular-to-circular transfer.

Phase 1 output is then used as the initial state for Phase 2, which is the
CR3BP three-body transfer that ``qalunar.reference.direct_collocation`` and
``qalunar.transcription`` solve. The split lets each tool handle what it is
good at: an analytical spiral for the hundreds-of-revolutions climb out of
Earth's gravity well, and the TFC+ELM / QUBO machinery for the three-body
handoff to the Moon.

Physical unit conversions (Earth-Moon CR3BP nondimensionalisation)::

    L = 384_400 km                          distance unit
    T = 27.321_661 * 86400 / (2 pi) seconds  time unit (~ 3.7569e5 s)
    V = L / T ~ 1023 m/s                     velocity unit
    A = V / T ~ 2.72e-3 m/s**2               acceleration unit

In these units Earth's two-body gravitational parameter is ``GM_Earth = 1 - mu``
and Moon's is ``GM_Moon = mu``, with ``mu ~ 0.01215``.

References
----------
* Edelbaum, T.N., *Propulsion Requirements for Controllable Satellites*,
  ARS Journal 31(8):1079-1089, 1961.
* Kechichian, J.A., *Reformulation of Edelbaum's Low-Thrust Transfer
  Problem Using Optimal Control Theory*, J. Guidance Control and
  Dynamics 20(5):988-994, 1997.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from qalunar.dynamics.cr3bp import EARTH_MOON_MU


# ---------------------------------------------------------------------------
# Earth-Moon CR3BP nondimensional -> SI conversion factors
# ---------------------------------------------------------------------------

LENGTH_KM: float = 384_400.0
"""Distance unit: Earth-Moon mean distance, kilometres."""

TIME_S: float = 27.321_661 * 86_400.0 / (2.0 * np.pi)
"""Time unit: 1 / n where n is the Moon's sidereal mean motion, seconds."""

VELOCITY_M_S: float = LENGTH_KM * 1000.0 / TIME_S
"""Velocity unit, metres per second. Approximately 1023 m/s."""

ACCELERATION_M_S2: float = VELOCITY_M_S / TIME_S
"""Acceleration unit, m/s**2. Approximately 2.72e-3 m/s**2 = 2.72 mm/s**2."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EdelbaumSpiral:
    """Closed-form summary of an Edelbaum spiral around Earth or Moon.

    All quantities are stored in CR3BP nondimensional units. Use the
    ``si_report`` helper to obtain a physical-unit dictionary suitable
    for printing or for a manual caption.

    Attributes
    ----------
    r1, r2 : float
        Initial and final circular-orbit radius measured from the
        primary body (``body``).
    v1, v2 : float
        Circular-orbit inertial speeds at r1 and r2, respectively.
    dv : float
        Total Delta-v required. Equals ``abs(v1 - v2)``.
    tof : float
        Time of flight. Equals ``dv / a_thrust`` under the coplanar
        Edelbaum assumption of constant tangential thrust.
    n_revs : float
        Total number of revolutions around the primary body during the
        spiral, obtained by integrating the instantaneous orbital
        angular rate against the linearly varying orbital speed v(t).
    a_thrust : float
        Constant tangential thrust-acceleration magnitude that was
        assumed.
    mu : float
        CR3BP mass parameter (``mu = m_moon / (m_earth + m_moon)``).
    body : str
        Primary body around which the spiral is computed: ``"earth"``
        (Phase-1 ascending spiral) or ``"moon"`` (Phase-3 descending
        spiral for lunar orbit insertion). Determines whether
        ``GM = 1 - mu`` (Earth) or ``GM = mu`` (Moon) and where the
        handoff state is placed in the synodic frame.
    """

    r1: float
    r2: float
    v1: float
    v2: float
    dv: float
    tof: float
    n_revs: float
    a_thrust: float
    mu: float
    body: str = "earth"

    @property
    def gm(self) -> float:
        """Gravitational parameter of the primary body."""
        if self.body == "earth":
            return 1.0 - self.mu
        if self.body == "moon":
            return self.mu
        raise ValueError(f"Unknown body {self.body!r}; expected 'earth' or 'moon'")

    def handoff_state_synodic(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Position and velocity in CR3BP synodic coordinates at end of spiral.

        The spacecraft is placed on a circular prograde orbit of radius
        ``r2`` around the primary body (``self.body``), at the moment its
        body-relative position crosses the ``+x`` direction. For an
        Earth-centered spiral this is ``(-mu + r2, 0)``; for a Moon-centered
        spiral it is ``(1 - mu + r2, 0)``.

        For ``body="earth"`` we use the small-mu approximation that
        treats Earth as stationary in the inertial frame (Earth's
        ``omega x r_earth`` of magnitude ``mu`` is dropped); the
        synodic velocity is then ``v_inertial - omega x r_spacecraft``
        with ``v_inertial = (0, v2)``.

        For ``body="moon"`` the Moon's inertial velocity ``omega x r_moon``
        has magnitude ``1 - mu``, which is comparable to or larger than
        the Moon-relative orbital speed and cannot be dropped. We
        therefore use the full formula
        ``v_inertial = v_rel_inertial + omega x r_moon``.
        """
        if self.body == "earth":
            r_syn = np.array([-self.mu + self.r2, 0.0])
            v_inertial = np.array([0.0, self.v2])
        elif self.body == "moon":
            moon_pos = np.array([1.0 - self.mu, 0.0])
            r_syn = moon_pos + np.array([self.r2, 0.0])
            v_rel_inertial = np.array([0.0, self.v2])
            v_moon_inertial = np.array([-moon_pos[1], moon_pos[0]])
            v_inertial = v_rel_inertial + v_moon_inertial
        else:
            raise ValueError(
                f"Unknown body {self.body!r}; expected 'earth' or 'moon'"
            )

        omega_cross_r = np.array([-r_syn[1], r_syn[0]])
        v_syn = v_inertial - omega_cross_r
        return r_syn, v_syn

    def si_report(self) -> dict[str, str]:
        """Return a human-readable physical-units summary of the spiral.

        Useful for the manual's Phase-1 caption and for progress logs.
        All numeric values are pre-formatted as strings with units
        embedded, so the caller can drop them straight into f-strings
        or LaTeX.
        """
        tof_days = self.tof * TIME_S / 86_400.0
        return {
            "body": self.body,
            "r1_km": f"{self.r1 * LENGTH_KM:,.0f} km",
            "r2_km": f"{self.r2 * LENGTH_KM:,.0f} km",
            "v1_m_s": f"{self.v1 * VELOCITY_M_S:,.0f} m/s",
            "v2_m_s": f"{self.v2 * VELOCITY_M_S:,.0f} m/s",
            "dv_m_s": f"{self.dv * VELOCITY_M_S:,.0f} m/s",
            "tof_days": f"{tof_days:,.1f} days",
            "tof_years": f"{tof_days / 365.25:.2f} yr",
            "n_revs": f"{self.n_revs:,.0f}",
            "a_thrust_mm_s2": (
                f"{self.a_thrust * ACCELERATION_M_S2 * 1000.0:.3f} mm/s**2"
            ),
        }


# ---------------------------------------------------------------------------
# The analytical spiral
# ---------------------------------------------------------------------------


def edelbaum_spiral(
    *,
    r1: float,
    r2: float,
    a_thrust: float,
    mu: float = EARTH_MOON_MU,
) -> EdelbaumSpiral:
    """Analytical coplanar Edelbaum spiral between two Earth-centered orbits.

    Assumes continuous tangential thrust in the Earth-centered two-body
    problem, ignoring Moon and Sun perturbations. The spiral stays
    instantaneously circular at each moment, so the instantaneous
    orbital speed evolves linearly in time::

        v(t) = v1 + sign(v2 - v1) * a_thrust * t,

    and integrates to ``|v1 - v2| = a_thrust * tof``. The revolution
    count comes from integrating the orbital angular rate
    ``n_orb = v / r = v**3 / GM`` against ``dt``, giving the
    closed-form result

    .. math::
       N_{rev} = \\frac{|v_1^4 - v_2^4|}{8 \\pi \\, GM_\\oplus \\, a_t}.

    Parameters
    ----------
    r1, r2 : float
        Initial and final circular-orbit radius from Earth, in CR3BP
        nondimensional units (1 unit = Earth-Moon distance). Must be
        positive and distinct.
    a_thrust : float
        Constant tangential thrust acceleration magnitude in CR3BP
        nondimensional units. Must be positive.
    mu : float, optional
        CR3BP mass parameter. Defaults to Earth-Moon.

    Returns
    -------
    EdelbaumSpiral
        Closed-form Phase-1 summary including the synodic handoff state.

    Raises
    ------
    ValueError
        If ``r1`` or ``r2`` is non-positive, if ``r1 == r2``, or if
        ``a_thrust`` is non-positive.
    """
    if r1 <= 0.0 or r2 <= 0.0:
        raise ValueError(f"r1 and r2 must be positive; got r1={r1}, r2={r2}")
    if r1 == r2:
        raise ValueError("r1 must differ from r2 for a non-trivial spiral")
    if a_thrust <= 0.0:
        raise ValueError(f"a_thrust must be positive; got {a_thrust}")

    gm_earth = 1.0 - mu
    v1 = float(np.sqrt(gm_earth / r1))
    v2 = float(np.sqrt(gm_earth / r2))
    dv = float(abs(v1 - v2))
    tof = float(dv / a_thrust)
    n_revs = float(abs(v1 ** 4 - v2 ** 4) / (8.0 * np.pi * gm_earth * a_thrust))

    return EdelbaumSpiral(
        r1=r1,
        r2=r2,
        v1=v1,
        v2=v2,
        dv=dv,
        tof=tof,
        n_revs=n_revs,
        a_thrust=a_thrust,
        mu=mu,
        body="earth",
    )


# ---------------------------------------------------------------------------
# Moon-centered descending spiral (Phase-3 lunar orbit insertion)
# ---------------------------------------------------------------------------


def edelbaum_spiral_moon_descending(
    *,
    r1: float,
    r2: float,
    a_thrust: float,
    mu: float = EARTH_MOON_MU,
) -> EdelbaumSpiral:
    """Analytical coplanar Edelbaum spiral *around the Moon*, descending.

    Phase-3 analogue of the Phase-1 Earth-ascending spiral. The
    spacecraft is assumed to begin on a high circular Moon-relative
    orbit at radius ``r1`` (e.g., the perilune of a flyby capture) and
    spiral down to a low circular Moon-relative orbit at radius
    ``r2 < r1`` (a target lunar orbit, LLO). Continuous anti-tangential
    thrust is applied, ignoring Earth perturbations -- a two-body
    Keplerian approximation around the Moon, the natural counterpart
    to the Phase-1 idealisation that ignored the Moon during the
    Earth-escape spiral.

    Note that ``r1`` and ``r2`` are measured **from the Moon**, in
    CR3BP nondimensional units (1 unit = Earth-Moon distance).

    Parameters
    ----------
    r1 : float
        Initial Moon-relative circular-orbit radius. Larger than ``r2``.
    r2 : float
        Final Moon-relative circular-orbit radius (target LLO).
    a_thrust : float
        Constant tangential thrust acceleration magnitude in CR3BP
        nondimensional units. Direction is anti-tangential during
        descent; only the magnitude matters for the closed-form
        Δv / TOF / revolution count.
    mu : float, optional
        CR3BP mass parameter. Defaults to Earth-Moon. ``GM_Moon = mu``.

    Returns
    -------
    EdelbaumSpiral
        Closed-form Phase-3 summary including the synodic-frame
        handoff state at the **end** of the descent (i.e. on the
        target LLO).

    Raises
    ------
    ValueError
        If radii are non-positive, equal, or if ``r1 < r2`` (which
        would be an ascending spiral, not a descending one).
    """
    if r1 <= 0.0 or r2 <= 0.0:
        raise ValueError(f"r1 and r2 must be positive; got r1={r1}, r2={r2}")
    if r1 == r2:
        raise ValueError("r1 must differ from r2 for a non-trivial spiral")
    if r1 < r2:
        raise ValueError(
            f"Descending spiral requires r1 > r2; got r1={r1}, r2={r2}. "
            "Did you swap the arguments?"
        )
    if a_thrust <= 0.0:
        raise ValueError(f"a_thrust must be positive; got {a_thrust}")

    gm_moon = mu
    v1 = float(np.sqrt(gm_moon / r1))
    v2 = float(np.sqrt(gm_moon / r2))
    dv = float(abs(v1 - v2))
    tof = float(dv / a_thrust)
    n_revs = float(abs(v1 ** 4 - v2 ** 4) / (8.0 * np.pi * gm_moon * a_thrust))

    return EdelbaumSpiral(
        r1=r1,
        r2=r2,
        v1=v1,
        v2=v2,
        dv=dv,
        tof=tof,
        n_revs=n_revs,
        a_thrust=a_thrust,
        mu=mu,
        body="moon",
    )
