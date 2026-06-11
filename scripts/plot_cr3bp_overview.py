"""Visualize the planar Earth-Moon CR3BP.

Plots:
    * primaries at (-mu, 0) and (1 - mu, 0)
    * triangular Lagrange points L4 / L5
    * zero-velocity curves at several Jacobi constants bracketing the
      collinear Lagrange point energies (C_L1 ~ 3.188, C_L2 ~ 3.172,
      C_L3 ~ 3.012, C_L4 = C_L5 = 3.0)
    * a handful of unforced trajectories propagated with DOP853 to
      sanity-check the RHS (tadpole librations around L4 and L5 plus a
      retrograde test orbit)

Run from the project root::

    python scripts/plot_cr3bp_overview.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from qalunar.dynamics import PlanarCR3BP


def pseudo_potential_2U(
    cr3bp: PlanarCR3BP, x: NDArray[np.float64], y: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Evaluate ``2 * Omega(x, y)`` on a grid.

    At zero velocity this equals the Jacobi integral, so contours of
    this field at level ``C`` are exactly the zero-velocity curves for
    Jacobi constant ``C``.
    """
    mu = cr3bp.mu
    r1 = np.sqrt((x + mu) ** 2 + y ** 2)
    r2 = np.sqrt((x - 1.0 + mu) ** 2 + y ** 2)
    return x ** 2 + y ** 2 + 2.0 * (1.0 - mu) / r1 + 2.0 * mu / r2 + mu * (1.0 - mu)


def propagate(
    cr3bp: PlanarCR3BP,
    state0: NDArray[np.float64],
    t_final: float,
    n_points: int = 4000,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Integrate the unforced CR3BP with DOP853 at tight tolerances."""

    def rhs(_t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        return cr3bp.rhs(y)

    sol = solve_ivp(
        rhs,
        (0.0, t_final),
        state0,
        method="DOP853",
        t_eval=np.linspace(0.0, t_final, n_points),
        rtol=1e-12,
        atol=1e-12,
    )
    return sol.t, sol.y


def main() -> None:
    cr3bp = PlanarCR3BP()
    mu = cr3bp.mu
    earth = cr3bp.earth_position()
    moon = cr3bp.moon_position()
    l4, l5 = cr3bp.lagrange_triangular_points()

    fig, ax = plt.subplots(figsize=(9.5, 9.0))

    # ------------------------------------------------------------------
    # Zero-velocity curves: contours of 2*Omega at several Jacobi energies
    # ------------------------------------------------------------------
    grid = np.linspace(-1.6, 1.6, 900)
    xx, yy = np.meshgrid(grid, grid)
    two_omega = pseudo_potential_2U(cr3bp, xx, yy)
    # Clip so the 1/r singularities at the primaries don't blow up the
    # color scale and hide the interesting structure.
    two_omega_clip = np.clip(two_omega, 2.8, 4.2)

    levels = [3.0, 3.05, 3.15, 3.172, 3.188, 3.3]
    cs = ax.contour(
        xx,
        yy,
        two_omega_clip,
        levels=levels,
        colors="#888888",
        linewidths=0.9,
        alpha=0.8,
    )
    ax.clabel(cs, inline=True, fontsize=7, fmt="C=%.3f")

    # ------------------------------------------------------------------
    # A few unforced trajectories
    # ------------------------------------------------------------------
    trajectories = [
        (
            "L4 tadpole libration",
            np.array([l4[0] - 0.03, l4[1] + 0.015, 0.0, 0.0]),
            40.0,
            "tab:orange",
        ),
        (
            "L5 tadpole libration",
            np.array([l5[0] - 0.03, l5[1] - 0.015, 0.0, 0.0]),
            40.0,
            "tab:green",
        ),
        (
            "Retrograde barycentric",
            np.array([-0.8, 0.0, 0.0, 1.5]),
            12.0,
            "tab:purple",
        ),
    ]

    for label, x0, tf, color in trajectories:
        _, y = propagate(cr3bp, x0, tf)
        ax.plot(y[0], y[1], color=color, lw=1.1, label=label, alpha=0.9)
        ax.plot(x0[0], x0[1], "o", color=color, ms=4)

    # ------------------------------------------------------------------
    # Primaries and triangular Lagrange points
    # ------------------------------------------------------------------
    ax.plot(*earth, "o", color="tab:blue", ms=13, label="Earth", zorder=5)
    ax.plot(*moon, "o", color="dimgray", ms=7, label="Moon", zorder=5)
    ax.plot(l4[0], l4[1], "^", color="tab:red", ms=9, label="L4", zorder=5)
    ax.plot(l5[0], l5[1], "v", color="tab:red", ms=9, label="L5", zorder=5)

    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.set_xlabel("x  (synodic, nondim)")
    ax.set_ylabel("y  (synodic, nondim)")
    ax.set_title(f"Planar Earth-Moon CR3BP  (mu = {mu:.6e})")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    out_path = Path(__file__).resolve().parent / "figures" / "cr3bp_overview.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved: {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
