"""Analyze the GMAT passive-safety V-bar standoff run.

Reads the relative-state ReportFile produced by the GMAT mission
``scripts/gmat/passive_safety_standoff.script`` (chaser expressed in the
target-centered LVLH frame: X = radial, Y = along-track / V-bar,
Z = cross-track) and verifies the passive-safety design:

* same semi-major axis  -> no secular along-track drift (V-bar standoff held),
* relative inclination di -> bounded cross-track oscillation of amplitude a*di,
* the chaser never approaches the target (minimum range >> 0).

Produces ``scripts/figures/passive_safety_standoff.png`` and prints the
numerical verification (cross-track amplitude vs analytic a*di, minimum
separation, residual J2 along-track drift).

Run from the project root::

    python -m scripts.analyze_passive_safety
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- physical constants / design parameters (must match the GMAT script) ---
MU_EARTH = 398600.4415  # km^3/s^2  (GMAT JGM-2 value)
A_KM = 7078.137  # target semi-major axis (700 km altitude)
DI_DEG = 0.01  # relative inclination
STANDOFF_KM = 5.0  # nominal V-bar standoff

# Default GMAT output location (override with a path argument).
DEFAULT_REPORT = Path(
    r"C:\Users\ginom\Downloads\gmat-win-R2025a\GMAT_R2025a\output\PassiveSafety.txt"
)
FIGURE_PATH = Path(__file__).resolve().parent / "figures" / "passive_safety_standoff.png"


def load_report(path: Path) -> dict[str, np.ndarray]:
    """Parse the whitespace-delimited GMAT ReportFile (one header line)."""
    raw = np.loadtxt(path, skiprows=1)
    return {
        "t": raw[:, 0],          # ElapsedSecs
        "x": raw[:, 1],          # radial  [km]
        "y": raw[:, 2],          # along-track / V-bar [km]
        "z": raw[:, 3],          # cross-track [km]
        "sma_t": raw[:, 4],      # target osculating SMA [km]
        "sma_c": raw[:, 5],      # chaser osculating SMA [km]
    }


def main() -> None:
    report = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_REPORT
    if not report.exists():
        raise SystemExit(f"GMAT report not found: {report}")

    d = load_report(report)
    t, x, y, z = d["t"], d["x"], d["y"], d["z"]
    rng = np.sqrt(x**2 + y**2 + z**2)

    period = 2.0 * np.pi * np.sqrt(A_KM**3 / MU_EARTH)
    n_orbits = t[-1] / period
    orbits = t / period

    di_rad = np.deg2rad(DI_DEG)
    a_di = A_KM * di_rad  # analytic cross-track amplitude [km]

    z_amp = np.max(np.abs(z))
    x_amp = np.max(np.abs(x))
    min_range = rng.min()
    max_range = rng.max()

    # Residual along-track (J2) drift: least-squares slope of Y(t). Over many
    # orbits the per-orbit libration averages out, leaving the secular rate.
    slope_km_s = np.polyfit(t, y, 1)[0]
    drift_per_orbit_m = slope_km_s * period * 1000.0
    drift_per_day_m = slope_km_s * 86400.0 * 1000.0

    # --------------------------------------------------------------------- #
    print("=" * 70)
    print("  PASSIVE-SAFETY V-BAR STANDOFF  -  GMAT J2 propagation")
    print("=" * 70)
    print(f"  Target orbit         : a = {A_KM:.3f} km  (alt {A_KM - 6378.137:.0f} km)")
    print(f"  Orbital period       : {period:.1f} s  ({period/60:.2f} min)")
    print(f"  Propagated           : {t[-1]:.0f} s  ({n_orbits:.2f} orbits)")
    print(f"  Relative inclination : di = {DI_DEG} deg")
    print("-" * 70)
    print("  CROSS-TRACK AMPLITUDE (the a*di check)")
    print(f"    analytic a*di          : {a_di*1000:.1f} m  ({a_di:.4f} km)")
    print(f"    GMAT max|Z| (J2)       : {z_amp*1000:.1f} m  ({z_amp:.4f} km)")
    print(f"    relative error         : {100*abs(z_amp-a_di)/a_di:.2f} %")
    print("-" * 70)
    print("  PASSIVE SAFETY")
    print(f"    along-track mean (Y)   : {y.mean():.3f} km  (V-bar standoff)")
    print(f"    radial amplitude max|X|: {x_amp*1000:.1f} m")
    print(f"    minimum range to target: {min_range:.3f} km")
    print(f"    maximum range to target: {max_range:.3f} km")
    print("-" * 70)
    print("  RESIDUAL J2 ALONG-TRACK DRIFT (from di; same-a nulls Keplerian drift)")
    print(f"    drift rate             : {drift_per_orbit_m:+.1f} m/orbit"
          f"  ({drift_per_day_m:+.0f} m/day)")
    sign = "away from" if drift_per_day_m < 0 else "toward"
    print(f"    direction              : {sign} the target (conservatively safe)"
          if drift_per_day_m < 0 else
          f"    direction              : {sign} the target")
    print(f"    SMA match (max|dSMA|)  : {np.max(np.abs(d['sma_t']-d['sma_c']))*1000:.2f} m"
          "  (osculating, oscillates together)")
    print("=" * 70)

    # --------------------------------------------------------------------- #
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Passive-safety V-bar standoff under J2  "
        f"(a={A_KM:.0f} km, di={DI_DEG} deg, ~{n_orbits:.1f} orbits)",
        fontsize=13, fontweight="bold",
    )

    # (a) RTN components vs time
    ax = axs[0, 0]
    ax.plot(orbits, y, label="along-track  Y (V-bar)", color="tab:blue")
    ax.plot(orbits, z, label="cross-track  Z", color="tab:green")
    ax.plot(orbits, x, label="radial  X", color="tab:red", lw=1)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("orbits"); ax.set_ylabel("relative position [km]")
    ax.set_title("(a) Relative position in target LVLH")
    ax.legend(loc="center left", fontsize=8); ax.grid(alpha=0.3)

    # (b) cross-track vs analytic envelope
    ax = axs[0, 1]
    ax.plot(orbits, z, color="tab:green", lw=1)
    ax.axhline(+a_di, color="k", ls="--", lw=1.2, label=f"+a*di = {a_di:.3f} km")
    ax.axhline(-a_di, color="k", ls="--", lw=1.2, label=f"-a*di = {-a_di:.3f} km")
    ax.set_xlabel("orbits"); ax.set_ylabel("cross-track Z [km]")
    ax.set_title("(b) Cross-track bounded by a*di")
    ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.3)

    # (c) range to target vs time
    ax = axs[1, 0]
    ax.plot(orbits, rng, color="tab:purple")
    ax.axhline(min_range, color="r", ls="--", lw=1,
               label=f"min range = {min_range:.2f} km")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("orbits"); ax.set_ylabel("range to target [km]")
    ax.set_title("(c) Separation never reaches the target")
    ax.legend(loc="lower left", fontsize=8); ax.grid(alpha=0.3)

    # (d) along-track / cross-track plane (the safety footprint)
    ax = axs[1, 1]
    sc = ax.scatter(y, z, c=orbits, cmap="viridis", s=4)
    ax.scatter([0], [0], marker="*", s=260, color="red",
               edgecolor="k", zorder=5, label="target")
    ax.set_xlabel("along-track Y [km]  (V-bar)")
    ax.set_ylabel("cross-track Z [km]")
    ax.set_title("(d) Y-Z plane: bounded loop, clear of target")
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.3)
    cbar = fig.colorbar(sc, ax=ax); cbar.set_label("orbits")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=130)
    print(f"  figure -> {FIGURE_PATH}")


if __name__ == "__main__":
    main()
