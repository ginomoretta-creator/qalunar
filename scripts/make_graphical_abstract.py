"""Graphical abstract for the Metascience in Aerospace submission.

One scene, no caption: the 36 Phase-2 ellipses flown in GMAT growing from the
transfer ellipse to the Moon, with the two perigee slots the QUBO fires on
every pass drawn in orange, and the binary string of one pass as a row of
qubits. Elements per pass come from scripts/figures/conae_phase2_binary.csv.

Run:  python -m scripts.make_graphical_abstract
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyBboxPatch
import numpy as np

FIG_DIR = Path(__file__).resolve().parent / "figures"
OUT = FIG_DIR / "graphical_abstract.png"
EARTH_R, MOON_R, MOON_DIST = 6_378.137, 1_737.4, 384_400.0
WINDOW, N_SLOTS = 0.15, 10           # 15 % of the period, ten equal-time slots
BG, C_TEXT, C_DIM = "#070d1a", "#e8edf5", "#8a94a8"
C_FIRE, C_COAST = "#ff7a2f", "#3b4a63"

plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix"})


def kepler_nu(M: np.ndarray, e: float) -> np.ndarray:
    E = M.copy()
    for _ in range(40):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    return 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))


def ellipse(a: float, e: float, nu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = a * (1 - e**2) / (1 + e * np.cos(nu))
    return -r * np.cos(nu) * 1e-3, r * np.sin(nu) * 1e-3      # 10^3 km, apogee to +x


def glow_disk(ax, x, y, r, color, halo=6.0, n=6, zorder=8):
    for k in range(n, 0, -1):
        ax.add_patch(Circle((x, y), r * (1 + halo * k / n), color=color, alpha=0.035, lw=0,
                            zorder=zorder - 1))
    ax.add_patch(Circle((x, y), r, color=color, lw=0, zorder=zorder))


def main() -> None:
    rows = list(csv.DictReader((FIG_DIR / "conae_phase2_binary.csv").open(encoding="utf-8")))
    summ = json.loads((FIG_DIR / "conae_phase2_binary_summary.json").read_text(encoding="utf-8"))
    passes = [(float(r["a_km"]), float(r["e"]), r["schedule"]) for r in rows]

    fig, ax = plt.subplots(figsize=(13.0, 6.5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(-70, 430)
    ax.set_ylim(-125, 125)
    ax.set_aspect("equal")
    ax.axis("off")

    # faint star field
    rng = np.random.default_rng(7)
    sx, sy = rng.uniform(-70, 430, 260), rng.uniform(-125, 125, 260)
    ax.scatter(sx, sy, s=rng.uniform(0.2, 2.2, 260), color="white", alpha=0.35, lw=0, zorder=1)

    # lunar orbit and the Moon at the arrival apogee direction
    ax.add_patch(Circle((0, 0), MOON_DIST * 1e-3, fill=False, ec="#2a3550", ls=(0, (5, 6)),
                        lw=1.0, zorder=2))
    glow_disk(ax, MOON_DIST * 1e-3, 0, MOON_R * 1e-3 * 4.5, "#c9ced8", halo=3.5)

    # the 36 ellipses: dim teal at the start, bright at lunar distance
    cmap = LinearSegmentedColormap.from_list("climb", ["#1a5f6e", "#2fb7c9", "#c7f3ff"])
    nu = np.linspace(-np.pi, np.pi, 900)
    n = len(passes)
    for k, (a, e, _) in enumerate(passes):
        x, y = ellipse(a, e, nu)
        c = cmap(k / (n - 1))
        ax.plot(x, y, color=c, lw=0.9, alpha=0.55 + 0.4 * k / (n - 1), zorder=3)

    # the fired slots of every pass, in orange, with a soft glow
    M_edges = np.linspace(-WINDOW * np.pi, WINDOW * np.pi, N_SLOTS + 1)
    for a, e, sched in passes:
        for j in range(N_SLOTS):
            if sched[j] != "1":
                continue
            Ms = np.linspace(M_edges[j], M_edges[j + 1], 40)
            x, y = ellipse(a, e, kepler_nu(Ms, e))
            ax.plot(x, y, color=C_FIRE, lw=7, alpha=0.06, solid_capstyle="round", zorder=4)
            ax.plot(x, y, color=C_FIRE, lw=2.2, alpha=0.9, solid_capstyle="round", zorder=5)

    glow_disk(ax, 0, 0, EARTH_R * 1e-3, "#3d8bff", halo=5.0)

    # the decision, drawn as qubits: the schedule of one pass, ten cells, two lit
    sched = passes[0][2]
    cx, cy, w, gap = 0.0, -74.0, 9.0, 2.6
    x0 = cx - (N_SLOTS * w + (N_SLOTS - 1) * gap) / 2
    for j in range(N_SLOTS):
        on = sched[j] == "1"
        ax.add_patch(FancyBboxPatch((x0 + j * (w + gap), cy - w / 2), w, w,
                                    boxstyle="round,pad=0,rounding_size=1.6",
                                    fc=C_FIRE if on else BG, ec=C_FIRE if on else C_COAST,
                                    lw=1.2, zorder=6))
        if on:
            ax.add_patch(FancyBboxPatch((x0 + j * (w + gap) - 2, cy - w / 2 - 2), w + 4, w + 4,
                                        boxstyle="round,pad=0,rounding_size=2.5",
                                        fc=C_FIRE, ec="none", alpha=0.18, zorder=5))
    ax.annotate("", xy=(2.5, -8), xytext=(cx, cy + w / 2 + 3),
                arrowprops=dict(arrowstyle="-", color=C_DIM, lw=0.8, alpha=0.8,
                                connectionstyle="arc3,rad=0.25"), zorder=6)
    ax.text(cx, cy - w / 2 - 6, "one qubit per slot  ·  fire two", color=C_DIM,
            fontsize=10.5, ha="center", va="top", zorder=6)

    # text: a title and one line of numbers
    ax.text(-62, 108, "Which arc to thrust?", color=C_TEXT, fontsize=27, weight="bold",
            ha="left", va="top", zorder=6)
    ax.text(-62, 86, "Binary thrust scheduling for quantum annealing, closed over GMAT",
            color=C_DIM, fontsize=12.5, ha="left", va="top", zorder=6)
    ax.text(422, -92,
            f"{summ['passes']} perigee passes · {summ['dv_total_m_s']:.0f} m/s · "
            f"{summ['fuel_used_kg']:.2f} kg Xe · 3 % duty · 15 W",
            color=C_TEXT, fontsize=11.5, ha="right", va="top", zorder=6)
    ax.text(422, -106, "12U CubeSat, 40 mN Hall thruster, flown in GMAT",
            color=C_DIM, fontsize=10.5, ha="right", va="top", zorder=6)

    fig.savefig(OUT, dpi=300, facecolor=BG)
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
