"""GMAT-backed truth oracle for binary thrust schedules.

Replaces the planar-CR3BP RK4 truth propagation inside
:func:`qalunar.qubo.thrust_scheduling.solve_iterative` with a headless
NASA GMAT run on real DE-series ephemerides. The QUBO is still built
from the cheap CR3BP/STM linearisation; only the *judge* changes: each
candidate schedule is flown in GMAT and the trust-region accept/reject
uses the flight-fidelity miss. The monotonicity guarantee of the
iterative driver (Proposition 1 of the paper) only requires a
consistent truth oracle, so it survives the swap unchanged.

Frame bridge
------------
qalunar states live in the barycentric planar synodic frame of the
ideal CR3BP (nondimensional; Earth at ``(-mu, 0)``, Moon at
``(1-mu, 0)``, unit rotation rate). GMAT states are exchanged through
an Earth-centred **ObjectReferenced Earth--Moon rotating frame**
(X from Earth to the real Moon, Z along the instantaneous orbit
normal) -- the standard frame for flying CR3BP designs in an ephemeris
model. The mapping uses the canonical Earth--Moon constants
(``L = 384,400 km``, ``V ~ 1.0232 km/s``):

* position:  ``r_km = (r_synodic + [mu, 0]) * L``
* velocity:  ``v_km_s = v_synodic * V``  (rotating-frame components)

and its exact inverse on output. The residual mismatch between the
ideal uniformly-rotating frame and the true (pulsating, elliptic)
Earth--Moon geometry is part of the fidelity gap being measured, not a
bridge error; anchor the epoch where the real Earth--Moon distance is
close to ``L`` (the default epoch has d ~ 385,600 km, +0.3%).

Control convention
------------------
Matches :func:`qalunar.qubo.thrust_scheduling.propagate_schedule`
exactly: at the start of each *burn* decision interval the
(anti)tangential unit vector is evaluated from the current
rotating-frame velocity and held constant in that frame for the whole
interval (an ``ElectricThruster`` whose ``CoordinateSystem`` is the
rotating frame, with ``ThrustDirection`` re-assigned per interval).
``DecrementMass = false`` so the thruster applies the same constant
acceleration as the CR3BP scheduler; consecutive coast intervals merge
into a single propagation.

GMAT version note
-----------------
On GMAT R2025a a ``Propagate ... {Sat.ElapsedSecs = X}`` stop condition
advances *by* X seconds relative to the start of that Propagate (it is
NOT cumulative from the epoch). The generated script relies on that
semantics and the parser cross-checks the reported total elapsed time,
so a regression on another GMAT version fails loudly instead of
silently flying the wrong arc.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from qalunar.dynamics.cr3bp import EARTH_MOON_MU
from qalunar.qubo.thrust_scheduling import (
    IterativeSchedulingResult,
    ThrustSchedulingConfig,
    ThrustSchedulingQubo,
)
from qalunar.reference.edelbaum import (
    ACCELERATION_M_S2,
    LENGTH_KM,
    TIME_S,
    VELOCITY_M_S,
)


__all__ = [
    "GmatOracleConfig",
    "build_gmat_linearized_qubo",
    "build_schedule_script",
    "epoch_plus_seconds",
    "find_gmat_console",
    "make_gmat_truth_propagator",
    "propagate_schedule_gmat",
    "rotating_km_to_synodic",
    "solve_gmat_iterative",
    "synodic_to_rotating_km",
]


V_KM_S: float = VELOCITY_M_S / 1000.0
"""Canonical synodic velocity unit in km/s (~1.0232)."""

_CONSOLE_NAME = "GmatConsole.exe" if os.name == "nt" else "GmatConsole"

# Searched in order after the environment variable and PATH. Kept free of
# any machine-specific path so a checkout works on someone else's computer.
_GMAT_SEARCH_ROOTS = (
    Path.home() / "Downloads" / "gmat-win-R2025a" / "GMAT_R2025a",
    Path.home() / "GMAT_R2025a",
    Path("C:/Program Files/GMAT/R2025a"),
    Path("/opt/GMAT/R2025a"),
    Path("/usr/local/GMAT/R2025a"),
)

_ERROR_PAT = re.compile(
    r"(\*\*\*\*\s*ERROR\s*\*\*\*\*|Interpreter Exception|"
    r"Could not read script|Execution Failed)",
    re.IGNORECASE,
)
_SUCCESS_PAT = re.compile(r"Mission run completed", re.IGNORECASE)


def find_gmat_console() -> Path:
    """Locate the GMAT console binary.

    Search order: the ``QALUNAR_GMAT_CONSOLE`` environment variable, then
    ``PATH``, then the conventional install roots of
    :data:`_GMAT_SEARCH_ROOTS`. The returned path is not guaranteed to
    exist; callers check ``.exists()`` and exit with instructions.
    """
    env = os.environ.get("QALUNAR_GMAT_CONSOLE")
    if env:
        return Path(env)
    found = shutil.which(_CONSOLE_NAME) or shutil.which("GmatConsole")
    if found:
        return Path(found)
    for root in _GMAT_SEARCH_ROOTS:
        candidate = root / "bin" / _CONSOLE_NAME
        if candidate.exists():
            return candidate
    # Nothing found: return the first candidate so the error message names
    # a plausible location and the env-var hint.
    return _GMAT_SEARCH_ROOTS[0] / "bin" / _CONSOLE_NAME


_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def epoch_plus_seconds(epoch_utc: str, seconds: float) -> str:
    """Advance a GMAT UTCGregorian epoch string by ``seconds``.

    Locale-independent (GMAT month abbreviations are fixed English).
    Sub-millisecond remainders are rounded to the millisecond GMAT
    accepts. Used by receding-horizon drivers to anchor each window's
    oracle at the correct absolute time.
    """
    from datetime import datetime, timedelta

    day_s, mon_s, year_s, hms = epoch_utc.strip().split()
    h, m, s = hms.split(":")
    sec, _, frac = s.partition(".")
    base = datetime(
        int(year_s), _MONTHS.index(mon_s) + 1, int(day_s),
        int(h), int(m), int(sec),
        int((frac or "0").ljust(6, "0")[:6]),
    )
    t = base + timedelta(seconds=float(seconds))
    ms = round(t.microsecond / 1000.0)
    if ms == 1000:
        t = t + timedelta(seconds=1)
        ms = 0
    return (f"{t.day:02d} {_MONTHS[t.month - 1]} {t.year} "
            f"{t.hour:02d}:{t.minute:02d}:{t.second:02d}.{ms:03d}")


@dataclass(frozen=True)
class GmatOracleConfig:
    """Configuration of the GMAT truth oracle.

    Parameters
    ----------
    gmat_console : Path or None
        Path to ``GmatConsole.exe``. ``None`` resolves via
        :func:`find_gmat_console`.
    epoch_utc : str
        UTCGregorian epoch mapped to qalunar ``t = t0``. The default,
        08 Jan 2026, has the real Earth--Moon distance within 0.3% of
        the canonical 384,400 km, so the frame bridge is anchored at
        near-mean lunar distance.
    point_masses : tuple of str
        Third bodies modelled as point masses. Earth is always the
        central body with a spherical-harmonic field (``gravity_degree``
        x ``gravity_order``) and is dropped from this list if present.
        ``("Luna",)`` is the ephemeris analogue of the CR3BP; the
        default ``("Luna", "Sun")`` is the four-body truth.
    gravity_degree, gravity_order : int
        Earth spherical-harmonic field truncation (JGM-2 in GMAT).
    srp : bool
        Solar radiation pressure on/off (spherical model, ``cr``,
        ``srp_area_m2``).
    shadow_model : str
        GMAT ``SolarPowerSystem.ShadowModel``: ``"DualCone"`` (default)
        cuts thrust in eclipse; ``"None"`` ignores eclipses.
    decrement_mass : bool
        Deplete propellant during burns (``True``, default). ``False``
        keeps the applied acceleration exactly constant, which is what
        the CR3BP linearisation assumes but not what a thruster does.
    isp_s, fuel_mass_kg : float
        Thruster specific impulse and loaded propellant.
    spacecraft_mass_kg : float
        Total spacecraft mass. Thrust force is sized as
        ``a * mass`` so the applied acceleration matches the
        scheduler's nondimensional ``thrust_magnitude``.
    accuracy, max_step_s : float
        RungeKutta89 adaptive-step settings.
    timeout_s : float
        Subprocess kill timeout per oracle call.
    mu : float
        CR3BP mass parameter used by the frame bridge.
    """

    gmat_console: Path | None = None
    epoch_utc: str = "08 Jan 2026 00:00:00.000"
    point_masses: tuple[str, ...] = ("Luna", "Sun")
    gravity_degree: int = 8
    gravity_order: int = 8
    srp: bool = True
    cr: float = 1.8
    srp_area_m2: float = 1.0
    shadow_model: str = "DualCone"
    decrement_mass: bool = True
    isp_s: float = 1640.0
    fuel_mass_kg: float = 5.0
    spacecraft_mass_kg: float = 100.0
    accuracy: float = 1e-12
    max_step_s: float = 1000.0
    timeout_s: float = 600.0
    mu: float = EARTH_MOON_MU

    def console(self) -> Path:
        return self.gmat_console if self.gmat_console is not None else find_gmat_console()


# ---------------------------------------------------------------------------
# Frame bridge
# ---------------------------------------------------------------------------


def synodic_to_rotating_km(
    state_synodic: NDArray[np.float64], mu: float = EARTH_MOON_MU,
) -> NDArray[np.float64]:
    """Barycentric synodic nondim state -> Earth-centred rotating km, km/s."""
    s = np.asarray(state_synodic, dtype=np.float64)
    return np.array([
        (s[0] + mu) * LENGTH_KM,
        s[1] * LENGTH_KM,
        s[2] * V_KM_S,
        s[3] * V_KM_S,
    ])


def rotating_km_to_synodic(
    state_rot_km: NDArray[np.float64], mu: float = EARTH_MOON_MU,
) -> NDArray[np.float64]:
    """Earth-centred rotating km, km/s -> barycentric synodic nondim."""
    s = np.asarray(state_rot_km, dtype=np.float64)
    return np.array([
        s[0] / LENGTH_KM - mu,
        s[1] / LENGTH_KM,
        s[2] / V_KM_S,
        s[3] / V_KM_S,
    ])


# ---------------------------------------------------------------------------
# Script generation
# ---------------------------------------------------------------------------


def _schedule_segments(schedule: NDArray[np.int64]) -> list[tuple[bool, int]]:
    """Compress a binary schedule into ``(is_burn, n_intervals)`` runs.

    Consecutive coast intervals merge into one segment; burn intervals
    stay one segment each because the thrust direction is re-evaluated
    at every burn interval start (the scheduler's convention).
    """
    segments: list[tuple[bool, int]] = []
    for q in np.asarray(schedule, dtype=np.int64):
        if q == 1:
            segments.append((True, 1))
        elif segments and not segments[-1][0]:
            segments[-1] = (False, segments[-1][1] + 1)
        else:
            segments.append((False, 1))
    return segments


def build_schedule_script(
    state0_synodic: NDArray[np.float64],
    t_span: tuple[float, float],
    schedule: NDArray[np.int64],
    sched_config: ThrustSchedulingConfig,
    oracle_config: GmatOracleConfig | None = None,
    report_name: str | None = None,
) -> str:
    """Generate the GMAT script that flies one binary schedule.

    The script is fully unrolled (no GMAT loops or conditionals), pure
    ASCII, one statement per line, and reports the final state in the
    Earth--Moon rotating frame.
    """
    cfg = oracle_config if oracle_config is not None else GmatOracleConfig()
    direction = sched_config.thrust_direction
    if sched_config.thrust_channels is not None:
        raise NotImplementedError(
            "GMAT oracle currently supports single-channel schedules only"
        )
    if direction not in ("tangential", "antitangential"):
        raise NotImplementedError(
            f"GMAT oracle supports tangential/antitangential, got {direction!r}"
        )
    sign = "" if direction == "tangential" else "-1 * "

    state_rot = synodic_to_rotating_km(state0_synodic, cfg.mu)
    t0, tf = t_span
    n_steps = int(np.asarray(schedule).size)
    dt_s = (tf - t0) * TIME_S / n_steps
    total_s = (tf - t0) * TIME_S

    thrust_n = (
        sched_config.thrust_magnitude * ACCELERATION_M_S2 * cfg.spacecraft_mass_kg
    )
    dry_mass = cfg.spacecraft_mass_kg - cfg.fuel_mass_kg
    report = report_name or f"qalunar_oracle_{uuid.uuid4().hex[:12]}.txt"
    # Earth is the primary body (harmonic field); it must not also be a
    # point mass or GMAT rejects the force model.
    points = ", ".join(b for b in cfg.point_masses if b != "Earth")
    srp_lines = [
        "FM.SRP = On;",
        "FM.SRP.Flux = 1367;",
        "FM.SRP.SRPModel = Spherical;",
        "FM.SRP.Nominal_Sun = 149597870.691;",
    ] if cfg.srp else []

    lines: list[str] = [
        "% Auto-generated by qalunar.highfidelity.gmat_oracle",
        f"% schedule = {''.join(str(int(q)) for q in schedule)}",
        f"% T = {tf - t0} nondim = {total_s:.6f} s, dt = {dt_s:.6f} s",
        "",
        "Create CoordinateSystem EarthMoonRot;",
        "EarthMoonRot.Origin = Earth;",
        "EarthMoonRot.Axes = ObjectReferenced;",
        "EarthMoonRot.XAxis = R;",
        "EarthMoonRot.ZAxis = N;",
        "EarthMoonRot.Primary = Earth;",
        "EarthMoonRot.Secondary = Luna;",
        "",
        "Create Spacecraft Sat;",
        "Sat.DateFormat = UTCGregorian;",
        f"Sat.Epoch = '{cfg.epoch_utc}';",
        "Sat.CoordinateSystem = EarthMoonRot;",
        "Sat.DisplayStateType = Cartesian;",
        f"Sat.X = {state_rot[0]:.12f};",
        f"Sat.Y = {state_rot[1]:.12f};",
        "Sat.Z = 0.0;",
        f"Sat.VX = {state_rot[2]:.15f};",
        f"Sat.VY = {state_rot[3]:.15f};",
        "Sat.VZ = 0.0;",
        f"Sat.DryMass = {dry_mass};",
        f"Sat.Cr = {cfg.cr:g};",
        f"Sat.SRPArea = {cfg.srp_area_m2:g};",
        "Sat.Tanks = {EPTank};",
        "Sat.Thrusters = {EPThruster};",
        "Sat.PowerSystem = EPSolar;",
        "",
        "Create ElectricTank EPTank;",
        "EPTank.AllowNegativeFuelMass = false;",
        f"EPTank.FuelMass = {cfg.fuel_mass_kg:g};",
        "",
        "Create ElectricThruster EPThruster;",
        "EPThruster.CoordinateSystem = EarthMoonRot;",
        "EPThruster.ThrustDirection1 = 1;",
        "EPThruster.ThrustDirection2 = 0;",
        "EPThruster.ThrustDirection3 = 0;",
        f"EPThruster.DecrementMass = {'true' if cfg.decrement_mass else 'false'};",
        "EPThruster.Tank = {EPTank};",
        "EPThruster.ThrustModel = ConstantThrustAndIsp;",
        f"EPThruster.ConstantThrust = {thrust_n:.12e};",
        f"EPThruster.Isp = {cfg.isp_s:g};",
        "EPThruster.MaximumUsablePower = 10;",
        "EPThruster.MinimumUsablePower = 0.001;",
        "",
        "Create SolarPowerSystem EPSolar;",
        "EPSolar.InitialMaxPower = 5;",
        "EPSolar.AnnualDecayRate = 0;",
        "EPSolar.Margin = 0;",
        f"EPSolar.ShadowModel = '{cfg.shadow_model}';",
        "",
        "Create FiniteBurn EPBurn;",
        "EPBurn.Thrusters = {EPThruster};",
        "",
        "Create ForceModel FM;",
        "FM.CentralBody = Earth;",
        "FM.PrimaryBodies = {Earth};",
        f"FM.GravityField.Earth.Degree = {cfg.gravity_degree};",
        f"FM.GravityField.Earth.Order = {cfg.gravity_order};",
        "FM.PointMasses = {" + points + "};",
        *srp_lines,
        "",
        "Create Propagator Prop;",
        "Prop.FM = FM;",
        "Prop.Type = RungeKutta89;",
        "Prop.InitialStepSize = 60;",
        f"Prop.Accuracy = {cfg.accuracy:g};",
        "Prop.MinStep = 0;",
        f"Prop.MaxStep = {cfg.max_step_s:g};",
        "",
        "Create ReportFile Rep;",
        f"Rep.Filename = '{report}';",
        "Rep.Precision = 16;",
        "Rep.WriteHeaders = false;",
        "",
        "Create Variable vx vy vn;",
        "",
        "BeginMissionSequence;",
        "",
    ]

    # R2025a semantics: each Propagate {ElapsedSecs = X} advances BY X.
    for is_burn, n_int in _schedule_segments(np.asarray(schedule, dtype=np.int64)):
        seg_s = n_int * dt_s
        if is_burn:
            lines += [
                f"vx = {sign}Sat.EarthMoonRot.VX;",
                f"vy = {sign}Sat.EarthMoonRot.VY;",
                "vn = sqrt( vx*vx + vy*vy );",
                "Sat.EPThruster.ThrustDirection1 = vx / vn;",
                "Sat.EPThruster.ThrustDirection2 = vy / vn;",
                "Sat.EPThruster.ThrustDirection3 = 0;",
                "BeginFiniteBurn EPBurn(Sat);",
                f"Propagate Prop(Sat) {{Sat.ElapsedSecs = {seg_s:.9f}}};",
                "EndFiniteBurn EPBurn(Sat);",
                "",
            ]
        else:
            lines += [
                f"Propagate Prop(Sat) {{Sat.ElapsedSecs = {seg_s:.9f}}};",
                "",
            ]

    lines += [
        "Report Rep Sat.ElapsedSecs Sat.EarthMoonRot.X Sat.EarthMoonRot.Y "
        "Sat.EarthMoonRot.Z Sat.EarthMoonRot.VX Sat.EarthMoonRot.VY "
        "Sat.EarthMoonRot.VZ Sat.EPTank.FuelMass;",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Headless execution
# ---------------------------------------------------------------------------


def _run_script(script_text: str, report_name: str, cfg: GmatOracleConfig) -> str:
    """Run a script through GmatConsole and return the report text."""
    console = cfg.console()
    if not console.exists():
        raise FileNotFoundError(
            f"GmatConsole not found at {console}; set QALUNAR_GMAT_CONSOLE "
            "or GmatOracleConfig.gmat_console"
        )
    bin_dir = console.parent

    with tempfile.NamedTemporaryFile(
        "w", suffix=".script", delete=False, encoding="ascii"
    ) as fh:
        fh.write(script_text)
        script_path = Path(fh.name)

    # A private log file per run: concurrent GmatConsole instances that share
    # the default GmatLog.txt fail at start-up ("specified log file is not a
    # valid log file"), which made the parallel finite-difference runs flaky.
    log_path = script_path.with_suffix(".log")
    try:
        proc = subprocess.run(
            [str(console), "-l", str(log_path), "-r", str(script_path)],
            cwd=str(bin_dir),
            capture_output=True,
            text=True,
            timeout=cfg.timeout_s,
        )
        raw = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode != 0:
            tail = "\n".join(raw.splitlines()[-15:])
            raise RuntimeError(
                f"GmatConsole exited with code {proc.returncode}"
                f"\n--- gmat output tail ---\n{tail}"
            )
        error_lines = [
            ln.strip() for ln in raw.splitlines() if _ERROR_PAT.search(ln)
        ]
        error_lines = [
            ln for ln in error_lines
            if "did not open" not in ln and "Library" not in ln
        ]
        if error_lines or not _SUCCESS_PAT.search(raw):
            tail = "\n".join(raw.splitlines()[-15:])
            raise RuntimeError(
                "GMAT oracle run failed: "
                + ("; ".join(error_lines[:5]) or "no success marker")
                + f"\n--- gmat output tail ---\n{tail}"
            )

        for d in (bin_dir.parent / "output", bin_dir / "output", bin_dir):
            p = d / report_name
            if p.exists():
                text = p.read_text(errors="replace")
                try:
                    p.unlink()
                except OSError:
                    pass
                return text
        raise RuntimeError(
            f"GMAT run completed but report {report_name!r} was not found"
        )
    finally:
        for p in (script_path, log_path):
            try:
                p.unlink()
            except OSError:
                pass


def propagate_schedule_gmat(
    state0_synodic: NDArray[np.float64],
    t_span: tuple[float, float],
    schedule: NDArray[np.int64],
    sched_config: ThrustSchedulingConfig,
    oracle_config: GmatOracleConfig | None = None,
    out_of_plane_log: list[dict[str, float]] | None = None,
) -> NDArray[np.float64]:
    """Fly a binary schedule in GMAT and return the final synodic state.

    Drop-in truth analogue of
    :func:`qalunar.qubo.thrust_scheduling.propagate_schedule` (final
    state only). The out-of-plane components that real dynamics
    introduce are projected out (planar bridge) but never silently: pass
    ``out_of_plane_log`` to collect ``{"z_km", "vz_km_s", "elapsed_s"}``
    per call, so the size of the projected miss can be reported.
    """
    cfg = oracle_config if oracle_config is not None else GmatOracleConfig()
    report_name = f"qalunar_oracle_{uuid.uuid4().hex[:12]}.txt"
    script = build_schedule_script(
        state0_synodic, t_span, schedule, sched_config, cfg, report_name,
    )
    text = _run_script(script, report_name, cfg)

    last = text.strip().splitlines()[-1].split()
    if len(last) < 7:
        raise RuntimeError(f"unexpected GMAT report line: {last!r}")
    elapsed, x, y, z_km, vx, vy, vz_km_s = (float(v) for v in last[:7])
    if out_of_plane_log is not None:
        out_of_plane_log.append(
            {"z_km": z_km, "vz_km_s": vz_km_s, "elapsed_s": elapsed}
        )

    total_s = (t_span[1] - t_span[0]) * TIME_S
    if abs(elapsed - total_s) > 1e-3 * total_s:
        raise RuntimeError(
            f"GMAT elapsed time {elapsed:.3f}s != expected {total_s:.3f}s; "
            "Propagate stop-condition semantics may have changed in this "
            "GMAT version"
        )

    return rotating_km_to_synodic(np.array([x, y, vx, vy]), cfg.mu)


def make_gmat_truth_propagator(
    oracle_config: GmatOracleConfig | None = None,
    out_of_plane_log: list[dict[str, float]] | None = None,
) -> Callable[
    [NDArray[np.float64], tuple[float, float], NDArray[np.int64], ThrustSchedulingConfig],
    NDArray[np.float64],
]:
    """Build the ``truth_propagator`` callable for ``solve_iterative``."""
    cfg = oracle_config if oracle_config is not None else GmatOracleConfig()

    def _truth(
        state0: NDArray[np.float64],
        t_span: tuple[float, float],
        schedule: NDArray[np.int64],
        sched_config: ThrustSchedulingConfig,
    ) -> NDArray[np.float64]:
        return propagate_schedule_gmat(
            state0, t_span, schedule, sched_config, cfg,
            out_of_plane_log=out_of_plane_log,
        )

    return _truth


# ---------------------------------------------------------------------------
# Fully GMAT-linearised QUBO: finite-difference impulse responses
# ---------------------------------------------------------------------------
#
# Anchoring the effective gap at the oracle's nominal endpoint (the
# ``nominal_final_override`` hook of ``build_thrust_scheduling_qubo``)
# corrects *where* the QUBO thinks it is, but the CR3BP impulse-response
# vectors still describe *how burns act* along the CR3BP nominal. On a
# sensitive arc (a lunar flyby) the ephemeris trajectory diverges so far
# from the CR3BP one that those sensitivities point the wrong way and
# the trust region rejects every candidate. The remedy is to measure the
# impulse responses in the truth model itself: one single-bit-flip GMAT
# run per decision slot,
#
#     b_j = +/- [ x_f(q_nom with bit j flipped) - x_f(q_nom) ],
#
# (sign chosen so the linear model reads x_f ~ x_nom + sum (q_j -
# q_nom_j) b_j). The QUBO is then assembled exclusively from
# high-fidelity data -- exact for single flips, first-order for
# combinations -- and the annealer keeps doing only the combinatorial
# part. Cost: N oracle runs per outer iteration, parallelised across
# GmatConsole subprocesses.


def _run_schedules_parallel(
    state0_synodic: NDArray[np.float64],
    t_span: tuple[float, float],
    schedules: list[NDArray[np.int64]],
    sched_config: ThrustSchedulingConfig,
    oracle_config: GmatOracleConfig,
    max_workers: int = 4,
) -> list[NDArray[np.float64]]:
    """Fly several schedules through GMAT concurrently (one subprocess each)."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(
                propagate_schedule_gmat,
                state0_synodic, t_span, qq, sched_config, oracle_config,
            )
            for qq in schedules
        ]
        return [f.result() for f in futures]


def build_gmat_linearized_qubo(
    state0_synodic: NDArray[np.float64],
    target_state: NDArray[np.float64],
    t_span: tuple[float, float],
    nominal_schedule: NDArray[np.int64],
    nominal_final: NDArray[np.float64],
    sched_config: ThrustSchedulingConfig,
    oracle_config: GmatOracleConfig | None = None,
    max_workers: int = 4,
) -> ThrustSchedulingQubo:
    """Assemble the scheduling QUBO from finite-difference GMAT responses.

    Parameters
    ----------
    nominal_final : (4,) ndarray
        The oracle's final state for ``nominal_schedule`` (the caller
        always has it already; passing it avoids a redundant GMAT run).

    Returns
    -------
    ThrustSchedulingQubo
        Same container the samplers consume; ``coast_trajectory`` is a
        single-row placeholder (no dense trajectory is propagated).
    """
    cfg = oracle_config if oracle_config is not None else GmatOracleConfig()
    q_nom = np.asarray(nominal_schedule, dtype=np.int64)
    target = np.asarray(target_state, dtype=np.float64)
    x_nom = np.asarray(nominal_final, dtype=np.float64)
    N = q_nom.size
    t0, tf = t_span
    dt = (tf - t0) / N

    flips = []
    for j in range(N):
        qq = q_nom.copy()
        qq[j] = 1 - qq[j]
        flips.append(qq)
    flip_finals = _run_schedules_parallel(
        state0_synodic, t_span, flips, sched_config, cfg, max_workers,
    )

    b_vectors = np.empty((N, 4))
    for j in range(N):
        delta = flip_finals[j] - x_nom
        b_vectors[j] = delta if q_nom[j] == 0 else -delta

    d_eff = (target - x_nom) + b_vectors.T @ q_nom.astype(np.float64)

    if sched_config.target_weights is not None:
        W = np.diag(np.asarray(sched_config.target_weights, dtype=np.float64))
    else:
        W = np.eye(4)
    b_W = (W @ b_vectors.T).T
    Q = b_vectors @ b_W.T
    Wd = W @ d_eff
    linear = sched_config.fuel_weight * np.ones(N) - 2.0 * b_vectors @ Wd
    constant = float(d_eff @ Wd)

    return ThrustSchedulingQubo(
        Q=Q,
        linear=linear,
        constant=constant,
        n_steps=N,
        n_channels=1,
        channel_labels=(sched_config.thrust_direction,),
        b_vectors=b_vectors,
        target_gap=d_eff,
        fuel_weight=sched_config.fuel_weight,
        t_grid=t0 + dt * np.arange(N),
        coast_trajectory=np.asarray(state0_synodic, dtype=np.float64)[None, :],
        thrust_magnitude=sched_config.thrust_magnitude,
        dt_decision=dt,
    )


def solve_gmat_iterative(
    state0_synodic: NDArray[np.float64],
    target_state: NDArray[np.float64],
    t_span: tuple[float, float],
    n_decision_steps: int,
    sampler: "Callable[[ThrustSchedulingQubo], NDArray[np.int64]]",
    sched_config: ThrustSchedulingConfig,
    oracle_config: GmatOracleConfig | None = None,
    max_iters: int = 6,
    tol: float = 1e-9,
    initial_schedule: NDArray[np.int64] | None = None,
    max_workers: int = 4,
    verbose: bool = False,
) -> IterativeSchedulingResult:
    """Iterative binary scheduling fully linearised in GMAT.

    The GMAT analogue of
    :func:`qalunar.qubo.thrust_scheduling.solve_iterative`: at each
    outer iteration the QUBO is rebuilt from finite-difference GMAT
    impulse responses around the incumbent schedule
    (:func:`build_gmat_linearized_qubo`), the sampler proposes a
    candidate, the candidate is flown in GMAT, and the trust-region
    rule accepts it only if the true (ephemeris) miss improves. The
    monotone-acceptance guarantee therefore holds with respect to the
    high-fidelity model. Cost per iteration: ``N + 1`` GMAT runs.
    """
    cfg = oracle_config if oracle_config is not None else GmatOracleConfig()
    M = n_decision_steps
    if sched_config.thrust_channels is not None:
        raise NotImplementedError("single-channel schedules only")

    if initial_schedule is None:
        q = np.zeros(M, dtype=np.int64)
    else:
        q = np.asarray(initial_schedule, dtype=np.int64).copy()
        if q.shape != (M,):
            raise ValueError(f"initial_schedule shape {q.shape} != ({M},)")

    target = np.asarray(target_state, dtype=np.float64)
    if sched_config.target_weights is not None:
        _w_sqrt = np.sqrt(np.asarray(sched_config.target_weights, dtype=np.float64))
    else:
        _w_sqrt = np.ones(4)

    def _norm(miss: NDArray[np.float64]) -> float:
        return float(np.linalg.norm(_w_sqrt * miss))

    x_nom = propagate_schedule_gmat(state0_synodic, t_span, q, sched_config, cfg)
    incumbent_miss = x_nom - target
    incumbent_norm = _norm(incumbent_miss)

    schedule_history = [q.copy()]
    miss_history = [incumbent_norm]
    converged = False
    reason = "max_iters reached"
    last_qubo: ThrustSchedulingQubo | None = None

    for it in range(max_iters):
        qubo = build_gmat_linearized_qubo(
            state0_synodic, target, t_span, q, x_nom,
            sched_config, cfg, max_workers,
        )
        last_qubo = qubo
        q_cand = np.asarray(sampler(qubo), dtype=np.int64)
        if np.array_equal(q_cand, q):
            converged = True
            reason = f"schedule fixed point at iter {it}"
            break

        x_cand = propagate_schedule_gmat(
            state0_synodic, t_span, q_cand, sched_config, cfg,
        )
        cand_norm = _norm(x_cand - target)
        if verbose:
            print(f"  gmat-iter {it}: incumbent ||miss||={incumbent_norm:.3e} "
                  f"-> candidate ||miss||={cand_norm:.3e}")

        if cand_norm > incumbent_norm:
            converged = True
            reason = (f"non-improving candidate at iter {it} "
                      f"(cand={cand_norm:.3e} > incumbent={incumbent_norm:.3e})")
            break

        improvement = incumbent_norm - cand_norm
        q = q_cand
        x_nom = x_cand
        incumbent_miss = x_cand - target
        incumbent_norm = cand_norm
        schedule_history.append(q.copy())
        miss_history.append(cand_norm)

        if abs(improvement) < tol:
            converged = True
            reason = f"||miss|| improvement {improvement:.3e} < tol at iter {it}"
            break
    else:
        last_qubo = build_gmat_linearized_qubo(
            state0_synodic, target, t_span, q, x_nom,
            sched_config, cfg, max_workers,
        )

    assert last_qubo is not None
    return IterativeSchedulingResult(
        schedule=q,
        final_qubo=last_qubo,
        true_final_state=incumbent_miss + target,
        true_miss=incumbent_miss,
        iterations=len(schedule_history) - 1,
        schedule_history=schedule_history,
        true_miss_norm_history=miss_history,
        converged=converged,
        converged_reason=reason,
    )
