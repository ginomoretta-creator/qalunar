# Block B results — CONAE 12U mission re-flown in GMAT with binary schedules (2026-09-05)

All numbers are read from GMAT R2025a reports. Vehicle: 13 kg dry + 3.5 kg Xe,
40 mN Hall thruster, Isp 1200 s, 493.5 W operating power. Requirements:
thrust window ≤ 15 % of the period (RQ-MIS-03), orbit-averaged power < 15 W
(RQ-MIS-05), ≥ 8 m/s per apogee (RQ-MIS-02).

Reference ellipse: apogee radius 76,922 km (CONAE), injection perigee 250 km
altitude (the earlier a/e pair put the perigee 7 km below the surface),
i = 39°, injection at TA = 150°. Force model: Earth 4×4, Moon + Sun, MSISE-90
drag, SRP, DualCone eclipses.

## Phase 1 — perigee raise (`run_conae_duty_cycle.py`)

| quantity | value |
|---|---|
| per-slot perigee gain (24 equal-time slots, 16.5 kg) | 136.5 km at apogee, 5.4 km at perigee |
| cardinality QUBO, K = ⌊0.15·24⌋ = 3 | SA, penalty 273, coefficient range 5.5; slots at TA 176/178/180°, = top-K |
| chosen schedule flown back in GMAT | +407.4 km measured vs +401.8 km predicted (+1.4 %) |
| 3 slots = 12.5 % duty | 61.7 W orbit-averaged — RQ-MIS-05 violated by the 15 % window itself |
| five 2626 s apogee burns | perigee 250 → 752 km, **31.9 m/s** (6.37 m/s per pass), 44.6 g Xe, 4.34 d, 15.3 W |

At the flight (wet) mass the thruster-table closure does not hold: 6.4 m/s per
pass against RQ-MIS-02's 8, and 15.3 W against RQ-MIS-05's 15.

## Phase 2 — apogee raise to lunar distance, binary (`run_conae_phase2_binary.py`)

One QUBO per perigee pass: 15 % thrust window centred on perigee split into
10 slots, K = 2 slots from the power budget (3.00 % duty, 14.8 W), per-slot
apogee gain measured in GMAT, chosen slots flown back.

| quantity | binary (this work) | continuous spiral (earlier manuscript) |
|---|---|---|
| Δv | **470.4 m/s** | 2187 m/s |
| propellant | **0.645 kg** | 2.79 kg |
| transfer time | 77.6 d (36 passes) | 9.5 d |
| thrust time | 52.8 h | 228 h |
| duty / orbit-averaged power | 3.00 % / 14.8 W every pass | 100 % / 493.5 W |
| superposition error (measured vs predicted gain) | +2.2 % mean, 12.4 % on the last pass | — |
| eclipsed slots | present in 25 passes, never selected | — |

Deterministic on re-run (same final fuel to 1e-11 kg). Per-pass Cartesian
states in `scripts/figures/conae_phase2_binary.csv`.

## Phase 3 — encounter and capture (`run_conae_phase3_encounter.py`)

| quantity | value |
|---|---|
| wait for first SOI entry (no propellant) | 61.0 d |
| periselene | 41,170 km radius, 0.922 km/s, E = +0.306 km²/s² |
| Δv to bind at perilune | 434 m/s |
| Δv the 15 W budget allows over the 1.38-d SOI transit | 9.2 m/s (2.1 %) |
| energy to bind at full duty | 23.5 kWh |
| reference energy-QUBO capture at 100 % duty (16 slots, 2 d) | 14/16 slots, 389 m/s, 20.7 kWh, flown E +0.045, **not captured** after 50 d |

Direct capture from a duty-limited apogee-raising spiral is infeasible for this
vehicle by a factor ~50 (fast, distant flyby after the 61-day wait).

### Continuous-thrust reference (`run_conae_lunar_trajectory.py`)

The departure epoch is the phasing of the encounter. A 2-h scan over the two
encounter windows of the spiral (23–24 Jan, 5–7 Feb 2026) gives a minimum
periselene of ~31,600 km; 23 Jan 22:00 is the slowest arrival.

| quantity | value |
|---|---|
| arrival periselene | 31,618 km, E = −0.002 km²/s² (day 16.41) |
| capture | apse-targeted braking arcs: bind at periselene with ra < 40,000 km, then rp → 12,000 km at aposelene, ra → 20,000 km at periselene |
| braking | 362 m/s, 0.420 kg Xe, 34.3 h thrust-on, 16.9 kWh, over 9.69 d |
| vs 15 W budget over the same 9.69 d | 4.9× the allowed thrust time (fast flyby: ~50×) |
| final orbit | 9,100 × 19,900 km, E = −0.169 km²/s² |
| 50-day coast | 27 revolutions, 100 % inside the SOI, periselene 9,278 → 2,775 km (min altitude 1,038 km) |

Earlier attempts, all flown in GMAT and rejected: a fixed 25-h brake (the
previous version: E = −0.075, unbound again after 50 days); a single long
brake from periselene (E = −0.114 but the periselene collapses to impact);
apse arcs from the 24 Jan 12:00 arrival (periselene 45,400 km: the first
bound orbit reaches 195,000 km and is lost to the Earth's tide); tighter
targets (10,000 × 15,000 km and 8,000 × 12,000 km: the aposelene brake
overshoots the periselene, impact). The earlier "stable" figure came from the
uncorrected ellipse (sub-surface perigee) and a 12.7-day coast.

## SMART-1 (`run_smart1_capture.py`, `run_smart1_trajectory_figure.py`)

With E_target −0.08 over a 3-day window the energy QUBO fires all 16 slots and
reaches E = −0.026 km²/s² (a = 94,800 km) with the spacecraft 73,700 km from
the Moon — bound by two-body energy, outside the Hill sphere. The figure now
flies the same 3-day window (it flew 2 days before; the defaults of both
scripts were aligned with the cached schedule).

No single-passage schedule of this thruster captures SMART-1 durably: a
tangential thruster removes at most a_T·s of specific energy along a path s,
and at a_T ≈ 0.29 mm/s² one passage near the Moon is worth ~0.05 km²/s²,
against ~0.10 needed to bring the aposelene inside the Hill sphere. Continued
braking from SOI entry or from periselene for up to 30 days was flown in GMAT
and loses the spacecraft in every case. SMART-1 itself needed weeks of
resonant approach.

Eclipses are neglected in the SMART-1 reproduction because the cached encounter
phasing was found with continuous power; with DualCone the spiral timing shifts
and the encounter is lost (stated in the manuscript). The 654 × 35,885 km GTO
is approximate; the actual injection was 742 × 36,016 km (ESA).
