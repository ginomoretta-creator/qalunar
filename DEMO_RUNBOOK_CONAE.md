# Live Demo Runbook — GMAT MCP for Pablo Servidia (CONAE)

**Goal.** Show Pablo, in ~10 minutes, that an AI (Claude) connected to NASA GMAT
through an MCP can *design and run missions conversationally* — starting from
the very CubeSat apogee-raising he helped plan, and extending it to a full
lunar mission. The wow is the **closed loop**: type in natural language → AI
writes the GMAT script → runs it on the real high-fidelity model → reads the
result → decides the next step.

---

## Before the meeting — setup checklist

- [ ] GMAT R2025a installed; GmatConsole reachable
      (`<GMAT_R2025a>/bin/GmatConsole.exe`, or set `QALUNAR_GMAT_CONSOLE`).
- [ ] GMAT MCP server running and connected to Claude. Confirm the `gmat` tools
      are available (`runGmat`, `searchDocs`, `getGmatIdioms`, `getGmatSample`,
      `listGmatSamples`).
- [ ] **Dry-run Act 1 the day before** to confirm the loop returns cleanly.
- [ ] Paper figures open in tabs for Act 3–4:
      `conae_lunar_trajectory.png`, `conae_duty_cycle.png`, and Table 4
      (mission breakdown) from the paper.
- [ ] Fallback scripts (bottom of this file) ready to paste if live generation
      stumbles.

---

## The arc (≈10 min)

### Act 0 — framing (say this)
> "This is Claude connected to NASA GMAT through an MCP I built. I describe a
> manoeuvre in plain language; it writes the GMAT script, runs it on the real
> model — full ephemerides, gravity field, finite burns — reads the result, and
> decides what to do next. Let me start from the CubeSat orbit you helped me
> plan."

### Act 1 — warm-up: propagate the CubeSat orbit *(≈15 s, always works)* ✓ validated
Type:
> "I have a CubeSat in a highly eccentric orbit: SMA 41646 km, ECC 0.847020,
> INC 39°, RAAN 0, AOP 0, TA 0. Propagate it to apoapsis in GMAT (Earth 4×4
> gravity, Moon and Sun as point masses) and report the apogee and perigee
> radii."

Expected: **apogee ≈ 76,921 km, perigee ≈ 6,371 km**. Then point out:
> "The perigee radius is 6,371 km — *below* the Earth's surface. This is the
> disposal ellipse; left alone it re-enters. The first job is to raise the
> perigee."

### Act 2 — CONAE's problem: apogee-raising to lift the perigee *(≈30 s)* ✓ pattern validated
Type:
> "Add a 40 mN Hall thruster (Isp 1200 s), dry mass 13 kg, a small xenon tank.
> Propagate to apogee, fire the thruster prograde for 2626 s, and report the
> new perigee and the ΔV. Which apse should I burn at to raise the perigee most
> efficiently, and why?"

Expected: **perigee rises ~120 km, ΔV ≈ 8 m/s**, and the AI explains that a
prograde burn at apogee raises the *opposite* apse (the perigee). Talking point:
> "This is the apogee-raising you helped me plan. And note — the optimizer picks
> apogee as the right place to burn with no orbital mechanics built in. That
> 'which arc to fire' decision is what we turned into an optimization problem."

### Act 3 — the full cislunar mission *(show, don't wait)*
Do **not** run the 20-day spiral live. Say:
> "We then let the same loop take it all the way to the Moon — raise the apogee
> to lunar distance, then brake at the encounter. Here's the trajectory it
> produced."

Show **`conae_lunar_trajectory.png`** (spiral out + capture hook). Narrate:
> "Same 40 mN thruster, same closed loop. The 12U reaches the Moon and captures
> into a *bound* lunar orbit — Moon-relative energy −0.088 km²/s², all validated
> in GMAT. About 2,500 m/s and 3.15 kg of xenon."

Show **Table 4** (phase-by-phase ΔV / time / xenon) if Pablo wants the numbers.

### Act 4 — the closed loop and the QUBO *(conceptual)*
> "The 'which slots to fire' decision is posed as a QUBO — a binary, quantum-
> *ready* optimization, solved classically here, but the exact form a quantum
> annealer would ingest. The loop never simulates the 2^N firing combinations:
> N runs to build the model, one to validate the choice — linear, not
> exponential."

Show **`conae_duty_cycle.png`** (the optimizer concentrating the burns at apogee).

### Close
> "So: an AI in the loop with a high-fidelity simulator, designing real
> manoeuvres and validating every step. I'd love to write this up for
> Metascience and keep developing it."

---

## Fallback scripts (paste into `runGmat` if live generation stumbles)

**Note:** via the MCP the script is passed as *content*, so you do not deal with
file paths. (The only gotcha if you ever run a file directly: GMAT is a native
Windows exe — give it a Windows path, not a `/tmp/...` one.)

### Act 1 — propagate to apogee
```matlab
Create Spacecraft Cube;
Cube.DateFormat = UTCGregorian;
Cube.Epoch = '01 Jan 2026 00:00:00.000';
Cube.CoordinateSystem = EarthMJ2000Eq;
Cube.DisplayStateType = Keplerian;
Cube.SMA = 41646;  Cube.ECC = 0.847020121;  Cube.INC = 39;
Cube.RAAN = 0;  Cube.AOP = 0;  Cube.TA = 0;
Create ForceModel FM;
FM.CentralBody = Earth;
FM.PrimaryBodies = {Earth};
FM.GravityField.Earth.Degree = 4;  FM.GravityField.Earth.Order = 4;
FM.PointMasses = {Luna, Sun};
Create Propagator Prop;  Prop.FM = FM;  Prop.Type = RungeKutta89;  Prop.MaxStep = 600;
Create ReportFile Rep;  Rep.Filename = 'demo.txt';  Rep.WriteHeaders = false;
BeginMissionSequence;
Report Rep Cube.Earth.RadApo Cube.Earth.RadPer;
Propagate Prop(Cube) {Cube.Earth.Apoapsis};
Report Rep Cube.Earth.RadApo Cube.Earth.RadPer;
```

### Act 2 — apogee burn raises the perigee
```matlab
Create Spacecraft Cube;
Cube.DateFormat = UTCGregorian;
Cube.Epoch = '01 Jan 2026 00:00:00.000';
Cube.CoordinateSystem = EarthMJ2000Eq;
Cube.DisplayStateType = Keplerian;
Cube.SMA = 41646;  Cube.ECC = 0.847020121;  Cube.INC = 39;
Cube.RAAN = 0;  Cube.AOP = 0;  Cube.TA = 0;
Cube.DryMass = 13;
Cube.Tanks = {Xe};  Cube.Thrusters = {Hall};  Cube.PowerSystem = SolarP;
Create ElectricTank Xe;  Xe.AllowNegativeFuelMass = false;  Xe.FuelMass = 1.0;
Create ElectricThruster Hall;
Hall.CoordinateSystem = Local;  Hall.Origin = Earth;  Hall.Axes = VNB;
Hall.ThrustDirection1 = 1;  Hall.ThrustDirection2 = 0;  Hall.ThrustDirection3 = 0;
Hall.DecrementMass = true;  Hall.Tank = {Xe};
Hall.ThrustModel = ConstantThrustAndIsp;  Hall.ConstantThrust = 0.040;  Hall.Isp = 1200;
Hall.MaximumUsablePower = 2;  Hall.MinimumUsablePower = 0.01;
Create SolarPowerSystem SolarP;  SolarP.InitialMaxPower = 2;  SolarP.ShadowModel = 'None';
Create FiniteBurn Burn;  Burn.Thrusters = {Hall};
Create ForceModel FM;
FM.CentralBody = Earth;  FM.PrimaryBodies = {Earth};
FM.GravityField.Earth.Degree = 4;  FM.GravityField.Earth.Order = 4;
FM.PointMasses = {Luna, Sun};
Create Propagator Prop;  Prop.FM = FM;  Prop.Type = RungeKutta89;  Prop.MaxStep = 300;
Create ReportFile Rep;  Rep.Filename = 'demo.txt';  Rep.WriteHeaders = false;
BeginMissionSequence;
Propagate Prop(Cube) {Cube.Earth.Apoapsis};
Report Rep Cube.Earth.RadPer Cube.XeTank.FuelMass;
BeginFiniteBurn Burn(Cube);
Propagate Prop(Cube) {Cube.ElapsedSecs = 2626};
EndFiniteBurn Burn(Cube);
Report Rep Cube.Earth.RadPer Cube.XeTank.FuelMass;
```

---

## Risk notes / gotchas

- **Keep live burns short** (Act 2 = 2626 s → GMAT returns in seconds). Never run
  the full 20-day spiral live; show the pre-rendered figure instead.
- `ElapsedSecs` stop conditions are **relative** in R2025a (each `Propagate`
  advances *by* the value, not to an absolute epoch).
- Valid central-body params need no coordinate system: `Sat.Earth.RadApo`,
  `.RadPer`, `.SMA`, `.TA`, `.Energy`. `VMAG` *does* need a CS.
- If the AI emits a script that errors, paste the matching fallback above.

## Honesty (non-negotiable, even in the demo)
- Say **"QUBO / quantum-ready, solved classically"** — never "quantum" alone (no
  QPU has run).
- The lunar spiral in the figure uses *continuous* thrust (inefficient on an
  eccentric orbit); the efficient duty-cycle-limited version is the scheduler's
  job. The capture is **real** — bound, E < 0, validated in GMAT.
