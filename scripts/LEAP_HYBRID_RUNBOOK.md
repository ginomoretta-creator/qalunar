# Leap Hybrid Sampler Runbook

This is the exact procedure to obtain a single hardware-tier datum
for the paper using D-Wave Leap Hybrid (BQM solver). Time to
complete: **5–10 minutes**, including the cloud submission.

## What Leap Hybrid is

`LeapHybridSampler` is D-Wave's production hybrid solver: classical
heuristics on the cloud orchestrating decomposition + QPU calls
under the hood. It is **not** the same as the open-source
`dwave-hybrid` framework that we already use locally for Kerberos.
Leap Hybrid runs on D-Wave's machines; Kerberos runs on yours.

For a paper claiming applicability of QA to cislunar scheduling,
**one Leap Hybrid datum is the credible-evidence floor**.

## Free time available

D-Wave's standard free Leap account includes ~1 minute per month
of hybrid solver time (verify on your account dashboard). One
submission of an N=200 problem typically costs 3–5 seconds of
that budget.

## Step 1 — Get your token

1. Sign in at <https://cloud.dwavesys.com/leap/>
2. Open your account profile (top right).
3. Copy the **API Token** field. It looks like
   `DEV-1234abcd...` (long random string).

## Step 2 — Configure dwave-cloud locally

In a terminal at the project root:

```bash
dwave config create
```

Press Enter to accept the default profile name (`defaults`),
default URL, default client. When it asks for the token, paste the
one from Step 1. Confirm with `dwave ping --client hybrid`; it
should report success.

Alternatively, for a one-shot run set the token as an environment
variable:

```bash
export DWAVE_API_TOKEN="DEV-1234abcd..."   # bash
$env:DWAVE_API_TOKEN = "DEV-1234abcd..."   # PowerShell
```

## Step 3 — Run the benchmark

```bash
python -m scripts.try_leap_hybrid
```

The script auto-detects the token. With it configured, it submits
the cislunar correction QUBO at N ∈ {20, 40, 60} to Leap Hybrid
and to local SA, then prints a comparison table. Without a token,
it prints these instructions and exits cleanly with an SA smoke
test.

Expected output format:

```
   N          method         energy    time(s)    qpu(us)
------------------------------------------------------------
  20              SA      X.XXXXe-XX      X.XXX          -
  20     leap_hybrid      X.XXXXe-XX      X.XXX     XXXXXX
  40              SA      X.XXXXe-XX      X.XXX          -
  40     leap_hybrid      X.XXXXe-XX      X.XXX     XXXXXX
  ...
```

The `qpu(us)` column is the QPU access time inside the hybrid
solver, in microseconds — that is the literal quantum-annealing
contribution.

## Step 4 — Paste back into the paper

In `paper/paper.tex` find Table~\ref{tab:scaling} (the scaling
table). Add a `Leap Hybrid (s)` column with the measured times.
Then update §6 (Discussion) to note that the hardware run was
performed and what it found.

If the result is "hybrid finds the same energy as SA but
faster/slower at large N", that is a publishable comparison. If it
finds something different, dig in (different solution, different
energy, hardware noise) — that is a result worth discussing in
§6.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `RuntimeError: No Leap API token found` | Re-run Step 2; check `~/.config/dwave/dwave.conf`. |
| `SolverNotFoundError: hybrid_binary_quadratic_model_version2 not available` | Your free tier is exhausted, or the solver name has changed. Check the [D-Wave dashboard](https://cloud.dwavesys.com/leap/). |
| Job hangs longer than 60 s | Either the problem is too large or the queue is busy. Cancel with Ctrl-C and re-submit a smaller N. |

## Notes for follow-on work

- For a publication-grade claim about quantum annealing (not just
  hybrid), Advantage 2 direct submission via
  `EmbeddingComposite(DWaveSampler())` is the next step. That
  requires explicit Advantage 2 access, which Sapienza or CINECA
  may need to arrange (Carbone et~al.\ used an ISCRA Type C grant).
- The same QUBO submitted to Leap's `LeapHybridNLSampler` (the
  nonlinear hybrid solver, the one Carbone et~al.\ used) gives a
  second hardware-tier datum. The Ocean SDK API is similar; only
  the sampler class name changes.
