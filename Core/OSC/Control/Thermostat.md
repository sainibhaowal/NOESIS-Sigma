# Thermostat module – NOESIS-Σ Core

## 1. Purpose

The thermostat controls **how many inner physics steps (S)** we run per token
and **how big each time step (dt)** is.

Goal:
- Spend **more compute** when the system is unstable or changing fast.
- Spend **less compute** when the system is calm.

This keeps the model stable and accurate, but also fast enough for realtime chat.

---

## 2. Key concepts

- **Energy E**: a scalar that measures how "excited" or unstable the current state is.
  (For example: gradient norm, loss proxy, or a simple norm of the state.)

- **S (inner steps)**: how many physics steps we run for this token.
  Higher S = deeper thinking / more accurate but more compute.

- **dt (time step)**: the size of each inner step.
  Smaller dt = safer and more precise integration, but requires more steps.

---

## 3. Behaviour (informal algorithm)

1. Start each token with some base values:
   - `S = S_min` (e.g. 64)
   - `dt = dt_max` (e.g. 0.005)

2. On each inner step:
   - Compute current energy `E_t`.
   - Compare with previous energy `E_{t-1}` to get `ΔE`.

3. If **|ΔE| is large** (transient / spike):
   - Increase S up to `S_max`.
   - Decrease dt down to `dt_min`.
   This gives more, smaller steps in a sensitive region.

4. If **|ΔE| stays small for M steps** (plateau / calm):
   - Decrease S (e.g. halve it or step it down).
   - Increase dt slightly.
   This saves compute in stable regions.

5. Clamp values:
   - `S_min <= S <= S_max`
   - `dt_min <= dt <= dt_max`

---

## 4. Interaction with the dynamics loop

- The outer loop (token generation) calls something like `step_many(...)`.
- `step_many`:
  - Resets / prepares the thermostat at token boundaries.
  - For each inner step:
    - Runs one physics step with current dt.
    - Passes the new energy to the thermostat.
    - The thermostat updates S and dt for future steps.

Result:
- For simple inputs, the model uses smaller S and bigger dt → faster.
- For difficult or unstable inputs, the model temporarily uses larger S and smaller dt → safer and more accurate.

---

## 5. Tests and trace

- `Tests/core/test_thermostat.py` checks basic behaviours (increasing S on transients, decreasing S on calm plateaus, dt bounds, etc.).
- `Tests/core/test_thermostat_acceptance.py` checks wiring into the real dynamics.
- `Scripts/trace_thermostat.py` produces CSV traces of `step, E, S, dt` to visually
  confirm the adaptation logic.

The sample trace in `Runtime/Logs/thermo_trace_*.csv` shows:
- S ramping up while E is slowly increasing.
- A transient spike in E.
- S held high during the spike.
- Then S being reduced again once the system becomes stable.
