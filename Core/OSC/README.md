# NOESIS-Σ — Core/README

Short developer reference for the **operator‑split core**. Covers public interfaces, params, environment knobs, snapshot manifest, and telemetry metrics used by `Core/dynamics.py`.

---

## 1) Overview

The core advances a **working state** `x ∈ R^d` via an operator‑split step:

* **Conservative update** (skew‐symmetric kernel `K` with spectral cap)
* **Dissipative prox** via **ICNN** potential `Φ` (`∇Φ` used in an implicit iteration)
* **Projection** to feasible set (norm ball with ε tolerance)

Determinism, snapshotting (sign/encrypt), telemetry hooks, and an optional Triton fast‑path for `K·x` are built in.

---

## 2) Public Interfaces

### 2.1 `EngineParams` (from `Core.params.load_params()`)

Key fields (non‑exhaustive):

| Field                   | Type            | Default | Notes                                     |       |         |
| ----------------------- | --------------- | ------: | ----------------------------------------- | ----- | ------- |
| `state_dim`             | int             |    1024 | Working dimension `d`                     |       |         |
| `dt`                    | float           |   0.005 | Step size                                 |       |         |
| `max_norm`              | float           |    16.0 | Projection radius                         |       |         |
| `spectral_cap`          | float           |     1.0 | Spectral norm cap for conservative part   |       |         |
| `implicit_iters`        | int             |       1 | ICNN implicit iterations                  |       |         |
| `implicit_tol`          | float           |    1e‑6 | Convergence tol (1e‑4 suggested for fp16) |       |         |
| `proj_eps`              | float           |    1e‑6 | Projection tolerance                      |       |         |
| `rank_r`                | int             |      32 | Low‑rank K (U,V) rank                      |       |         |
| `clip_nan_policy`       | str             | `raise` | `raise                                    | clamp | rewind` |
| `max_step_time_ms`      | float           |   200.0 | Watchdog threshold                        |       |         |
| `deterministic`         | bool            |    True | RNG & algorithm flags                     |       |         |
| `dtype`                 | torch.dtype     |    fp32 | `float16/32/64`                           |       |         |
| `device`                | torch.device    |     cpu | e.g. `cuda:0`                             |       |         |
| `enable_triton`         | bool            |   False | Triton kernel path for `K·x`              |       |         |
| `allow_mixed_precision` | bool            |   False | autocast guard in dissipative step        |       |         |
| `icnn`                  | nn.Module       |    None | Must implement `forward()` & `grad()`     |       |         |
| `telemetry`             | TelemetryClient | default | `.record(name, payload)`                  |       |         |
| `thermostat_hook`       | callable        |    None | `hook(engine, metrics)`                   |       |         |
| `snapshot_*key/pem`     | bytes           |    None | AES key & Ed25519 PEMs (optional)         |       |         |

Helpers:

* `load_params(path: Optional[str]) -> EngineParams`
* `save_params(params, path)`

### 2.2 `OperatorSplitEngine`

```python
eng = OperatorSplitEngine(params, icnn: Optional[nn.Module] = None)
eng.set_seed(int)
eng.set_device(torch.device)
eng.set_dtype(torch.dtype)

x1 = eng.step(x0, *, sim_graft=None, trace_id=None, telemetry_tag=None)
xn = eng.step_many(x0, n_steps: int | None = None, *, steps: int | None = None,
                   telemetry_tag: str | None = None)

E = eng.energy(x)
eng.save_snapshot(dirpath, sign=True, encrypt=True)
eng.load_snapshot(dirpath, verify=True, decrypt=True)
```

**Shapes:** `x` may be `[d]` or batched `[B,d]`; output matches input batch semantics.

**SIM graft:** optional additive vector (broadcastable to `[B,d]`).

---

## 3) Environment Variables (read by `Core/params.py`)

```
NOESIS_DEVICE=cpu|cuda:0
NOESIS_STATE_DIM=1024
NOESIS_DT=0.005
NOESIS_MAX_NORM=16.0
NOESIS_SPECTRAL_CAP=1.0
NOESIS_IMPLICIT_ITERS=1
NOESIS_IMPLICIT_TOL=1e-6
NOESIS_CLIP_NAN_POLICY=raise|clamp|rewind
NOESIS_MAX_STEP_TIME_MS=200.0
NOESIS_ENABLE_TRITON=true|false
NOESIS_SNAPSHOT_AES_KEY_PATH=Runtime/Config/keys/model.aes.key
NOESIS_SNAPSHOT_SIGN_PRIVATE_PEM_PATH=Runtime/Config/keys/ed25519_private.pem
NOESIS_SNAPSHOT_SIGN_PUBLIC_PEM_PATH=Runtime/Config/keys/ed25519_public.pem
```

Place secrets under `Runtime/Config/keys/` (`chmod 600`).

---

## 4) Snapshot Manifest & Files (canonical)

**Directory layout:**

```
<snapshot_dir>/
  manifest.json        # metadata
  snapshot.bin         # torch.save blob (possibly AES‑GCM encrypted)
  snapshot.sig         # Ed25519 signature (optional)
```

**`manifest.json` schema:**

```json
{
  "version": "1.0",
  "created_at": 1690000000,
  "step_count": 123,
  "state_dim": 1024,
  "dt": 0.005,
  "spectral_est": 0.12345,
  "snapshot_sha256": "<hex_sha256_of_snapshot.bin>",
  "signed": true,
  "encrypted": true,
  "icnn_included": true,
  "icnn_arch": "ICNN-v1",
  "engine_git_ref": "<commit-sha-or-tag>"
}
```

**`snapshot.bin` contents (via `torch.save`)** must include keys:

* `K_matrix` **or** `K_U` & `K_V` (CPU tensors)
* `last_good_state` (CPU tensor)
* `rng.torch` (`torch.get_rng_state()`)
* `rng.cuda` (`torch.cuda.get_rng_state_all()` or `None`)
* `icnn_state` (if ICNN present & supports `state_dict()`)
* `meta` (dict: `created_at`, `step_count`, `state_dim`)

**Security model:**

* Sign the **plaintext blob** with Ed25519 → write `snapshot.sig`.
* Optionally encrypt `snapshot.bin` with AES‑GCM (32‑byte key). Store key in `Runtime/Config/keys/model.aes.key`.

---

## 5) Telemetry Metrics (names & payload fields)

The core uses a simple `TelemetryClient.record(name: str, payload: dict)` contract. Canonical names:

* `noesis.core.step`

  * Fields: `trace_id, step_count, elapsed_ms, energy_before, energy_after, norm_after, projection_count`
* `noesis.core.watchdog`

  * Fields: `elapsed_ms, max_ms`
* `noesis.core.nan_event`

  * Fields: `policy` (`raise|clamp|rewind`), `trace_id`
* `noesis.core.spectral_est`

  * Fields: `estimate, cap`

> Tip: Persist to `Runtime/Logs/noesis.log` (via `loguru`) and/or export to Prometheus in API layer.

---

## 6) Triton / Kernels

* Torch fallback lives in `Core/kernels/placeholder.py`.
* Optional Triton path in `Core/Kernels/triton_kernel.py` (enable with `NOESIS_ENABLE_TRITON=true` and `device=cuda`).
* Engine auto‑selects Triton **only** when CUDA available and flag is set; otherwise uses Torch matmuls/low‑rank.

---

## 7) Determinism & RNG

* `EngineParams.deterministic=True` enables deterministic algorithms where possible.
* Snapshots capture `rng.torch` and `rng.cuda` states.
* Tests include deterministic replay from snapshot.

---

## 8) Logging & Thermostat

* Recommended logging helper: `Tools/logging_setup.py` → logs to `Runtime/Logs/noesis.log` and keeps console quiet (`WARNING+`).
* Optional `thermostat_hook(engine, metrics)` reacts to slow steps or repeated projections; may reduce `dt`, emit alerts, or dump snapshots.

---

## 9) Minimal Usage Snippets

**Create engine and step:**

```python
from Core import load_params, OperatorSplitEngine
from Core.icnn import ICNN
import torch

p = load_params(); p.state_dim = 1024; p.icnn = ICNN(p.state_dim, [128,64])
eng = OperatorSplitEngine(p)
x0 = torch.randn(p.state_dim, device=eng.device, dtype=p.dtype)
x1 = eng.step(x0, telemetry_tag="fast")
```

**Snapshot round‑trip:**

```python
eng.save_snapshot("Runtime/Snapshots/snap_0001", sign=True, encrypt=True)
eng2 = OperatorSplitEngine(load_params())
eng2.load_snapshot("Runtime/Snapshots/snap_0001", verify=True, decrypt=True)
```

**Triton fast‑path (optional):**

```bash
NOESIS_DEVICE=cuda NOESIS_ENABLE_TRITON=true python -m Scripts.bench_vram_quick
```

---

## 10) Testing Markers

* **Fast (PR)**: `pytest -q -m fast` (≤ 5 min) — small dims (≤ 32), ≤ 100 steps
* **Nightly**: `pytest -q -m nightly` — long‑run stability / VRAM flatness (GPU when present)

> Ensure `pytest.ini` registers custom marks to silence warnings.

---

## 11) Permissions & Paths (ops hygiene)

* `Runtime/Config/keys/*` → `chmod 600`
* `Runtime/Snapshots/*` → `chmod 640` (rotate periodically)
* `Runtime/Logs/*` → append‑only by service user

---

## 12) Contact Points / Next Steps

* Wire receipts to API responses (`Output/receipts.py`).
* Connect Verifier adapter (`Verifier/adapter.py`) to enforce policy.
* Integrate SIM graft (`SIM/query.py`) for memory‑augmented steps.

> With Core green and VRAM flat, the next milestone is **SIM** + **API** integration.
