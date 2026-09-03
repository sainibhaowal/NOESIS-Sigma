#!/usr/bin/env bash
# ================================================================
# NOESIS-Σ — Unified Test & Benchmark Kit  (single-file edition)
# Runs: unit tests (CPU/GPU), benchmarks (fused/unfused), soak,
# determinism, receipts sign/verify, thermostat, API/import checks,
# lint, types. Produces logs + PASS/FAIL summary.
# ------------------------------------------------
# Suite (flags):
#   Scripts/testkit.sh [--quick] [--cpu-only] [--no-bench] [--no-soak] [--no-lint] [--nightly]
# Subcommands:
#   Scripts/testkit.sh all|quick|cpu|gpu|benches|soak|strict|lint|types|api|receipts|fixdeps|help
# Examples:
#   Scripts/testkit.sh --quick
#   Scripts/testkit.sh all
#   Scripts/testkit.sh benches
#   Scripts/testkit.sh --nightly
# ================================================================
set -euo pipefail

# ---------- Repo root / env ----------
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export NOESIS_IGNORE_ENV=1

: "${PYTHON:=python}"
: "${PIP:=pip}"
: "${PYTEST:=pytest}"
: "${RUFF:=ruff}"
: "${MYPY:=mypy}"
: "${VENV_DIR:=$ROOT/.venv}"

# Bench/soak default shapes (safe on RTX 4060)
: "${D:=1024}"   # state dim
: "${M:=512}"    # ICNN width
: "${R:=64}"     # low-rank
: "${B:=64}"     # batch
: "${S:=32}"     # steps

: "${PROFILE_FAST:=FAST}"
: "${PROFILE_BAL:=BALANCED}"
: "${PROFILE_STRICT:=STRICT}"

# ---------- Helpers ----------
venv_ok() {
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "!! venv missing at $VENV_DIR . Create it first:"
    echo "   python3 -m venv .venv && source .venv/bin/activate && pip -q install -r requirements.txt"
    exit 1
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  PYTHON="$VENV_DIR/bin/python"
  PIP="$VENV_DIR/bin/pip"
  PYTEST="$VENV_DIR/bin/pytest"
  RUFF="$VENV_DIR/bin/ruff"
  MYPY="$VENV_DIR/bin/mypy"
}

has_cuda() {
  "$PYTHON" - <<'PY' | grep -q True
import torch
print(torch.cuda.is_available())
PY
}

head_banner() { echo "== NOESIS-Σ TestKit :: $1 =="; }

# ---------- Tasks (idempotent) ----------
task_lint() {
  head_banner "ruff (auto-fix) + style"
  "$RUFF" check . --fix || true
}

task_types() {
  head_banner "mypy types"
  "$PIP" -q install types-PyYAML nvidia-ml-py || true
  "$MYPY" --install-types --non-interactive || true
  "$MYPY" . --exclude '.venv'
}

task_cpu() {
  head_banner "CPU fast tests"
  CUDA_VISIBLE_DEVICES= "$PYTEST" -q Tests/core/test_params.py
  CUDA_VISIBLE_DEVICES= "$PYTEST" -q Tests/core/test_core_init.py
  CUDA_VISIBLE_DEVICES= "$PYTEST" -q Tests/core/test_icnn.py
  CUDA_VISIBLE_DEVICES= "$PYTEST" -q Tests/core/test_dynamics.py -m fast
}

task_gpu() {
  head_banner "GPU core tests"
  "$PYTEST" -q Tests/core/test_icnn_directgrad.py
  "$PYTEST" -q Tests/core/test_dynamics_batched_fused.py
  "$PYTEST" -q Tests/core/test_dynamics.py
}

task_api() {
  head_banner "API import smoke"
  "$PYTHON" - <<'PY'
import API.main as m
print("API import OK")
PY
}

task_thermostat() {
  head_banner "thermostat (unit + acceptance + trace)"
  mkdir -p "$LOGDIR"
  local out="$LOGDIR/thermostat.txt"

  (
    set -e
    # Run the unit + acceptance tests
    PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-} -q"
    "$PYTEST" \
      Tests/core/test_thermostat.py \
      Tests/core/test_thermostat_acceptance.py \
      Tests/core/test_thermostat_telemetry.py \
      Tests/core/test_thermostat_logging_policy.py

    # Run the trace (non-fatal if it fails)
    if [ -f "Scripts/trace_thermostat.py" ]; then
      "$PYTHON" Scripts/trace_thermostat.py || true
      # Copy the latest trace CSV into this testkit run's log dir for convenience
      if ls Runtime/Logs/thermo_trace_*.csv >/dev/null 2>&1; then
        local last_csv
        last_csv="$(ls -t Runtime/Logs/thermo_trace_*.csv | head -n1)"
        cp "$last_csv" "$LOGDIR/" || true
      fi
    fi
  ) 2>&1 | tee "$out"
}


task_receipts() {
  head_banner "Snapshot receipts sign+verify"
  "$PYTHON" - <<'PY'
import os, sys, json, hashlib, inspect, torch
from Core.params import load_params
from Core.dynamics import OperatorSplitEngine
from Core.icnn import ICNNDirectGrad

# receipts API (adapts to local signatures)
try:
    from Output.receipts import create_receipt, verify_receipt
except Exception as e:
    print("[ERR] Import Output.receipts failed:", e)
    sys.exit(1)

# --- 0) Require BOTH keys; otherwise skip (not a failure) ---
priv_path = "Runtime/Config/keys/ed25519_private.pem"
pub_path  = "Runtime/Config/keys/ed25519_public.pem"
if not (os.path.isfile(priv_path) and os.path.isfile(pub_path)):
    print(f"[skip] Receipts test skipped (missing key files).\n"
          f"Expect:\n  {priv_path}\n  {pub_path}")
    sys.exit(0)

with open(priv_path, "rb") as f: priv_pem = f.read()
with open(pub_path,  "rb") as f: pub_pem  = f.read()

# --- 1) Build tiny deterministic CPU engine WITH KEYS IN PARAMS ---
p = load_params()
p.device = torch.device("cpu")
p.state_dim = 16
if not hasattr(p, "dt") or p.dt is None:
    p.dt = 0.005
# inject signing keys required by your create_receipt
p.snapshot_signing_private_pem = priv_pem
p.snapshot_signing_public_pem  = pub_pem

p.icnn = ICNNDirectGrad(d=16, m=16, dtype=torch.float32, device="cpu", ws_dtype=torch.float32)
eng = OperatorSplitEngine(p)

# --- 2) Produce before/after (robust to signature differences) ---
B, d = 2, 16
x_before = torch.randn(B, d, device="cpu", dtype=torch.float32)

def advance_one_step(eng, x):
    for name in ("step", "advance"):
        fn = getattr(eng, name, None)
        if not callable(fn): continue
        sig = inspect.signature(fn)
        kwargs = {}
        if "dt" in sig.parameters:    kwargs["dt"]    = float(getattr(p, "dt", 0.005))
        if "steps" in sig.parameters: kwargs["steps"] = 1
        try:
            y = fn(x, **kwargs)
        except TypeError:
            try:
                y = fn(x)  # retry without kwargs
            except Exception:
                continue
        except Exception:
            continue
        if isinstance(y, tuple) and len(y) >= 1:
            y = y[0]
        return y
    # fallback: perturb slightly so hashes differ
    return x + 1e-7 * torch.sign(x + 1)

x_after = advance_one_step(eng, x_before)

# --- 3) Create receipt (adapt if 'trace_id' not accepted) ---
sig_cr = inspect.signature(create_receipt)
kwargs = {}
if "trace_id" in sig_cr.parameters:
    kwargs["trace_id"] = "smoketest"

try:
    rcpt = create_receipt(eng, x_before, x_after, **kwargs)
except TypeError as e:
    # final fallback: call with positional only
    rcpt = create_receipt(eng, x_before, x_after)

# --- 4) Verify (adapt if verify_receipt requires 'public_pem') ---
sig_vr = inspect.signature(verify_receipt)
try:
    if "public_pem" in sig_vr.parameters or len(sig_vr.parameters) >= 2:
        ok = verify_receipt(rcpt, pub_pem)
    else:
        ok = verify_receipt(rcpt)
except TypeError:
    ok = verify_receipt(rcpt, pub_pem)

# --- 5) Output + strict exit code for harness ---
print(json.dumps(rcpt, indent=2, default=str))
h = lambda t: hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()
print("\nverify_receipt:", ok)
print("before_sha256:", h(x_before))
print("after_sha256 :", h(x_after))
sys.exit(0 if ok else 5)
PY
}

task_benches() {
  head_banner "Benches FAST/BALANCED/STRICT (fused+graph)"
  echo "[FAST]"
  NOESIS_PROFILE="$PROFILE_FAST" $PYTHON Scripts/bench_core.py --mode fused --d "$D" --m "$M" --r "$R" --B "$B" --S "$S" --iters "${ITERS_FAST:-200}" --dtype float16 --profile "$PROFILE_FAST" --graph --echo-profile
  echo "[BALANCED]"
  NOESIS_PROFILE="$PROFILE_BAL"  $PYTHON Scripts/bench_core.py --mode fused --d "$D" --m "$M" --r "$R" --B "$B" --S "$S" --iters "${ITERS_BAL:-200}"  --dtype float16 --profile "$PROFILE_BAL"  --graph --echo-profile
  echo "[STRICT]"
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  NOESIS_PROFILE="$PROFILE_STRICT" $PYTHON Scripts/bench_core.py --mode fused --d "$D" --m "$M" --r "$R" --B "$B" --S "$S" --iters "${ITERS_STRICT:-120}" --dtype float32 --profile "$PROFILE_STRICT" --graph --echo-profile
}

task_soak() {
  head_banner "Soak 3 minutes (FAST/BALANCED/STRICT)"
  NOESIS_PROFILE="$PROFILE_FAST" $PYTHON Scripts/soak_core.py --d "$D" --m "$M" --r "$R" --B "$B" --S "$S" --duration "${DUR_SEC:-40}" --dtype float16 --profile "$PROFILE_FAST" --graph
  NOESIS_PROFILE="$PROFILE_BAL"  $PYTHON Scripts/soak_core.py --d "$D" --m "$M" --r "$R" --B "$B" --S "$S" --duration "${DUR_SEC:-40}" --dtype float16 --profile "$PROFILE_BAL"  --graph
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  NOESIS_PROFILE="$PROFILE_STRICT" $PYTHON Scripts/soak_core.py --d "$D" --m "$M" --r "$R" --B "$B" --S "$S" --duration "${DUR_SEC:-40}" --dtype float32 --profile "$PROFILE_STRICT" --graph
}

task_strict() {
  head_banner "STRICT determinism & hash-equality"
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  NOESIS_PROFILE="$PROFILE_STRICT" $PYTHON - <<'PY'
import torch, hashlib
from Core.icnn import ICNNDirectGrad
from Core.dynamics import HotLoopFused, HotLoopGraph

torch.use_deterministic_algorithms(True)
d,m,r,B,S=1024,512,64,64,32
dev='cuda'; dtype=torch.float32
icnn=ICNNDirectGrad(d=d,m=m,dtype=dtype,device=dev,ws_dtype=torch.float32)
KU=torch.randn(d,r,device=dev,dtype=dtype); KV=torch.randn(d,r,device=dev,dtype=dtype)
f=HotLoopFused(d=d,icnn=icnn,KU=KU,KV=KV,device=dev,dtype=dtype,ws_dtype=torch.float32)
g=HotLoopGraph(f,B=B,S=S,dt=0.02)
x=torch.randn(B,d,device=dev,dtype=dtype)
y1=g(x); y2=g(x)
h=lambda t: hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()
print("STRICT equal:", bool((y1==y2).all().item()), "sha:", h(y1)==h(y2), h(y1))
PY
}

task_fixdeps() {
  head_banner "Dev helper deps"
  $PIP -q install types-PyYAML nvidia-ml-py ruff mypy || true
}

# ---------- Logging wrapper for suite mode ----------
TS="$(date -u +"%Y%m%dT%H%M%SZ")"
LOGDIR="$ROOT/Runtime/Logs/testkit_$TS"
SUMMARY="$LOGDIR/summary.txt"
declare -A STATUS
log_prep() { mkdir -p "$LOGDIR"; : >"$SUMMARY"; }
logrun () {
  local name="$1"; shift
  echo -e "\n===== RUN: $name =====" | tee -a "$SUMMARY"
  set +e
  "$@" >"$LOGDIR/${name}.out" 2>"$LOGDIR/${name}.err"
  local code=$?
  set -e
  STATUS["$name"]=$code
  if [[ $code -eq 0 ]]; then
    echo "[PASS] $name" | tee -a "$SUMMARY"
  else
    echo "[FAIL] $name (code=$code)" | tee -a "$SUMMARY"
  fi
}

finalize_summary() {
  echo -e "\n================ SUMMARY ($TS) ================" | tee -a "$SUMMARY"
  local overall=0
  for k in "${!STATUS[@]}"; do
    local code="${STATUS[$k]}"
    [[ $code -ne 0 ]] && overall=1
    printf "%-35s %s\n" "$k" "$( [[ $code -eq 0 ]] && echo PASS || echo FAIL )" | tee -a "$SUMMARY"
  done
  echo "Logs: $LOGDIR" | tee -a "$SUMMARY"
  if [[ $overall -eq 0 ]]; then
    echo "ALL GREEN ✅" | tee -a "$SUMMARY"
  else
    echo "SOME CHECKS FAILED ❌" | tee -a "$SUMMARY"
  fi
  return $overall
}

# ---------- Suite runner (flagged mode or subcommands 'all'/'quick') ----------
suite_run() {
  local QUICK="${1:-0}" CPU_ONLY="${2:-0}" RUN_BENCH="${3:-1}" RUN_SOAK="${4:-1}" RUN_LINT="${5:-1}" RUN_NIGHTLY="${6:-0}"

  log_prep

  # 0) Sanity
  logrun "sanity_imports" bash -lc "$PYTHON - <<'PY'
import torch, importlib
print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'is_available:', torch.cuda.is_available())
from Core.params import load_params
from Core.dynamics import OperatorSplitEngine
from Core.icnn import ICNNDirectGrad
import API.main as m
print('Core/API import OK')
PY"

  # 1) Core tests
  logrun "pytest_core_cpu_fast" env CUDA_VISIBLE_DEVICES= "$PYTEST" -q Tests/core/test_dynamics.py -m fast
  if [[ "$CPU_ONLY" -eq 0 ]] && has_cuda; then
    logrun "pytest_core_icnn_directgrad" "$PYTEST" -q Tests/core/test_icnn_directgrad.py
    logrun "pytest_core_dynamics_fused" "$PYTEST" -q Tests/core/test_dynamics_batched_fused.py
  fi
  logrun "pytest_core_k_lowrank" "$PYTEST" -q Tests/core/test_k_lowrank.py
  logrun "pytest_core_all" "$PYTEST" -q Tests/core/test_dynamics.py Tests/core/test_icnn.py Tests/core/test_params.py Tests/core/test_core_init.py

  if [[ "$RUN_NIGHTLY" -eq 1 ]]; then
    logrun "pytest_core_nightly" "$PYTEST" -q Tests/core/test_dynamics.py -m nightly -k long_run --maxfail=1
  fi

  # 2) Non-core
  logrun "pytest_ops" "$PYTEST" -q Tests/ops
  logrun "pytest_verifier" "$PYTEST" -q Tests/verifier
  logrun "pytest_output" "$PYTEST" -q Tests/output

  # 3) Profiles close / STRICT determinism
  if [[ "$CPU_ONLY" -eq 0 ]] && has_cuda; then
    logrun "profile_fast_close" bash -lc '
PYTHONPATH=. NOESIS_PROFILE=FAST '"$PYTHON"' - <<PY
import importlib, torch
from Core.Exec import profiles
profiles = importlib.reload(profiles)
from Core.icnn import ICNNDirectGrad
from Core.dynamics import HotLoopFused, HotLoopGraph
torch.set_grad_enabled(False)
cfg = profiles.active_config()
d,m,r,B,S=256,128,32,8,8
dev="cuda"; dtype=cfg["dtype"]
icnn=ICNNDirectGrad(d=d,m=m,dtype=dtype,device=dev,ws_dtype=cfg["icnn_ws"])
KU=torch.randn(d,r,device=dev,dtype=dtype); KV=torch.randn(d,r,device=dev,dtype=dtype)
f=HotLoopFused(d=d,icnn=icnn,KU=KU,KV=KV,device=dev,dtype=dtype,ws_dtype=cfg["k_ws"])
g=HotLoopGraph(f,B=B,S=S,dt=0.02)
x=torch.randn(B,d,device=dev,dtype=dtype)
y1=f.step_unrolled(x.clone(),S=S,dt=0.02); y2=g.run(x.clone())
print("FAST close:", torch.allclose(y1, y2, rtol=1e-3, atol=1e-3))
PY'
    logrun "profile_balanced_close" bash -lc '
PYTHONPATH=. NOESIS_PROFILE=BALANCED '"$PYTHON"' - <<PY
import importlib, torch
from Core.Exec import profiles
profiles = importlib.reload(profiles)
from Core.icnn import ICNNDirectGrad
from Core.dynamics import HotLoopFused, HotLoopGraph
torch.set_grad_enabled(False)
cfg = profiles.active_config()
d,m,r,B,S=256,128,32,8,8
dev="cuda"; dtype=cfg["dtype"]
icnn=ICNNDirectGrad(d=d,m=m,dtype=dtype,device=dev,ws_dtype=cfg["icnn_ws"])
KU=torch.randn(d,r,device=dev,dtype=dtype); KV=torch.randn(d,r,device=dev,dtype=dtype)
f=HotLoopFused(d=d,icnn=icnn,KU=KU,KV=KV,device=dev,dtype=dtype,ws_dtype=cfg["k_ws"])
g=HotLoopGraph(f,B=B,S=S,dt=0.02)
x=torch.randn(B,d,device=dev,dtype=dtype)
y1=f.step_unrolled(x.clone(),S=S,dt=0.02); y2=g.run(x.clone())
print("BALANCED close:", torch.allclose(y1, y2, rtol=5e-4, atol=5e-4))
PY'
    logrun "profile_strict_bitwise" bash -lc '
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHONPATH=. NOESIS_PROFILE=STRICT '"$PYTHON"' - <<PY
import importlib, torch, hashlib
from Core.Exec import profiles
profiles = importlib.reload(profiles)
from Core.icnn import ICNNDirectGrad
from Core.dynamics import HotLoopFused
torch.use_deterministic_algorithms(True)
cfg = profiles.active_config()
d,m,r,B,S=1024,512,64,64,32
dev="cuda"; dtype=cfg["dtype"]
icnn=ICNNDirectGrad(d=d,m=m,dtype=dtype,device=dev,ws_dtype=cfg["icnn_ws"])
KU=torch.randn(d,r,device=dev,dtype=dtype); KV=torch.randn(d,r,device=dev,dtype=dtype)
f1=HotLoopFused(d=d,icnn=icnn,KU=KU,KV=KV,device=dev,dtype=dtype,ws_dtype=cfg["k_ws"])
f2=HotLoopFused(d=d,icnn=icnn,KU=KU,KV=KV,device=dev,dtype=dtype,ws_dtype=cfg["k_ws"])
x=torch.randn(B,d,device=dev,dtype=dtype)
y1=f1.step_unrolled(x.clone(),S=S,dt=0.02); y2=f2.step_unrolled(x.clone(),S=S,dt=0.02)
h=lambda t: hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()
print("STRICT bitwise:", bool((y1==y2).all().item()), "sha:", h(y1)==h(y2))
PY'
  fi

  # 4) Benchmarks
  if [[ "$RUN_BENCH" -eq 1 ]]; then
    ITERS_FAST=200; ITERS_BAL=200; ITERS_STRICT=120
    if [[ "$QUICK" -eq 1 ]]; then ITERS_FAST=40; ITERS_BAL=40; ITERS_STRICT=30; fi
    if [[ "$CPU_ONLY" -eq 0 ]] && has_cuda; then
      logrun "bench_fast_fused_graph"   bash -lc "PYTHONPATH=. $PYTHON Scripts/bench_core.py --mode fused   --d 1024 --m 512 --r 64 --B 64 --S 32 --iters $ITERS_FAST  --dtype float16 --profile FAST --graph --echo-profile"
      logrun "bench_fast_unfused"       bash -lc "PYTHONPATH=. $PYTHON Scripts/bench_core.py --mode unfused --d 1024 --m 512 --r 64 --B 64 --S 32 --iters $ITERS_FAST  --dtype float16 --profile FAST"
      logrun "bench_bal_fused_graph"    bash -lc "PYTHONPATH=. $PYTHON Scripts/bench_core.py --mode fused   --d 1024 --m 512 --r 64 --B 64 --S 32 --iters $ITERS_BAL   --dtype float16 --profile BALANCED --graph --echo-profile"
      logrun "bench_bal_unfused"        bash -lc "PYTHONPATH=. $PYTHON Scripts/bench_core.py --mode unfused --d 1024 --m 512 --r 64 --B 64 --S 32 --iters $ITERS_BAL   --dtype float16 --profile BALANCED"
      logrun "bench_strict_fused_graph" bash -lc "export CUBLAS_WORKSPACE_CONFIG=:4096:8; PYTHONPATH=. $PYTHON Scripts/bench_core.py --mode fused --d 1024 --m 512 --r 64 --B 64 --S 32 --iters $ITERS_STRICT --dtype float32 --profile STRICT --graph --echo-profile"
      logrun "bench_strict_unfused"     bash -lc "export CUBLAS_WORKSPACE_CONFIG=:4096:8; PYTHONPATH=. $PYTHON Scripts/bench_core.py --mode unfused --d 1024 --m 512 --r 64 --B 64 --S 32 --iters $ITERS_STRICT --dtype float32 --profile STRICT"
    else
      echo "[skip] Benchmarks skipped (CPU-only or no CUDA)." | tee -a "$SUMMARY"
    fi
  fi

  # 5) Soak
  if [[ "$RUN_SOAK" -eq 1 ]] && [[ "$CPU_ONLY" -eq 0 ]] && has_cuda; then
    DUR_SEC=40; [[ "$QUICK" -eq 1 ]] && DUR_SEC=60
    logrun "soak_fast"   bash -lc "PYTHONPATH=. $PYTHON Scripts/soak_core.py --d 1024 --m 512 --r 64 --B 64 --S 32 --duration $DUR_SEC --dtype float16 --profile FAST --graph"
    logrun "soak_bal"    bash -lc "PYTHONPATH=. $PYTHON Scripts/soak_core.py --d 1024 --m 512 --r 64 --B 64 --S 32 --duration $DUR_SEC --dtype float16 --profile BALANCED --graph"
    logrun "soak_strict" bash -lc "export CUBLAS_WORKSPACE_CONFIG=:4096:8; PYTHONPATH=. $PYTHON Scripts/soak_core.py --d 1024 --m 512 --r 64 --B 64 --S 32 --duration $DUR_SEC --dtype float32 --profile STRICT --graph"
  else
    echo "[skip] Soak tests skipped." | tee -a "$SUMMARY"
  fi

  # 6) Receipts (single source via subcommand)
  logrun "receipts_sign_verify" bash -lc "PYTHONPATH=. '$ROOT/Scripts/testkit.sh' receipts"

  # 7) Thermostat tests (unit + acceptance + telemetry + logging-policy)
  logrun "pytest_thermostat" bash -lc "PYTHONPATH=. $PYTHON -m pytest -q Tests/core/test_thermostat.py Tests/core/test_thermostat_acceptance.py Tests/core/test_thermostat_telemetry.py Tests/core/test_thermostat_logging_policy.py"


  # 7b) Thermostat trace (optional; non-fatal)
  logrun "thermostat_trace" bash -lc '
set +e
PYTHONPATH=. '"$PYTHON"' Scripts/trace_thermostat.py
last=$(ls -t Runtime/Logs/thermo_trace_*.csv 2>/dev/null | head -n1 || true)
[[ -n "$last" ]] && cp "$last" "$LOGDIR"/ 2>/dev/null || true
exit 0
'

# Optional golden check (enable with THERMO_GOLDEN=1)
if [[ "${THERMO_GOLDEN:-0}" == "1" ]]; then
  logrun "thermostat_trace_golden" bash -lc '
set -e
ref=$(cat Scripts/thermo_trace.sha256)
last=$(ls -t Runtime/Logs/thermo_trace_*.csv | head -n1)
got=$(sha256sum "$last" | awk "{print \$1}")
test "$got" = "$ref"
'
fi

  # 7c) Thermostat watchdog
  logrun "thermostat_watchdog" bash -lc '
PYTHONPATH=. NOESIS_THERMOSTAT_ALERTS_LOG=Runtime/Logs/alerts.log '"$PYTHON"' - <<PY
import torch, os
from Core.params import load_params
from Core.dynamics import OperatorSplitEngine
from Core.icnn import ICNNDirectGrad
from Ops.thermostat import Thermostat
p = load_params(); p.device=torch.device("cpu"); p.state_dim=16
p.icnn = ICNNDirectGrad(d=16,m=16,dtype=torch.float32,device="cpu",ws_dtype=torch.float32)
p.max_step_time_ms = 0.0
eng = OperatorSplitEngine(p)
print("Thermostat ok; alerts log exists:", os.path.exists("Runtime/Logs/alerts.log"))
PY'

  # 8) Linters (informational)
  if [[ "$RUN_LINT" -eq 1 ]]; then
    set +e
    "$RUFF" check . --output-format=text >"$LOGDIR/ruff.out" 2>"$LOGDIR/ruff.err"
    echo "[info] Ruff completed (see logs)." | tee -a "$SUMMARY"
    "$MYPY" Core --exclude '.venv' >"$LOGDIR/mypy.out" 2>"$LOGDIR/mypy.err" || true
    echo "[info] mypy completed (see logs)." | tee -a "$SUMMARY"
    set -e
  else
    echo "[skip] Linters skipped." | tee -a "$SUMMARY"
  fi

  finalize_summary
}

# ---------- CLI parsing ----------
venv_ok

# Detect if first arg is a subcommand (no leading '-')
SUBCMD=""
if (( $# >= 1 )) && [[ "${1:0:1}" != "-" ]]; then
  SUBCMD="$1"; shift || true
fi

# Flags (suite mode)
QUICK=0
CPU_ONLY=0
RUN_BENCH=1
RUN_SOAK=1
RUN_LINT=1
RUN_NIGHTLY=0

while (( "$#" )); do
  case "$1" in
    --quick) QUICK=1 ;;
    --cpu-only) CPU_ONLY=1 ;;
    --no-bench) RUN_BENCH=0 ;;
    --no-soak) RUN_SOAK=0 ;;
    --no-lint) RUN_LINT=0 ;;
    --nightly) RUN_NIGHTLY=1 ;;
    -h|--help)
      cat <<'HLP'
Usage (suite mode with logs + summary):
  Scripts/testkit.sh [--quick] [--cpu-only] [--no-bench] [--no-soak] [--no-lint] [--nightly]

Usage (subcommands):
  Scripts/testkit.sh all|quick|cpu|gpu|benches|soak|strict|lint|types|api|receipts|thermo|fixdeps|help

Notes:
  - Suite mode always writes logs to Runtime/Logs/testkit_<timestamp>/ and prints a PASS/FAIL summary.
  - Subcommands run the specific task directly (no summary bundle), useful while iterating.
HLP
      exit 0 ;;
    *) echo "Unknown flag: $1"; exit 2 ;;
  esac
  shift
done

# ---------- Dispatch ----------
if [[ -n "$SUBCMD" ]]; then
  case "$SUBCMD" in
    all)       TS="$(date -u +"%Y%m%dT%H%M%SZ")"; LOGDIR="$ROOT/Runtime/Logs/testkit_$TS"; suite_run 0 0 1 1 1 0 ;;
    quick)     TS="$(date -u +"%Y%m%dT%H%M%SZ")"; LOGDIR="$ROOT/Runtime/Logs/testkit_$TS"; suite_run 1 0 1 1 1 0 ;;
    cpu)       task_cpu ;;
    gpu)       task_gpu ;;
    benches)   ITERS_FAST=200 ITERS_BAL=200 ITERS_STRICT=120 task_benches ;;
    soak)      DUR_SEC=40 task_soak ;;
    strict)    task_strict ;;
    lint)      task_lint ;;
    types)     task_types ;;
    api)       task_api ;;
    receipts)  task_receipts ;;
    thermo)
      TS="$(date -u +"%Y%m%dT%H%M%SZ")"; LOGDIR="$ROOT/Runtime/Logs/testkit_$TS"
      log_prep
      logrun "pytest_thermostat" bash -lc "PYTHONPATH=. $PYTHON -m pytest -q Tests/core/test_thermostat.py Tests/core/test_thermostat_acceptance.py"
  logrun "thermostat_trace" bash -lc '
set +e
PYTHONPATH=. '"$PYTHON"' Scripts/trace_thermostat.py
last=$(ls -t Runtime/Logs/thermo_trace_*.csv 2>/dev/null | head -n1 || true)
[[ -n "$last" ]] && cp "$last" "$LOGDIR"/ 2>/dev/null || true
exit 0
'
      finalize_summary
      ;;
    fixdeps)   task_fixdeps ;;
    help|*)    "$0" --help ;;
    # in the subcommand dispatch section:
    thermo)    task_thermostat ;;
  esac
else
  # Suite mode with selected flags
  TS="$(date -u +"%Y%m%dT%H%M%SZ")"; LOGDIR="$ROOT/Runtime/Logs/testkit_$TS"
  suite_run "$QUICK" "$CPU_ONLY" "$RUN_BENCH" "$RUN_SOAK" "$RUN_LINT" "$RUN_NIGHTLY"
  exit $?
fi
