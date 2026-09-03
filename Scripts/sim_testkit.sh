#!/usr/bin/env bash
set -uo pipefail

MODE="${1:---quick}"

RUN_ID="${SIM_BENCH_RUN_ID:-SIMBENCH_$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT_DIR="$(pwd)"
BENCH_DIR="${SIM_BENCH_DIR:-$ROOT_DIR/Runtime/Benchmarks/$RUN_ID}"
LOG_DIR="$ROOT_DIR/Runtime/Logs"
SUMMARY_PATH="$BENCH_DIR/summary.json"
LOG_PATH="$LOG_DIR/sim_testkit_${RUN_ID}.log"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
PYTEST_BIN="${PYTEST_BIN:-$ROOT_DIR/.venv/bin/pytest}"

mkdir -p "$BENCH_DIR" "$LOG_DIR" "$BENCH_DIR/plots"

export NOESIS_REPO_ROOT="$ROOT_DIR"
export SIM_BENCH_RUN_ID="$RUN_ID"
export SIM_BENCH_DIR="$BENCH_DIR"
export SIM_ALLOW_SQLITE_FOR_TESTS="${SIM_ALLOW_SQLITE_FOR_TESTS:-1}"

STATUS="PASS"
FAIL_REASON=""

write_summary() {
  "$PYTHON_BIN" - <<PY
import json
out = {
  "run_id": """${RUN_ID}""",
  "status": """${STATUS}""",
  "fail_reason": """${FAIL_REASON}""",
}
with open("${SUMMARY_PATH}", "w", encoding="utf-8") as f:
  json.dump(out, f, indent=2)
PY
}

trap write_summary EXIT

run_step() {
  local name="$1"
  shift
  echo "[SIM] $name" | tee -a "$LOG_PATH"
  if ! "$@" 2>&1 | tee -a "$LOG_PATH"; then
    STATUS="FAIL"
    FAIL_REASON="$name"
    return 1
  fi
}

run_step "Unit tests" "$PYTEST_BIN" -q \
  Tests/Unit/test_sim_s0_basic_io.py \
  Tests/Unit/test_sim_s0_tenant_isolation.py \
  Tests/Unit/test_sim_s1_ot_stability.py \
  Tests/Unit/test_sim_s1_relevance.py \
  Tests/Unit/test_sim_s2_hot_bounded_under_load.py \
  Tests/Unit/test_sim_s2_ttl_deletes_logged.py \
  Tests/Unit/test_sim_s3_security_rbac.py \
  Tests/Unit/test_sim_s3_encryption_at_rest.py \
  Tests/Unit/test_sim_s3_ledger_integrity.py \
  Tests/Unit/test_sim_s4_snapshots_determinism.py \
  Tests/Unit/test_sim_s4_rollback_correctness.py \
  Tests/Sim/Tests/Unit/test_ot_transport_stability.py \
  Tests/Sim/Tests/Unit/test_compaction_mass_preservation.py \
  Tests/Sim/Tests/Unit/test_ttl_correctness.py \
  Tests/Sim/Tests/Unit/test_encryption_roundtrip.py \
  Tests/Sim/Tests/Unit/test_ledger_append_only.py \
  Tests/Sim/Tests/Unit/test_consent_revocation.py || exit 1

if [[ "$MODE" == "--acceptance" || "$MODE" == "--nightly" ]]; then
  run_step "Acceptance tests (short)" "$PYTEST_BIN" -q \
    Tests/Sim/Tests/Acceptance/test_vram_flatness_curve.py \
    Tests/Sim/Tests/Acceptance/test_latency_curve.py \
    Tests/Sim/Tests/Acceptance/test_soak_agent.py \
    Tests/Sim/Tests/Acceptance/test_rollback_recovery.py \
    Tests/Sim/Tests/Acceptance/test_consent_revocation_enforcement.py || exit 1
fi

if [[ "$MODE" == "--acceptance" || "$MODE" == "--nightly" ]]; then
  run_step "Bench (telemetry)" "$PYTHON_BIN" Scripts/sim_bench.py --runs 1 --fill 200 --reads 10 || exit 1
  run_step "Plots" "$PYTHON_BIN" Scripts/sim_plot.py --input "$BENCH_DIR/telemetry.jsonl" --out-dir "$BENCH_DIR/plots" || exit 1
fi

if [[ "$MODE" == "--nightly" ]]; then
  run_step "Soak (nightly)" "$PYTHON_BIN" Scripts/sim_soak_agent.py --duration-sec 3600 --out "$BENCH_DIR/soak_summary.json" || exit 1
fi

echo "[SIM] Done: $STATUS"
