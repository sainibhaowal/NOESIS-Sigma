#!/usr/bin/env bash
set -euo pipefail

export PYTHONHASHSEED=0
export TZ=UTC

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTEST_BIN="${PYTEST_BIN:-pytest}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

status=0

run_suite() {
  local name="$1"
  local path="$2"

  if [ ! -e "$path" ]; then
    echo "SKIP $name (missing: $path)"
    return 0
  fi

  echo "RUN $name"
  if ! $PYTEST_BIN -q "$path"; then
    echo "FAIL $name"
    status=1
  else
    echo "PASS $name"
  fi
}

run_suite "unit" "$ROOT/Tests/Unit"
run_suite "integration" "$ROOT/Tests/Sim"
run_suite "acceptance" "$ROOT/Tests/Acceptance"

if [ $status -eq 0 ]; then
  echo "VERIFY PASS"
else
  echo "VERIFY FAIL"
fi

exit $status
