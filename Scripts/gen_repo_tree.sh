#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# NOESIS-Σ — repo tree dumper (fixed)
# Writes a pretty ascii tree of the repo to OUT (default: Noesis_tree.txt)
# Excludes: any ".venv" directories and any "__pycache__" directories.
# -----------------------------------------------------------------------------
set -euo pipefail

# === CONFIG ===
REPO_ROOT="/home/sephi-asi/NOESIS-Σ"           # adjust if needed
OUT="$REPO_ROOT/Noesis_tree.txt"
# === END CONFIG ===

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found in PATH." >&2
  exit 2
fi

cd "$REPO_ROOT" || { echo "ERROR: cannot cd to $REPO_ROOT" >&2; exit 1; }

rm -f "$OUT"

# create a secure temp file for the path list
NOESIS_TMP=$(mktemp /tmp/noesis_paths.XXXXXX) || { echo "mktemp failed"; exit 1; }
# export so python can read it via os.environ
export NOESIS_TMP
trap 'rm -f "$NOESIS_TMP"' EXIT

# Generate a sorted list of paths while pruning any .venv and __pycache__ directories
LC_ALL=C find . \
  \( -name '.venv' -type d -prune \) -o \
  \( -name '__pycache__' -type d -prune \) -o \
  \( -name '.git' -type d -prune \) -o \
  \( -name 'Web' -type d -prune \) -o \
  -print \
| sed 's#^\./##' \
| sort \
> "$NOESIS_TMP"

# Use python to build tree; read temp path from environment variable NOESIS_TMP
python3 - <<'PY' > "$OUT"
import os
from pathlib import Path

tmp = os.environ.get("NOESIS_TMP")
if not tmp:
    raise SystemExit("ERROR: NOESIS_TMP not set in environment")
tmp_path = Path(tmp)
lines = [l.rstrip() for l in tmp_path.read_text(encoding="utf-8").splitlines() if l.rstrip()]
tree = {}
for path in lines:
    # Avoid adding the output file itself (if inside repo)
    if path == Path(os.environ.get("REPO_OUT_NAME", "")).name:
        continue
    if path == ".":
        continue
    parts = path.split('/')
    cur = tree
    for part in parts:
        cur = cur.setdefault(part, {})
def walk(d, prefix=''):
    keys = sorted(d.keys())
    for i,k in enumerate(keys):
        last = (i == len(keys)-1)
        connector = '└── ' if last else '├── '
        print(prefix + connector + k)
        newpref = prefix + ('    ' if last else '│   ')
        walk(d[k], newpref)
print("NOESIS/")
walk(tree)
PY

echo "Wrote $OUT"
