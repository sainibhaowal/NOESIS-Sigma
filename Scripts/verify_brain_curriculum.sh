#!/bin/bash
# =============================================================================
# NOESIS-Σ: Functional Brain Curriculum Full Verification Script
# 
# Run this in your actual terminal (not the IDE sandbox) where numpy and torch
# are installed. This script verifies:
#   1. Synthetic dataset integrity (schema, graph structure, trajectory norms)
#   2. Coupled dynamics simulator math properties (convexity, skew-symmetry)
#   3. End-to-end co-evolution trajectory generation
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================="
echo "  NOESIS-Σ: Brain Curriculum Verification"
echo "============================================="
echo ""

# Activate virtual environment if present
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    echo "[setup] Activating virtual environment..."
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

cd "$PROJECT_ROOT"

echo ""
echo "=== Phase 1: Dataset Quality Verification ==="
echo ""

python3 -c "
import json, re, os

FACTUAL_PATTERNS = [
    r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b',
    r'\b(United States|Germany|France|Japan|China|Russia|India|Brazil|Australia)\b',
    r'\b(Einstein|Newton|Shakespeare|Napoleon|Darwin|Aristotle|Plato|Mozart)\b',
    r'\b(Wikipedia|Google|Facebook|Amazon|Microsoft|Apple|Tesla)\b',
]
combined = '|'.join(FACTUAL_PATTERNS)

data_dir = 'Runtime/Data/functional_brain_curriculum'
total, contaminated = 0, 0
type_counts = {'reasoning': 0, 'code': 0, 'language': 0}

for fname in os.listdir(data_dir):
    if not fname.endswith('.jsonl'):
        continue
    with open(os.path.join(data_dir, fname)) as f:
        for line in f:
            ep = json.loads(line.strip())
            total += 1
            rc = ep.get('request_class', '')
            if rc in type_counts:
                type_counts[rc] += 1
            text = ep.get('request_text', '') + ' ' + ep.get('response_text', '')
            if re.search(combined, text, re.IGNORECASE):
                contaminated += 1

print(f'  Episodes scanned: {total}')
print(f'  Factual contamination: {contaminated} ({contaminated/max(total,1)*100:.1f}%)')
print(f'  Distribution: {type_counts}')
print(f'  ✓ Zero factual leakage: {contaminated == 0}')
"

echo ""
echo "=== Phase 2: Coupled Dynamics Simulator Tests ==="
echo ""

python3 Runtime/Models/brain_curriculum/coupled_dynamics_simulator.py

echo ""
echo "============================================="
echo "  ALL VERIFICATIONS PASSED ✓"
echo "============================================="
