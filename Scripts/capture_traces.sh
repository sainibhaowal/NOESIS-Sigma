#!/usr/bin/env bash
set -euo pipefail

API_URL="${API_URL:-http://127.0.0.1:9000}"
TENANT_ID="${TENANT_ID:-default}"
USER_ID="${USER_ID:-golden}"

TS="$(date +%Y%m%dT%H%M%SZ)"
OUT_DIR="Runtime/Logs/golden_traces/${TS}"
mkdir -p "$OUT_DIR"

summary_file="$OUT_DIR/summary.txt"

echo "golden_traces run: $TS" > "$summary_file"
echo "api_url: $API_URL" >> "$summary_file"
echo "tenant_id: $TENANT_ID" >> "$summary_file"
echo "user_id: $USER_ID" >> "$summary_file"

touch "$OUT_DIR/_run_errors.log"

post_decode() {
  local name="$1"
  local text="$2"
  local profile="$3"
  local extra_json="$4"

  local req_file="$OUT_DIR/${name}.request.json"
  local out_file="$OUT_DIR/${name}.response.json"

  cat > "$req_file" <<JSON
{
  "text": "${text}",
  "profile": "${profile}",
  "manager_mode": false,
  "max_tokens": 256,
  "chunk_max_tokens": 600,
  "manager_max_steps": 6${extra_json}
}
JSON

  if ! curl -s --max-time 8 -X POST "$API_URL/decoder/decode" \
    -H 'Content-Type: application/json' \
    -H "X-Tenant-Id: $TENANT_ID" \
    -H "X-User-Id: $USER_ID" \
    -d @"$req_file" > "$out_file"; then
    echo "${name}: request failed" >> "$OUT_DIR/_run_errors.log"
  fi

  echo "- ${name}: profile=${profile}" >> "$summary_file"
}

# 3 FAST
post_decode "fast_01" "Summarize the causes of climate change in 3 bullets." "FAST" ""
post_decode "fast_02" "Extract key dates from: 'The project started in 2019 and shipped in 2023.'" "FAST" ""
post_decode "fast_03" "Draft a 2-sentence email declining a meeting." "FAST" ""

# 3 BALANCED
post_decode "balanced_01" "Explain photosynthesis to a 12-year-old." "BALANCED" ""
post_decode "balanced_02" "Give a short pros/cons list for electric cars." "BALANCED" ""
post_decode "balanced_03" "Summarize the book '1984' in 5 bullets." "BALANCED" ""

# 3 STRICT
post_decode "strict_01" "Write a 5-step outline for a database backup policy." "STRICT" ""
post_decode "strict_02" "List 6 risks in a microservices architecture." "STRICT" ""
post_decode "strict_03" "Draft a brief incident report template." "STRICT" ""

# Multi-turn continuity test (2-step)
post_decode "continuity_01" "Remember that my favorite color is teal. Confirm stored." "BALANCED" ""
post_decode "continuity_02" "What is my favorite color?" "BALANCED" ""

echo "errors: $(wc -l < "$OUT_DIR/_run_errors.log") (see _run_errors.log)" >> "$summary_file"
