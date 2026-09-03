#!/usr/bin/env bash
# NOESIS-Σ — Scripts/bootstrap.sh (Golden Edition)
# Bootstraps a fresh checkout:
#   - Creates/refreshes local venv
#   - Installs requirements (dev/prod profiles)
#   - Creates Runtime/{Config,Logs,Snapshots,Jobs,Backups,Cache,Sim}
#   - Optionally generates signing/encryption keys (safe, idempotent)
#   - Performs a smoke run + snapshot to verify Core wiring
#
# Usage:
#   Scripts/bootstrap.sh [--dev|--prod] [--with-keys] [--force-keys] [--reset-venv] [--python=3.11] [--no-smoke]
#   Scripts/bootstrap.sh --help
#
# Notes:
#   - Never commits keys. Keys are placed under Runtime/Config/keys with restrictive perms.
#   - Idempotent: safe to re-run. Combine --reset-venv to recreate the venv from scratch.

set -Eeuo pipefail

# ---------- styling ----------
if command -v tput >/dev/null 2>&1; then
  BOLD="$(tput bold)"; DIM="$(tput dim)"; RED="$(tput setaf 1)"; GREEN="$(tput setaf 2)"; YELLOW="$(tput setaf 3)"; BLUE="$(tput setaf 4)"; RESET="$(tput sgr0)"
else
  BOLD=""; DIM=""; RED=""; GREEN=""; YELLOW=""; BLUE=""; RESET=""
fi

info () { echo -e "${BLUE}ℹ${RESET} $*"; }
ok   () { echo -e "${GREEN}✔${RESET} $*"; }
warn () { echo -e "${YELLOW}⚠${RESET} $*"; }
err  () { echo -e "${RED}✖${RESET} $*" >&2; }
die  () { err "$*"; exit 1; }

trap 'err "Bootstrap failed at line $LINENO"; exit 1' ERR

# ---------- repo root detection ----------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

[[ -f "pyproject.toml" ]] || die "Run from a valid repo: pyproject.toml not found at ${REPO_ROOT}"

# ---------- defaults / flags ----------
PROFILE="dev"             # dev|prod
WITH_KEYS="no"            # yes|no
FORCE_KEYS="no"           # yes|no (allows overwrite)
RESET_VENV="no"           # yes|no
PY_REQ="3.11"             # preferred python major.minor
DO_SMOKE="yes"            # run smoke snapshot
PYBIN=""                  # resolved python binary

usage () {
  cat <<EOF
${BOLD}NOESIS-Σ Bootstrap${RESET}

${BOLD}Usage:${RESET}
  Scripts/bootstrap.sh [--dev|--prod] [--with-keys] [--force-keys] [--reset-venv] [--python=3.11] [--no-smoke]

${BOLD}Options:${RESET}
  --dev            Install development stack (default): requirements.txt + requirements-dev.txt + requirements-web.txt + requirements-train.txt (if present)
  --prod           Install production stack: requirements-prod.txt (+ requirements-web.txt if present) (+ requirements-gguf-encrypt.txt if present)
  --with-keys      Generate Ed25519 + AES keys under Runtime/Config/keys (safe; refuses if files exist unless --force-keys)
  --force-keys     Overwrite existing keys (use with care)
  --reset-venv     Remove and recreate .venv from scratch
  --python=V       Prefer specific python (e.g., 3.11)
  --no-smoke       Skip the smoke test snapshot
  --help           Show this help

${BOLD}What it does:${RESET}
  • Creates .venv and installs profile-specific requirements
  • Creates Runtime/{Config,Logs,Snapshots,Jobs,Backups,Cache,Sim}
  • Copies Runtime/Config/.env.template -> .env (if missing)
  • Optionally generates keys via Security/keygen.sh
  • Runs a smoke step + snapshot at Runtime/Snapshots/bootstrap_smoke/<ts>
EOF
}

# ---------- parse args ----------
for arg in "$@"; do
  case "$arg" in
    --dev) PROFILE="dev";;
    --prod) PROFILE="prod";;
    --with-keys) WITH_KEYS="yes";;
    --force-keys) FORCE_KEYS="yes";;
    --reset-venv) RESET_VENV="yes";;
    --python=*) PY_REQ="${arg#*=}";;
    --no-smoke) DO_SMOKE="no";;
    --help|-h) usage; exit 0;;
    *) die "Unknown option: $arg (see --help)";;
  esac
done

info "Repo: ${REPO_ROOT}"
info "Profile: ${PROFILE}"
info "With keys: ${WITH_KEYS} (force: ${FORCE_KEYS})"
info "Reset venv: ${RESET_VENV}"
info "Preferred Python: ${PY_REQ}"
info "Smoke snapshot: ${DO_SMOKE}"

# ---------- python resolution ----------
pick_python () {
  local req="$1"
  for c in "python${req}" "python${req%.*}" python3 python; do
    if command -v "$c" >/dev/null 2>&1; then
      echo "$c"; return 0
    fi
  done
  return 1
}

PYBIN="$(pick_python "${PY_REQ}")" || die "No suitable python found (wanted ~${PY_REQ}). Install Python ${PY_REQ}+."
# Version check (require >= 3.11)
VER_STR="$("$PYBIN" - <<'PY'
import sys
print(".".join(map(str, sys.version_info[:3])))
PY
)"
IFS=. read -r MAJ MIN PATCH <<<"${VER_STR}"
if (( MAJ < 3 || (MAJ == 3 && MIN < 11) )); then
  die "Python >= 3.11 required, found ${VER_STR}"
fi
ok "Using ${PYBIN} (Python ${VER_STR})"

# ---------- venv ----------
if [[ "${RESET_VENV}" == "yes" && -d ".venv" ]]; then
  warn "Removing existing .venv (--reset-venv)"
  rm -rf .venv
fi

if [[ ! -d ".venv" ]]; then
  info "Creating virtual environment at .venv"
  "${PYBIN}" -m venv .venv
fi
VENV_PY=".venv/bin/python"
VENV_PIP=".venv/bin/pip"
[[ -x "${VENV_PY}" ]] || die "Failed to create venv python at ${VENV_PY}"

# Upgrade pip tooling
"${VENV_PY}" -m pip install --upgrade pip setuptools wheel >/dev/null
ok "Venv ready"

# ---------- requirements ----------
install_req_if_present () {
  local file="$1"
  if [[ -f "$file" ]]; then
    info "Installing ${file}"
    "${VENV_PIP}" install -r "$file"
  else
    info "Skipping ${file} (not present)"
  fi
}

case "${PROFILE}" in
  dev)
    install_req_if_present "requirements.txt"
    install_req_if_present "requirements-dev.txt"
    install_req_if_present "requirements-web.txt"
    install_req_if_present "requirements-train.txt"
    ;;
  prod)
    install_req_if_present "requirements-prod.txt" || true
    # fall back to base if prod file absent
    if [[ ! -f "requirements-prod.txt" ]]; then
      install_req_if_present "requirements.txt"
    fi
    install_req_if_present "requirements-web.txt"
    install_req_if_present "requirements-gguf-encrypt.txt"
    ;;
  *) die "Unknown profile ${PROFILE}";;
esac
ok "Dependencies installed for ${PROFILE}"

# ---------- runtime dirs ----------
mk_runtime_dirs () {
  mkdir -p Runtime/Config/keys
  mkdir -p Runtime/Logs
  mkdir -p Runtime/Snapshots
  mkdir -p Runtime/Jobs
  mkdir -p Runtime/Backups
  mkdir -p Runtime/Cache
  mkdir -p Runtime/Sim
}
mk_runtime_dirs
ok "Runtime directories created"

# ---------- .env provisioning ----------
if [[ -f "Runtime/Config/.env" ]]; then
  info "Runtime/Config/.env exists (leaving as-is)"
else
  if [[ -f "Runtime/Config/.env.template" ]]; then
    cp -n Runtime/Config/.env.template Runtime/Config/.env
    chmod 600 Runtime/Config/.env || true
    ok "Seeded Runtime/Config/.env from template"
  else
    warn "Runtime/Config/.env.template missing — create it to enable env-based config"
  fi
fi

# ---------- keys generation (optional) ----------
maybe_make_keys () {
  local key_dir="Runtime/Config/keys"
  local aes="${key_dir}/model.aes.key"
  local priv="${key_dir}/ed25519_private.pem"
  local pub="${key_dir}/ed25519_public.pem"

  if [[ "${WITH_KEYS}" != "yes" ]]; then
    info "Skipping key generation (no --with-keys)"
    return 0
  fi

  if [[ -f "${aes}" || -f "${priv}" || -f "${pub}" ]]; then
    if [[ "${FORCE_KEYS}" == "yes" ]]; then
      warn "Overwriting existing keys (--force-keys)"
    else
      warn "Keys already exist; refusing to overwrite without --force-keys"
      return 0
    fi
  fi

  if [[ -x "Security/keygen.sh" ]]; then
    info "Generating keys via Security/keygen.sh"
    if [[ "${FORCE_KEYS}" == "yes" ]]; then
      FORCE_KEYS_FLAG="--force"
    else
      FORCE_KEYS_FLAG=""
    fi
    # Provide a non-interactive path: the script should be idempotent
    bash Security/keygen.sh ${FORCE_KEYS_FLAG}
  else
    warn "Security/keygen.sh not executable or missing; generating minimal keys here"
    # Fallback: minimal keygen (requires openssl + python cryptography)
    if ! command -v openssl >/dev/null 2>&1; then
      die "openssl is required to generate AES key"
    fi
    mkdir -p "${key_dir}"
    head -c 32 /dev/urandom > "${aes}"
    chmod 600 "${aes}"
    # Use python cryptography to make ed25519 keypair
    "${VENV_PY}" - <<'PY'
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
from pathlib import Path
key_dir = Path("Runtime/Config/keys")
key_dir.mkdir(parents=True, exist_ok=True)
sk = Ed25519PrivateKey.generate()
pk = sk.public_key()
priv_pem = sk.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption(),
)
pub_pem = pk.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo,
)
(Path(key_dir/"ed25519_private.pem")).write_bytes(priv_pem)
(Path(key_dir/"ed25519_public.pem")).write_bytes(pub_pem)
PY
    chmod 600 "${priv}" || true
    chmod 644 "${pub}" || true
  fi

  ok "Keys ready at Runtime/Config/keys (private 600, public 644)"
}
maybe_make_keys

# ---------- smoke test + snapshot ----------
if [[ "${DO_SMOKE}" == "yes" ]]; then
  info "Running smoke step + snapshot"
  SNAP_DIR="Runtime/Snapshots/bootstrap_smoke/$(date +%Y%m%dT%H%M%S)"
  mkdir -p "${SNAP_DIR}"
  "${VENV_PY}" - <<PY
import os, time, json, torch
from Core import load_params, OperatorSplitEngine
from Core.icnn import ICNN

p = load_params()
p.device = "cpu"
p.state_dim = int(os.getenv("NOESIS_STATE_DIM", "16"))
if p.icnn is None:
    p.icnn = ICNN(p.state_dim, [64, 32])

eng = OperatorSplitEngine(p)
x = torch.randn(p.state_dim)
y = eng.step(x)

snap_dir = "${SNAP_DIR}"
sign = bool(getattr(p, "snapshot_signing_private_pem", None) and getattr(p, "snapshot_signing_public_pem", None))
encrypt = bool(getattr(p, "snapshot_aes_key", None))
eng.save_snapshot(snap_dir, sign=sign, encrypt=encrypt)

print(json.dumps({
  "snapshot_dir": snap_dir,
  "signed": sign,
  "encrypted": encrypt,
  "step_count": eng.step_count
}))
PY
  ok "Smoke snapshot written to ${SNAP_DIR}"
else
  info "Skipping smoke snapshot (--no-smoke)"
fi

# ---------- final summary ----------
cat <<EOF
${GREEN}✔ Bootstrap complete${RESET}

Next steps:
  ${DIM}# activate venv${RESET}
  source .venv/bin/activate

  ${DIM}# run fast tests${RESET}
  pytest -q -m fast

  ${DIM}# optional: start API (after wiring)${RESET}
  uvicorn API.main:app --reload --host 127.0.0.1 --port 9000

Profile: ${PROFILE}
Keys: ${WITH_KEYS} (force=${FORCE_KEYS})
Venv: ${REPO_ROOT}/.venv
EOF
