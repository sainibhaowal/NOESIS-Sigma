# Pytest logging setup for NOESIS-Σ
import os
import sys
from pathlib import Path

import pytest
from loguru import logger

os.environ.setdefault("NOESIS_IGNORE_ENV", "1")
os.environ.setdefault("NOESIS_PROFILE", "AUTO")
os.environ.setdefault("SIM_ALLOW_SQLITE_FOR_TESTS", "1")

_SLOW_PATTERNS = (
    "soak",
    "latency_curve",
    "vram_flatness_curve",
    "hot_bounded_under_heavy_write_load",
)

# Skip legacy tests that depend on removed modules.
_LEGACY_BASENAME_MATCH = (
    "test_long_task_",
    "test_planner_",
    "test_router_",
    "test_c0_",
    "test_c1_",
    "test_checkpoint_",
    "test_chunk_state_machine_transitions",
    "test_resume_finds_latest_valid_checkpoint",
    "test_runner_idempotency",
    "test_artifact_hashing_manifest",
    "test_strict_chunk_receipt_binding",
)


def _basename(path) -> str:
    value = getattr(path, "basename", None)
    if value is None:
        value = getattr(path, "name", "")
    return str(value)


def _ext(path) -> str:
    value = getattr(path, "ext", None)
    if value is None:
        value = getattr(path, "suffix", "")
    return str(value)


def pytest_ignore_collect(collection_path, config):
    if not _basename(collection_path).startswith("test_") or _ext(collection_path) != ".py":
        return False
    name = _basename(collection_path)
    return any(name.startswith(p) for p in _LEGACY_BASENAME_MATCH)

@pytest.fixture(scope="session", autouse=True)
def _configure_logging():
    tmp_dir = os.environ.get("TMPDIR", "Runtime/Logs")
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    logger.remove()
    log_file = os.path.join(tmp_dir, "noesis.log")
    logger.add(log_file, level="DEBUG", rotation="10 MB", enqueue=True)
    logger.add(sys.stderr, level="ERROR")  # tests: only errors to console


def pytest_collection_modifyitems(config, items):
    run_nightly = os.getenv("NOESIS_RUN_NIGHTLY", "0").strip().lower() in {"1", "true", "yes"}
    if run_nightly:
        return
    skip_slow = pytest.mark.skip(reason="slow/nightly test (set NOESIS_RUN_NIGHTLY=1 to run)")
    for item in items:
        nodeid = item.nodeid
        if any(pat in nodeid for pat in _SLOW_PATTERNS):
            item.add_marker(pytest.mark.nightly)
            item.add_marker(skip_slow)
