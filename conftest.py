# Ensure project root is on sys.path for tests, no matter where pytest starts.
import os
import sys
from typing import Iterable

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# Ignore legacy test modules that depend on removed packages.
_LEGACY_IMPORT_MARKERS: Iterable[str] = (
    "Core.LongTask",
    "Core.Planner",
    "External.Routing",
    "Core.Continuity",
    "Core.Reconstruction",
)


def pytest_ignore_collect(collection_path, config):
    if not collection_path.name.startswith("test_") or not collection_path.suffix == ".py":
        return False
    try:
        if hasattr(collection_path, "read_text"):
            content = collection_path.read_text(encoding="utf-8", errors="ignore")
        else:
            content = collection_path.read().decode("utf-8", errors="ignore")
    except Exception:
        return False
    return any(marker in content for marker in _LEGACY_IMPORT_MARKERS)
