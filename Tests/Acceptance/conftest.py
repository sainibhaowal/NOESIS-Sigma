"""
Tests/Acceptance/conftest.py

Fixtures and mocks for API acceptance tests.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest


# Mock external dependencies that may not be installed in test environment
# Create a recursive MagicMock that handles submodule access
class MockModule(MagicMock):
    def __getattr__(self, name):
        if name in ("__file__", "__loader__", "__spec__"):
            return None
        return super().__getattr__(name)


# Register mocks before any imports
_MOCK_MODULES = [
    "qdrant_client",
    "qdrant_client.http",
    "qdrant_client.http.models",
    "qdrant_client.models",
    "qdrant_client.models.distance",
    "qdrant_client.models.vector_params",
]

for module_name in _MOCK_MODULES:
    sys.modules[module_name] = MockModule()


# Create temporary directories for tests
_temp_dirs = {}


@pytest.fixture(scope="session", autouse=True)
def setup_test_directories():
    """Set up temporary directories for checkpoint/audit testing."""
    # Create temp directories for this test session
    test_temp = tempfile.mkdtemp(prefix="noesis_test_")
    
    checkpoint_root = Path(test_temp) / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    
    audit_root = Path(test_temp) / "audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    
    _temp_dirs["checkpoint_root"] = checkpoint_root
    _temp_dirs["audit_root"] = audit_root
    _temp_dirs["test_temp"] = test_temp
    
    # Patch environment to use temp directories
    os.environ["CHECKPOINT_ROOT"] = str(checkpoint_root)
    os.environ["AUDIT_ROOT"] = str(audit_root)
    
    yield
    
    # Cleanup after all tests
    if _temp_dirs.get("test_temp"):
        shutil.rmtree(_temp_dirs["test_temp"], ignore_errors=True)


@pytest.fixture
def admin_key(monkeypatch):
    """Set admin API key for testing."""
    test_key = "test_admin_key_12345"
    monkeypatch.setenv("NOESIS_ADMIN_API_KEY", test_key)
    return test_key


@pytest.fixture
def create_test_checkpoint(monkeypatch):
    """Create a test checkpoint directory."""
    def _create(checkpoint_id: str = "cp_test_001"):
        checkpoint_root = _temp_dirs.get("checkpoint_root")
        if not checkpoint_root:
            checkpoint_root = Path(tempfile.mkdtemp(prefix="checkpoints_"))
        
        checkpoint_dir = checkpoint_root / checkpoint_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create state file
        state_file = checkpoint_dir / "state.json"
        state_file.write_text('{"state": "checkpoint"}')
        
        return checkpoint_dir
    
    return _create
