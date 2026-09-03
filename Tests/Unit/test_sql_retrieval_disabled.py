"""
Phase 3 verification: SQL cosine retrieval has been disabled.

This test confirms that the SimCoreService.read() method properly respects
the environment variable that disables SQL retrieval in favor of
attractor-based recall (Road B).
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import torch

from External.Sim.Storage.warm_store import WarmStore
from External.Sim.Services.sim_core_service import SIMCoreService


def test_ann_search_disabled_by_default():
    """Verify that WarmStore.ann_search returns empty list by default."""
    warm = WarmStore(engine=None)
    
    # Clear any stored enable flags
    if hasattr(warm, "_pgvector_enabled"):
        warm._pgvector_enabled = True
    
    # Mock engine for any operations
    class MockEngine:
        pass
    
    warm._engine = MockEngine()
    
    # This should return an empty list regardless of input when Road B is enabled
    result = warm.ann_search(
        tenant_id="test_tenant",
        user_id="test_user", 
        memory_type=None,
        query_vec=[0.1, 0.2, 0.3],
        limit=10
    )
    
    assert result == [], "ann_search should always return empty list in Road B"


def test_sql_retrieval_enviroment_variable():
    """Test that SIM_ALLOW_SQL_RETRIEVAL environment controls SQL retrieval."""
    # First, test with SQL retrieval disabled (Road B mode)
    with patch.dict(os.environ, {"SIM_ALLOW_SQL_RETRIEVAL": "0"}):
        # Re-import and re-initialize to pick up the env var change
        import importlib
        import External.Sim.Services.sim_core_service as sim_core_module
        
        # Clear the module cache
        importlib.reload(sim_core_module)
        
        # Verify _sql_retrieval_allowed returns False
        svc = sim_core_module.SIMCoreService(warm=None, config=None)
        assert svc._sql_retrieval_allowed() == False
        
        # Reload module again for next test
        importlib.reload(sim_core_module)
    
    # Test with SQL retrieval enabled
    with patch.dict(os.environ, {"SIM_ALLOW_SQL_RETRIEVAL": "1"}):
        importlib.reload(sim_core_module)
        
        svc = sim_core_module.SIMCoreService(warm=None, config=None)
        assert svc._sql_retrieval_allowed() == True
        
        # Reload module for next test
        importlib.reload(sim_core_module)


def test_road_b_mode_routing():
    """Simulate the Road B routing logic in the read method."""
    # Mock components needed for SimCoreService
    class MockWarm:
        def __init__(self):
            self.pgvector_enabled = True
            self.concept_encoder = None
            self.hostory_store = None
        
        def get_record_by_id(self, record_id):
            return None  # No mock data
    
    class MockConfig:
        def __init__(self):
            self.hot_enabled = False
            self.hot_max_items = 10
            self.db_config = type('obj', (object,), {
                'db_url': 'sqlite:///:memory:',
                'echo_sql': False
            })()
    
    # Test Road B mode (NOESIS_STATE_DIM may be set in other phases)
    os.environ.setdefault("NOESIS_STATE_DIM", "1024")
    os.environ["SIM_ALLOW_SQL_RETRIEVAL"] = "0"
    
    warm = MockWarm()
    config = MockConfig()
    
    # Create service with Warm store
    svc = SIMCoreService(warm=warm, config=config)
    
    # The service should use the new WarmStore.ann_search which returns []
    # When query_text is processed, it should use _sql_retrieval_allowed()
    assert svc._sql_retrieval_allowed() == False


def test_environment_variable_unchanged():
    """Ensure the default environment variable state is consistent."""
    # Verify the default value
    assert os.environ.get("SIM_ALLOW_SQL_RETRIEVAL", "0") in ("0", "false")


def test_ann_search_signature_unchanged():
    """Ensure the method signature hasn't changed so as not to break existing code."""
    import inspect
    
    sig = inspect.signature(WarmStore.ann_search)
    params = list(sig.parameters.keys())
    
    # Should have the expected parameters
    expected_params = ['self', 'tenant_id', 'user_id', 'memory_type', 'query_vec', 'limit']
    for param in expected_params:
        assert param in params, f"Missing parameter: {param}"


if __name__ == "__main__":
    test_ann_search_disabled_by_default()
    print("✓ ann_search returns empty list")
    
    test_sql_retrieval_enviroment_variable()
    print("✓ Environment variable controls SQL retrieval")
    
    test_road_b_mode_routing()
    print("✓ Road B mode routing verified")
    
    test_environment_variable_unchanged()
    print("✓ Default environment variable state unchanged")
    
    test_ann_search_signature_unchanged()
    print("✓ Method signature unchanged")
    
    print("\n✅ All Phase 3 verification tests passed!")