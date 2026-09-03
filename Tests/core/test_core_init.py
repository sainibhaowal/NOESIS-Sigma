# ================================================================
#  NOESIS-Σ — Golden Edition
#  Test: Core lazy exports are present
# ================================================================
def test_core_lazy_exports():
    import Core
    from Core import EngineParams, ICNNDirectGrad, OperatorSplitEngine, load_params

    # Functions/classes importable
    assert callable(load_params)
    assert EngineParams is not None
    assert OperatorSplitEngine is not None
    assert ICNNDirectGrad is not None

    # Symbols cached on Core root
    assert hasattr(Core, "EngineParams")
    assert hasattr(Core, "OperatorSplitEngine")
    assert hasattr(Core, "ICNNDirectGrad")
    assert hasattr(Core, "ICNN")          # back-compat alias exists
    assert hasattr(Core, "load_params")
