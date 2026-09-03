# Core/__init__.py
# ─────────────────────────────────────────────────────────────────────────────
# NOESIS-Σ — Core package marker with lazy exports (Golden Edition)
#
# Public API (lazy-loaded on first access):
#   - EngineParams, OperatorSplitEngine, HotLoopFused   (from Core.OSC.dynamics)
#   - ICNNDirectGrad, ICNN (alias → ICNNDirectGrad)     (from Core.OSC.icnn)
#   - load_params                                       (from Core.OSC.params)
#
# Design:
#   • Avoid importing heavy deps (e.g., torch) at package import time.
#   • Use PEP 562 (__getattr__) to load symbols on-demand.
#   • Provide __dir__ for better IDE discovery.
#
# Acceptance smoke:
#   >>> import Core
#   >>> from Core import OperatorSplitEngine, EngineParams, HotLoopFused
#   >>> from Core import ICNN, ICNNDirectGrad, load_params
#   >>> isinstance(ICNN, type) and isinstance(ICNNDirectGrad, type)
#
# Versioning:
#   • If installed as a package, __version__ comes from distribution metadata.
#   • Otherwise, defaults to "0.0.0" for source-tree usage.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import importlib
from typing import Any, Dict, Tuple

__all__ = [
    # Engine & params
    "EngineParams",
    "OperatorSplitEngine",
    "HotLoopFused",
    # ICNNs
    "ICNNDirectGrad",
    "ICNN",  # alias for back-compat → ICNNDirectGrad
    # Params loader
    "load_params",
    # Version
    "__version__",
    # Procedure system (May 2026)
    "ProcedureAttractor",
    "ProcedureStep",
    "ProcedureMetadata",
    "ProcedureRegistry",
    "ProcedureStateEncoder",
    "ProceduralCarver",
    "ProcedureComposer",
    "ProcedureGraphExtractor",
]

# Try to expose a package version if installed; fall back for source tree usage.
try:
    try:
        # Python ≥ 3.8
        from importlib.metadata import version as _pkg_version

        __version__ = _pkg_version("noesis-sigma")
    except Exception:  # pragma: no cover
        # Python < 3.8 backport
        from importlib_metadata import version as _pkg_version_legacy

        __version__ = _pkg_version_legacy("noesis-sigma")
except Exception:  # not installed as a package
    __version__ = "0.0.0"

# Map exported names → (module_path, attribute_name)
# Notes:
#  • ICNN maps to ICNNDirectGrad for back-compat with older test/code paths.
#  • All heavy modules are only imported on first access.
_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    # Engine & params
    "EngineParams": ("Core.OSC.dynamics", "EngineParams"),
    "OperatorSplitEngine": ("Core.OSC.dynamics", "OperatorSplitEngine"),
    "HotLoopFused": ("Core.OSC.dynamics", "HotLoopFused"),
    # ICNNs
    "ICNNDirectGrad": ("Core.OSC.icnn", "ICNNDirectGrad"),
    "ICNN": ("Core.OSC.icnn", "ICNNDirectGrad"),  # alias
    # Params loader
    "load_params": ("Core.OSC.params", "load_params"),
    # Procedure system (May 2026)
    "ProcedureAttractor": ("Core.OSC.procedure_attractor", "ProcedureAttractor"),
    "ProcedureStep": ("Core.OSC.procedure_attractor", "ProcedureStep"),
    "ProcedureMetadata": ("Core.OSC.procedure_attractor", "ProcedureMetadata"),
    "ProcedureRegistry": ("Core.OSC.procedure_attractor", "ProcedureRegistry"),
    "ProcedureStateEncoder": ("Core.OSC.procedure_attractor", "ProcedureStateEncoder"),
    "ProceduralCarver": ("Core.OSC.procedure_carver", "ProceduralCarver"),
    "ProcedureComposer": ("Core.OSC.composition_engine", "ProcedureComposer"),
    "ProcedureGraphExtractor": ("Core.Cognition.procedure_extractor", "HybridProcedureExtractor"),
}


def __getattr__(name: str) -> Any:  # PEP 562
    """
    Lazy attribute loader for Core.* public API.

    On first access:
      1) Imports the target module (e.g., Core.dynamics).
      2) Fetches the requested attribute.
      3) Caches it in module globals to avoid repeat import/lookup cost.

    Raises:
      AttributeError: if the requested symbol is not part of the public API.
    """
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'Core' has no attribute '{name}'")
    module_path, attr_name = target
    module = importlib.import_module(module_path)
    try:
        value = getattr(module, attr_name)
    except AttributeError as e:  # pragma: no cover
        raise AttributeError(f"'{module_path}' has no attribute '{attr_name}'") from e
    globals()[name] = value  # cache for subsequent lookups
    return value


def __dir__() -> Any:
    """
    Enhance IDE/tab-completion by including lazily exported names.
    """
    return sorted(set(list(globals().keys()) + list(_LAZY_EXPORTS.keys())))
