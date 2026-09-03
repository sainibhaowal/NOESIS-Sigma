from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Dict, Tuple

__all__ = [
    "EngineParams",
    "OperatorSplitEngine",
    "HotLoopFused",
    "ICNNDirectGrad",
    "ICNN",
    "load_params",
    "__version__",
]

if TYPE_CHECKING:
    from Core.OSC.dynamics import EngineParams, HotLoopFused, OperatorSplitEngine
    from Core.OSC.icnn import ICNNDirectGrad
    from Core.OSC.params import load_params

try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("noesis-sigma")
except Exception:
    __version__ = "0.0.0"

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "EngineParams": ("Core.OSC.dynamics", "EngineParams"),
    "OperatorSplitEngine": ("Core.OSC.dynamics", "OperatorSplitEngine"),
    "HotLoopFused": ("Core.OSC.dynamics", "HotLoopFused"),
    "ICNNDirectGrad": ("Core.OSC.icnn", "ICNNDirectGrad"),
    "ICNN": ("Core.OSC.icnn", "ICNNDirectGrad"),
    "load_params": ("Core.OSC.params", "load_params"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'Core' has no attribute '{name}'")
    module_path, attr_name = target
    module = importlib.import_module(module_path)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> Any:
    return sorted(set(list(globals().keys()) + list(_LAZY_EXPORTS.keys())))
