"""SigmaStorm — a non-attention sequence architecture built on the Sigma transition law."""

from .config import SigmaConfig
from .model import SigmaStorm, chunked_cross_entropy
from .state import BlockState, SigmaState, SlotMemoryState

__all__ = [
    "SigmaConfig",
    "SigmaStorm",
    "SigmaState",
    "BlockState",
    "SlotMemoryState",
    "chunked_cross_entropy",
]
