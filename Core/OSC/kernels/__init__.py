# NOESIS-Σ Core.kernels package
from .placeholder import apply_K_dense as apply_K_dense
from .placeholder import apply_K_lowrank as apply_K_lowrank

__all__ = ["apply_K_dense", "apply_K_lowrank"]
