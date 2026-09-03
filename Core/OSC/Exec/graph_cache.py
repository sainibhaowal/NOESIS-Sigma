# ================================================================
#  NOESIS-Σ — Golden Edition
#  GraphBucketManager: LRU cache of HotLoopGraph per (shape/profile)
# ================================================================
from collections import OrderedDict
from typing import Tuple

import torch

from Core.OSC.dynamics import HotLoopFused, HotLoopGraph
from Core.OSC.Exec.profiles import ProfileConfig

Key = Tuple[int, int, float, torch.dtype, torch.dtype, torch.dtype, bool, str]
# (B,S,dt,dtype,icnn_ws,k_ws,deterministic,profile_name)


class GraphBucketManager:
    def __init__(self, capacity: int = 8):
        self.capacity = capacity
        self._cache: OrderedDict[Key, HotLoopGraph] = OrderedDict()

    def get(
        self,
        fused: HotLoopFused,
        B: int,
        S: int,
        dt: float,
        dtype: torch.dtype,
        prof: ProfileConfig,
    ) -> HotLoopGraph:
        key: Key = (
            B,
            S,
            float(dt),
            dtype,
            fused.icnn._ws_dtype,
            fused.lrK.ws_dtype,
            bool(prof.deterministic),
            prof.name,
        )
        g = self._cache.get(key)
        if g is not None:
            self._cache.move_to_end(key)
            return g
        g = HotLoopGraph(fused, B=B, S=S, dt=dt)
        self._cache[key] = g
        while len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
        return g
