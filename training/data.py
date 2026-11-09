"""Memory-mapped token shards with a resumable cursor.

Reads the same format the NOESIS corpus builder already produces, so the 200M-token corpus can be
used without re-tokenising::

    {"dtype": "uint32", "eos_token_id": 2,
     "shards": [{"path": "shard-00000.bin", "tokens": 50000000}, ...]}

Documents are separated by an EOS token, so every sampled token is a real training token and no
padding mask is needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class ShardError(RuntimeError):
    pass


class PackedShards:
    def __init__(self, root: str | Path, rank: int = 0, world_size: int = 1, seed: int = 0) -> None:
        self.root = Path(root)
        meta_path = self.root / "metadata.json"
        if not meta_path.exists():
            raise ShardError(f"{meta_path} not found; build the corpus shards first")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.dtype = np.dtype(meta.get("dtype", "uint32"))
        self.eos_token_id = int(meta.get("eos_token_id", 2))
        entries = meta.get("shards") or []
        if not entries:
            raise ShardError("metadata.json lists no shards")
        self.paths: list[Path] = [self.root / item["path"] for item in entries]
        self.lengths = [int(item["tokens"]) for item in entries]
        for path in self.paths:
            if not path.exists():
                raise ShardError(f"{path} is listed but missing")
        if world_size > len(self.paths):
            raise ShardError("more ranks than shards; build more shards")

        self.rank = rank
        self.world_size = world_size
        self.owned = list(range(rank, len(self.paths), world_size))
        self.rng = np.random.default_rng(seed + rank)
        self.cursor = [0, 0]  # [index into owned, token offset]
        # Tokens consumed since the start. Every rank consumes the same number per step, so this
        # scalar — not the per-rank cursor — is what a checkpoint has to carry: rank 0 writes the
        # checkpoint but every rank must resume at its own position.
        self.consumed = 0
        self._maps: dict[int, np.memmap] = {}

    # ------------------------------------------------------------------ io

    def _map(self, shard: int) -> np.memmap:
        if shard not in self._maps:
            self._maps[shard] = np.memmap(self.paths[shard], dtype=self.dtype, mode="r")
        return self._maps[shard]

    def _take(self, count: int) -> np.ndarray:
        pieces = []
        filled = 0
        while filled < count:
            owned_index, offset = self.cursor
            shard = self.owned[owned_index % len(self.owned)]
            available = self.lengths[shard] - offset
            if available <= 0:
                self.cursor = [(owned_index + 1) % len(self.owned), 0]
                continue
            take = min(count - filled, available)
            pieces.append(np.asarray(self._map(shard)[offset : offset + take], dtype=np.int64))
            filled += take
            offset += take
            self.cursor = [owned_index, offset]
            if offset >= self.lengths[shard]:
                self.cursor = [owned_index + 1, 0]
        self.consumed += count
        return np.concatenate(pieces)

    # -------------------------------------------------------------- public

    def sample(self, batch: int, length: int) -> np.ndarray:
        """``[batch, length]`` int64 token windows, consumed sequentially from the cursor."""
        return self._take(batch * length).reshape(batch, length)

    def state_dict(self) -> dict:
        return {"tokens": self.consumed}

    def _cursor_after(self, tokens: int) -> list[int]:
        index, offset, remaining = 0, 0, max(int(tokens), 0)
        while remaining > 0:
            shard = self.owned[index % len(self.owned)]
            take = min(remaining, self.lengths[shard] - offset)
            remaining -= take
            offset += take
            if offset >= self.lengths[shard]:
                index += 1
                offset = 0
        return [index, offset]

    def load_state_dict(self, state: dict) -> None:
        if "tokens" in state:
            self.consumed = int(state["tokens"])
            self.cursor = self._cursor_after(self.consumed)
            return
        # Legacy checkpoints stored one rank's cursor. Convert it to a token count against that
        # rank's shard ownership, then re-derive the position for this rank.
        index, offset = state["cursor"]
        owned = list(range(int(state.get("rank", 0)), len(self.paths), self.world_size))
        self.consumed = int(sum(self.lengths[shard] for shard in owned[: int(index)]) + int(offset))
        self.cursor = self._cursor_after(self.consumed)

    def __len__(self) -> int:
        return sum(self.lengths[index] for index in self.owned)
