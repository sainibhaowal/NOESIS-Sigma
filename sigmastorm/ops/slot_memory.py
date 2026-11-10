"""Bounded slot memory — the long-range ``M_t`` of the Sigma state.

The delta rule forgets on a timescale set by its decay spectrum; the slot memory is a
long-horizon prototype path. A fixed number of differentiable slots hold prototypes of the keys
the stream has seen and the values written to them.

Writes are per token (Gap D): every position produces a novelty score (distance to the closest
prototype), an evidence score (a learned confidence from the token and the residual stream) and a
soft assignment over slots. Gated by ``novelty * evidence``, the token contributes to the
accumulator of every slot it is assigned to. Two passes compute the novelty: the first scores a
token against the slots as consolidated at the start of the block; its gated writes are folded into
a causal running prototype so the second pass also compares each token with what was written
*before it inside the same block*. Without that, every copy of a repeated filler measures its
novelty against a stale prototype, floods the accumulator, and erases the distinct tokens written
next to it. The write routing also subtracts a penalty on the causal write mass already carried by
a slot, so similar tokens do not all pile into one argmax slot. An optional novelty dead-zone maps
near-duplicates to exactly zero: without it a repeated filler with novelty 0.02 still writes a
fraction of a pseudo-observation per token, and a few hundred repetitions sum past one and overwrite
the slot with the filler mean.

The read is a mixture over the slots plus, optionally, a null (background) slot whose value is the
zero vector. Mass routed to the null slot removes the read, so a query whose address matches no
stored prototype is answered by the residual stream alone instead of by an arbitrary slot. Every
``slot_block`` tokens the
accumulators are consolidated into a bounded soft update::

    D_j    = sum_t gate_t * a_tj
    eta_j  = D_j / (D_j + 1)
    mean_v = (sum_t gate_t a_tj v_t) / D_j
    V_j   <- (1 - eta_j) V_j + eta_j mean_v
    K_j   <- normalize((1 - eta_j) K_j + eta_j normalize(mean_k))
    use_j <- (1 - eta_j) use_j + eta_j

eta is the Bayesian weight of D_j pseudo-observations, so a single confident token moves a
slot a little and a full block of agreeing tokens replaces it almost entirely. The accumulator, the
token counter and the ring state are all carried in SlotMemoryState; this is the state-carrying
condition used by the paper's split-equivalence gate. The current gate covers one split, not every
possible chunk boundary, length, or backend.

Cost is ``O(T * slots * width)``: linear in the sequence, constant in memory.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..state import SlotMemoryState


class SlotMemory(nn.Module):
    def __init__(
        self,
        width: int,
        slots: int = 32,
        temperature: float = 0.5,
        block: int = 64,
        load_penalty: float = 1.0,
        metric: str = "cosine",
        shrinkage: float = 0.05,
        novelty_floor: float = 0.0,
        read_temperature: float | None = None,
        abstain: bool = False,
    ) -> None:
        super().__init__()
        if slots < 1 or block < 1:
            raise ValueError("slots and block must be positive")
        if metric not in ("cosine", "whitened"):
            raise ValueError(f"unknown slot metric {metric!r}")
        if not 0.0 <= novelty_floor < 1.0:
            raise ValueError("novelty_floor must be in [0, 1)")
        if read_temperature is not None and read_temperature <= 0.0:
            raise ValueError("read_temperature must be positive")
        self.slots = slots
        self.temperature = temperature
        self.block = block
        self.metric = metric
        self.shrinkage = shrinkage
        # Dead-zone on the write gate. A token below the floor has novelty rescaled to exactly 0,
        # so repeated filler cannot accumulate pseudo-observations across blocks and overwrite the
        # symbols it surrounds. The gradient is zero below the floor; above it, novelty is
        # linearly rescaled by 1 - novelty_floor before the evidence gate.
        self.novelty_floor = novelty_floor
        # The read has its own temperature: routing wants a sharp assignment, but a read that is
        # sharp before the addresses are trained selects an arbitrary slot.
        self.read_temperature = temperature if read_temperature is None else read_temperature
        # Match-confidence abstention: the read is multiplied by sigmoid(k * (best_match - t)) over
        # the occupied slots. A query that matches no stored prototype is answered by the residual
        # stream alone instead of by a softmax that must still hand out unit mass to some slot. The
        # gate is a per-token likelihood ratio; k and t are learned.
        self.abstain = abstain
        if abstain:
            self.abstain_threshold = nn.Parameter(torch.tensor(0.4))
            self.abstain_sharpness = nn.Parameter(torch.tensor(10.0))
        # Routing a token to an already-loaded slot is penalised in proportion to the causal write
        # mass accumulated there. With sharp routing and similar token keys the softmax otherwise
        # picks the same argmax slot for every token (measured: 1-2 of 16 slots written), so
        # distinct symbols overwrite each other and recall is chance.
        self.load_penalty = load_penalty
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.readout = nn.Linear(width, width, bias=False)
        self.evidence = nn.Sequential(nn.Linear(width * 2, width // 4), nn.SiLU(), nn.Linear(width // 4, 1))
        # Learned slot prototypes. Starting every slot from a zero key makes the first write scores
        # identical, so every slot receives the same mean key and the ring degenerates into one
        # prototype duplicated (measured: off-diagonal key cosine 0.94, read entropy pinned near
        # uniform). The prototypes give each slot a distinct starting address; the name is kept for
        # checkpoint compatibility.
        self.slot_bias = nn.Parameter(torch.randn(slots, width))
        # Write routing is sharper than read routing: assigning a token to its nearest prototype is
        # what makes the consolidated keys distinct. A read-temperature softmax over 100+ dims is
        # within a few percent of uniform and cannot separate the slots on its own.
        self.write_temperature = max(temperature * 0.2, 1e-2)
        # Detached read distribution of the last call, for the Gap D audit only.
        self.last_read = None

    def initial_state(self, batch: int, device, dtype) -> SlotMemoryState:
        keys = F.normalize(self.slot_bias, dim=-1, eps=1e-6).to(device=device, dtype=dtype)
        return SlotMemoryState(
            keys=keys.unsqueeze(0).expand(batch, -1, -1),
            values=torch.zeros(batch, self.slots, keys.shape[-1], device=device, dtype=dtype),
            usage=torch.zeros(batch, self.slots, device=device, dtype=dtype),
            acc_num=torch.zeros(batch, self.slots, keys.shape[-1], device=device, dtype=dtype),
            acc_key_num=torch.zeros(batch, self.slots, keys.shape[-1], device=device, dtype=dtype),
            acc_den=torch.zeros(batch, self.slots, device=device, dtype=dtype),
            causal_key_num=torch.zeros(batch, self.slots, keys.shape[-1], device=device, dtype=dtype),
            causal_den=torch.zeros(batch, self.slots, device=device, dtype=dtype),
            acc_count=0,
        )

    def _whitening(self, keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """ZCA address metric from the current slot keys (Gap D).

        Slot keys in the trained 20M model share a dominant residual-stream direction (pairwise
        cosine 0.47-0.81, 0.40-0.57 after centering), so cosine similarities all sit near 1, the
        read softmax is uniform (entropy 3.05-3.16 of ln 24 = 3.178) and the memory behaves as an
        average of every slot. Whitening removes the mean direction and rescales every address
        direction to unit variance:

            mu = mean_j K_j
            C  = cov(K), with C <- (1-g) C + g (tr C / slots) I
            W  = C^{-1/2}
            address(z) = normalize(W (z - mu) A^T)      (A = K - mu)

        The coordinates live in the slots-dimensional span, so W is an ``slots x slots`` matrix and
        the whole metric costs O(slots^3 + T * slots * width). The eigen-decomposition is taken
        under ``no_grad`` (a slowly varying metric); gradients still flow through the centred keys
        and queries, so the model learns which deviations should be discriminative.
        """
        if self.metric != "whitened" or keys.shape[1] < 2:
            return None
        keys_f = keys.to(torch.float32)
        slots = keys_f.shape[1]
        mu = keys_f.mean(dim=1, keepdim=True)
        basis = keys_f - mu
        cov = torch.matmul(basis, basis.transpose(1, 2)) / max(slots - 1, 1)
        mean_var = torch.diagonal(cov, dim1=-2, dim2=-1).mean(dim=-1)[:, None, None]
        cov = (1.0 - self.shrinkage) * cov + self.shrinkage * mean_var * torch.eye(
            slots, device=keys_f.device, dtype=keys_f.dtype
        )
        with torch.no_grad():
            eigenvalues, eigenvectors = torch.linalg.eigh(cov.detach())
            inverse_sqrt = eigenvalues.clamp_min(1e-8).rsqrt()
            transform = torch.matmul(eigenvectors * inverse_sqrt.unsqueeze(1), eigenvectors.transpose(1, 2))
        return mu, basis.detach(), transform

    def _address(
        self, z: torch.Tensor, whitening: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
    ) -> torch.Tensor:
        if whitening is None:
            return F.normalize(z, dim=-1, eps=1e-6)
        mu, basis, transform = whitening
        z_f = z.to(torch.float32)
        batch = z_f.shape[0]
        # ``z`` may have any leading time-like dimensions (query tokens [B,T,d] or causal
        # prototypes [B,T,S,d]); flatten them, apply the per-sample metric, then restore the shape.
        leading = z_f.shape[1:-1]
        flat = z_f.reshape(batch, -1, z_f.shape[-1])
        coordinates = torch.matmul(flat - mu, basis.transpose(1, 2))
        address = torch.matmul(coordinates, transform.transpose(1, 2))
        return F.normalize(address.reshape(batch, *leading, -1), dim=-1, eps=1e-6)

    def _novelty(self, scores: torch.Tensor) -> torch.Tensor:
        """1 - best match, with an optional dead-zone that maps near-duplicates to exactly zero."""
        novelty = (1.0 - scores.max(dim=-1).values).clamp(0.0, 1.0)
        if self.novelty_floor > 0.0:
            novelty = ((novelty - self.novelty_floor) / (1.0 - self.novelty_floor)).clamp_min(0.0)
        return novelty

    def _read(
        self,
        x: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        usage: torch.Tensor,
        whitening: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        # The learned projection is a refinement on top of the raw stream direction, not a free
        # linear map. Without the identity term the projection is free to collapse every token to
        # nearly the same direction during training (measured: distinct-token key cosine 0.80-0.98),
        # which routes all writes into one slot and makes the ring unaddressable.
        query = F.normalize(x + self.query(x), dim=-1)
        query_address = self._address(query, whitening)
        key_address = self._address(keys, whitening)
        scores = torch.matmul(query_address, key_address.transpose(1, 2))
        confidence = None
        if self.abstain:
            occupied = usage > 1e-3
            match = scores.masked_fill(~occupied[..., None, :], -1.0).max(dim=-1).values
            confidence = torch.sigmoid(self.abstain_sharpness * (match - self.abstain_threshold)).to(
                scores.dtype
            )
            self.last_confidence = confidence.detach()
        scores = scores + torch.log(usage.clamp_min(1e-3).to(scores.dtype))[:, None, :]
        weights = torch.softmax(scores / self.read_temperature, dim=-1)
        self.last_read = weights.detach()
        read = torch.matmul(weights.to(values.dtype), values)
        if confidence is not None:
            read = read * confidence[..., None].to(read.dtype)
        return self.readout(read)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor, state: SlotMemoryState | None
    ) -> tuple[torch.Tensor, SlotMemoryState]:
        batch, length, width = x.shape
        if state is None:
            state = self.initial_state(batch, x.device, x.dtype)

        keys, values, usage = state.keys, state.values, state.usage
        acc_num, acc_key_num, acc_den = state.acc_num, state.acc_key_num, state.acc_den
        causal_key_num, causal_den = state.causal_key_num, state.causal_den
        count = state.acc_count
        reads = []
        offset = 0
        while offset < length:
            take = min(self.block - count, length - offset)
            chunk_x = x[:, offset : offset + take]
            chunk_r = residual[:, offset : offset + take]
            # One metric per consolidation block: ``keys`` only change when a block consolidates,
            # so this is exactly chunk-invariant.
            whitening = self._whitening(keys)

            reads.append(self._read(chunk_x, keys, values, usage, whitening))

            token_keys = F.normalize(chunk_x + self.key(chunk_x), dim=-1, eps=1e-6)
            token_values = self.value(chunk_r)
            if whitening is None:
                norm_keys = F.normalize(keys, dim=-1, eps=1e-6)
                slot_scores = torch.matmul(token_keys, norm_keys.transpose(1, 2))
            else:
                token_address = self._address(token_keys, whitening)
                slot_scores = torch.matmul(token_address, self._address(keys, whitening).transpose(1, 2))
            evidence = torch.sigmoid(self.evidence(torch.cat([chunk_x, chunk_r], dim=-1))).squeeze(-1)

            # First pass: novelty against the last consolidation only. Its writes are accumulated
            # causally into a running prototype, so the second pass can score each token against
            # what was written *before it inside this block*. Without that, every copy of a repeated
            # filler measures its novelty against the stale consolidated prototype, floods the
            # accumulator and erases the distinct symbols written beside it (measured: block 64
            # threw away the symbols block 8 kept, taking presence recall from 1.00 to chance).
            novelty0 = self._novelty(slot_scores)
            with torch.no_grad():
                assignment0 = (
                    torch.softmax(slot_scores / self.write_temperature, dim=-1)
                    * (novelty0 * evidence.detach()).clamp(0.0, 1.0)[..., None]
                )
                weights = assignment0.to(torch.float32)
                running_key_num = causal_key_num.to(torch.float32)[:, None] + torch.cumsum(
                    weights.unsqueeze(-1) * token_keys.to(torch.float32).unsqueeze(2), dim=1
                )
                running_den = causal_den.to(torch.float32)[:, None] + torch.cumsum(weights, dim=1)
                # Shift by one position: a token must see the writes before it, not its own.
                prior_key_num = torch.cat(
                    [causal_key_num.to(torch.float32)[:, None], running_key_num[:, :-1]], dim=1
                )
                prior_den = torch.cat([causal_den.to(torch.float32)[:, None], running_den[:, :-1]], dim=1)
                prior = F.normalize(prior_key_num / prior_den.clamp_min(1e-6)[..., None], dim=-1, eps=1e-6)
                if whitening is None:
                    prior_scores = torch.einsum("btd,btsd->bts", token_keys.to(torch.float32), prior)
                else:
                    prior_address = self._address(prior, whitening)
                    prior_scores = torch.einsum("bti,btji->btj", token_address, prior_address)
                prior_scores = prior_scores.masked_fill(prior_den <= 0, -2.0)

            write_scores = torch.maximum(slot_scores, prior_scores.to(token_keys.dtype))
            novelty = self._novelty(write_scores)
            gate = (novelty * evidence).clamp(0.0, 1.0)
            # Load balancing acts on routing only: the gate keeps its own novelty margin, so a
            # repeated token is still recognised as redundant instead of being pushed into an
            # empty slot.
            route_scores = write_scores - self.load_penalty * torch.log1p(prior_den.to(write_scores.dtype))
            assignment = torch.softmax(route_scores / self.write_temperature, dim=-1) * gate[..., None]

            # ``assignment`` is float32 in whitened mode; keep the carried state's dtype so the
            # recursion and the checkpoint state stay identical to the cosine path.
            acc_num = acc_num + torch.einsum("bns,bnd->bsd", assignment.to(token_values.dtype), token_values)
            acc_key_num = acc_key_num + torch.einsum(
                "bns,bnd->bsd", assignment.to(token_keys.dtype), token_keys
            )
            acc_den = acc_den + assignment.sum(dim=1).to(acc_den.dtype)
            # Carry the first-pass causal accumulator so a block split across calls sees the same
            # prior writes as an unsplit block.
            causal_key_num = running_key_num[:, -1].to(token_keys.dtype)
            causal_den = running_den[:, -1].to(token_keys.dtype)
            count += take
            offset += take

            if count == self.block:
                # Soft consolidation: eta is the weight of the observations collected in this block.
                eta = (acc_den / (acc_den + 1.0))[..., None]
                mean_values = acc_num / acc_den.clamp_min(1e-6)[..., None]
                mean_keys = F.normalize(acc_key_num / acc_den.clamp_min(1e-6)[..., None], dim=-1, eps=1e-6)
                keys = F.normalize((1.0 - eta) * keys + eta * mean_keys, dim=-1, eps=1e-6)
                values = (1.0 - eta) * values + eta * mean_values
                usage = ((1.0 - eta.squeeze(-1)) * usage + eta.squeeze(-1)).clamp(0.0, 1.0)
                acc_num = torch.zeros_like(acc_num)
                acc_key_num = torch.zeros_like(acc_key_num)
                acc_den = torch.zeros_like(acc_den)
                causal_key_num = torch.zeros_like(causal_key_num)
                causal_den = torch.zeros_like(causal_den)
                count = 0

        read = torch.cat(reads, dim=1) if len(reads) > 1 else reads[0]
        return read, SlotMemoryState(
            keys, values, usage, acc_num, acc_key_num, acc_den, causal_key_num, causal_den, count
        )
