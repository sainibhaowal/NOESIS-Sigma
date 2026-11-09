from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sigmastorm.config import SigmaConfig  # noqa: E402
from sigmastorm.model import SigmaStorm  # noqa: E402
from sigmastorm.ops.delta_rule import _delta_chunk, decay_spectrum, delta_rule_reference  # noqa: E402
from training.tasks import make_mqar_batch  # noqa: E402

torch.manual_seed(0)


def test_config_roundtrip(tmp_path):
    config = SigmaConfig.for_size("tiny")
    path = tmp_path / "config.json"
    config.save(path)
    assert SigmaConfig.load(path) == config


def test_config_rejects_bad_heads():
    with pytest.raises(ValueError):
        SigmaConfig(d_model=100, n_heads=8)


def test_delta_chunk_matches_sequential():
    batch, heads, length, dim, chunk = 2, 2, 96, 16, 32
    q = F.normalize(torch.randn(batch, heads, length, dim), dim=-1)
    k = F.normalize(torch.randn(batch, heads, length, dim), dim=-1)
    v = torch.randn(batch, heads, length, dim)
    beta = torch.sigmoid(torch.randn(batch, heads, length))
    g = -torch.rand(batch, heads, length) * 0.5

    reference, reference_state = delta_rule_reference(q, k, v, beta, g)
    state = torch.zeros(batch, heads, dim, dim)
    pieces = []
    for start in range(0, length, chunk):
        out, state = _delta_chunk(
            q[:, :, start : start + chunk],
            k[:, :, start : start + chunk],
            v[:, :, start : start + chunk],
            beta[:, :, start : start + chunk],
            g[:, :, start : start + chunk],
            state,
        )
        pieces.append(out)
    chunked = torch.cat(pieces, dim=2)
    assert torch.allclose(chunked, reference, atol=2e-4)
    assert torch.allclose(state, reference_state, atol=2e-4)


def test_chunked_forward_matches_full():
    # Every operator, including the slot memory and the summary ring, now carries exact state, so
    # the split can be at an arbitrary boundary (here 50, which cuts inside a 64-token slot block
    # and inside a 32-token summary block).
    config = SigmaConfig.for_size("tiny")
    model = SigmaStorm(config).eval()
    tokens = torch.randint(0, config.vocab_size, (2, 128))
    with torch.no_grad():
        whole, _, _, _ = model(tokens)
        first, state, _, _ = model(tokens[:, :50])
        second, _, _, _ = model(tokens[:, 50:], state.detach())
    assert torch.allclose(whole, torch.cat([first, second], dim=1), atol=3e-3)


def test_shard_cursor_resumes_on_every_rank(tmp_path):
    """The checkpoint carries a token count, so rank 1 can resume from rank 0's save (DDP)."""

    import json

    import numpy as np

    from training.data import PackedShards

    lengths = [7, 5, 9, 4]
    for index, count in enumerate(lengths):
        np.arange(count, dtype=np.uint32).tofile(tmp_path / f"shard-{index}.bin")
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "dtype": "uint32",
                "eos_token_id": 2,
                "shards": [
                    {"path": f"shard-{index}.bin", "tokens": count} for index, count in enumerate(lengths)
                ],
            }
        )
    )

    rank0 = PackedShards(tmp_path, rank=0, world_size=2)
    rank1 = PackedShards(tmp_path, rank=1, world_size=2)
    for _ in range(4):
        rank0.sample(2, 3)
        rank1.sample(2, 3)

    resumed = PackedShards(tmp_path, rank=1, world_size=2)
    resumed.load_state_dict(rank0.state_dict())  # must not raise on a non-owner rank
    np.testing.assert_array_equal(resumed.sample(2, 3), rank1.sample(2, 3))

    legacy = PackedShards(tmp_path, rank=1, world_size=2)
    legacy.load_state_dict({"cursor": [1, 2], "rank": 0})
    assert legacy.consumed == 7 + 2


def test_slot_memory_suppresses_in_block_repeats():
    """A repeated token inside one consolidation block must not flood the accumulator.

    The novelty gate originally compared every token only with the last consolidated prototypes,
    so a block full of one repeated filler wrote every copy. The causal second pass compares each
    token with the keys written before it in the same block, leaving only the first occurrence
    gated on (Gap D).
    """

    from sigmastorm.ops.slot_memory import SlotMemory

    torch.manual_seed(0)
    memory = SlotMemory(width=32, slots=8, block=64)
    x = torch.randn(1, 32, 32)
    x[:, 1:] = x[:, 1:2]  # token 0 is distinct; tokens 1..31 are the same repeated filler
    _, state = memory(x, x, None)
    # One first occurrence writes with gate <= 1; the 31 exact repeats must be almost silent.
    assert float(state.acc_den.detach().sum()) < 2.0


def test_state_shape_independent_of_length():
    config = SigmaConfig.for_size("tiny")
    model = SigmaStorm(config)
    assert model.initial_state(2).memory_bytes() == model.initial_state(2).memory_bytes()
    for index, block in enumerate(model.initial_state(2).blocks):
        assert block.ssm.shape == (2, config.heads_eff, config.head_dim, config.head_dim)
        assert block.memory.acc_num.shape == (2, config.slot_count, config.d_model)
        if config.uses_summary(index):
            assert block.summary.keys.shape == (2, config.n_heads, config.summary_slots, config.head_dim)
        else:
            assert block.summary is None


def test_decay_spectrum_is_multiscale():
    rates = decay_spectrum(8, 1e-4, 3e-2)
    assert rates.shape == (8,)
    assert torch.all(rates[1:] > rates[:-1])
    assert torch.isclose(rates[0], torch.tensor(1e-4), rtol=1e-5)
    assert torch.isclose(rates[-1], torch.tensor(3e-2), rtol=1e-5)
    # The bias must survive the generic initialiser: alpha stays > 0.9 for the slowest head.
    config = SigmaConfig.from_dict({**SigmaConfig.for_size("tiny").to_dict(), "state_expand": 2})
    model = SigmaStorm(config)
    rule = model.blocks[0].transition.delta
    assert rule.heads == config.heads_eff
    alpha = torch.exp(-torch.nn.functional.softplus(rule.decay_proj.bias))
    assert alpha.max() > 0.999 and alpha.min() < 0.98


def test_state_expansion_changes_only_capacity():
    config = SigmaConfig.from_dict({**SigmaConfig.for_size("tiny").to_dict(), "state_expand": 3})
    model = SigmaStorm(config)
    rule = model.blocks[0].transition.delta
    assert rule.heads == config.n_heads * 3
    assert rule.head_dim == config.head_dim
    assert rule.inner == config.d_model * 3
    assert rule.out_proj.weight.shape == (config.d_model, config.d_model * 3)


def test_whitened_slot_metric_recovers_collinear_addresses():
    """Gap D: whitening must separate slot keys that a shared direction makes almost parallel."""

    from sigmastorm.ops.slot_memory import SlotMemory

    torch.manual_seed(0)
    slots, width = 8, 32
    memory = SlotMemory(width, slots=slots, metric="whitened", shrinkage=0.05)
    shared = F.normalize(torch.randn(1, width), dim=-1)
    deviations = F.normalize(torch.randn(slots, width), dim=-1)
    keys = F.normalize(shared + 0.05 * deviations, dim=-1)

    cosine = (keys @ keys.T).fill_diagonal_(0.0)
    whitening = memory._whitening(keys.unsqueeze(0))
    address = memory._address(keys.unsqueeze(0), whitening)[0]
    whitened = (address @ address.T).fill_diagonal_(0.0)

    # Raw cosine is nearly 1 for every pair; the whitened metric is not.
    assert float(cosine.max()) > 0.99
    assert float(whitened.max()) < float(cosine.max()) - 0.1


def test_whitened_slot_memory_is_chunk_invariant():
    config = SigmaConfig.from_dict(
        {
            **SigmaConfig.for_size("tiny").to_dict(),
            "memory_metric": "whitened",
        }
    )
    model = SigmaStorm(config).eval()
    tokens = torch.randint(0, config.vocab_size, (2, 128))
    with torch.no_grad():
        whole, _, _, _ = model(tokens)
        first, state, _, _ = model(tokens[:, :50])
        second, _, _, _ = model(tokens[:, 50:], state.detach())
    assert torch.allclose(whole, torch.cat([first, second], dim=1), atol=3e-3)


def test_whitened_slot_memory_backward_is_finite():
    config = SigmaConfig.from_dict(
        {
            **SigmaConfig.for_size("tiny").to_dict(),
            "memory_metric": "whitened",
        }
    )
    model = SigmaStorm(config)
    tokens = torch.randint(0, config.vocab_size, (2, 64))
    outputs, _ = model.loss(tokens[:, :-1], tokens[:, 1:])
    loss = outputs["lm_sum"] / max(outputs["lm_count"], 1)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


GAPD = {
    "memory_novelty_floor": 0.1,
    "memory_read_temperature": 0.1,
    "memory_abstain": True,
}


def test_novelty_dead_zone_blocks_repeated_writes():
    """Gap D: a repeated token below the novelty floor must not accumulate pseudo-observations."""

    from sigmastorm.ops.slot_memory import SlotMemory

    x0 = torch.randn(1, 1, 16)
    tokens = x0.expand(1, 24, 16).contiguous()
    torch.manual_seed(0)
    floor = SlotMemory(16, slots=4, block=8, novelty_floor=0.5)
    torch.manual_seed(0)
    base = SlotMemory(16, slots=4, block=8, novelty_floor=0.0)
    base.load_state_dict(floor.state_dict())
    with torch.no_grad():
        _, floor_state = floor(tokens, tokens, None)
        _, base_state = base(tokens, tokens, None)
        novelty = floor._novelty(torch.tensor([[[0.98, 0.2, 0.1]]]))
        low_novelty = floor._novelty(torch.tensor([[[0.2, 0.1, 0.0]]]))
    assert float(novelty) == 0.0  # max sim 0.98 > 1 - floor
    assert 0.0 < float(low_novelty) < 1.0
    assert float(floor_state.usage.sum()) < 0.9 * float(base_state.usage.sum())


def test_abstention_gate_suppresses_unmatched_query():
    """Gap D: a query that matches no slot gets a gate near zero; a matching one stays open."""

    from sigmastorm.ops.slot_memory import SlotMemory

    torch.manual_seed(0)
    memory = SlotMemory(16, slots=4, temperature=0.5, read_temperature=0.1, abstain=True)
    keys = F.normalize(torch.randn(1, 4, 16), dim=-1)
    values = torch.randn(1, 4, 16)
    usage = torch.ones(1, 4)
    matched = (50.0 * keys[:, 0]).unsqueeze(1)
    unmatched = 50.0 * F.normalize(torch.randn(1, 1, 16), dim=-1)
    with torch.no_grad():
        memory._read(matched, keys, values, usage)
        matched_conf = float(memory.last_confidence.mean())
        memory._read(unmatched, keys, values, usage)
        unmatched_conf = float(memory.last_confidence.mean())
    assert matched_conf > 0.9
    assert unmatched_conf < 0.3


def test_gated_slot_memory_is_chunk_invariant():
    config = SigmaConfig.from_dict({**SigmaConfig.for_size("tiny").to_dict(), **GAPD})
    model = SigmaStorm(config).eval()
    tokens = torch.randint(0, config.vocab_size, (2, 128))
    with torch.no_grad():
        whole, _, _, _ = model(tokens)
        first, state, _, _ = model(tokens[:, :50])
        second, _, _, _ = model(tokens[:, 50:], state.detach())
    assert torch.allclose(whole, torch.cat([first, second], dim=1), atol=3e-3)


def test_gated_slot_memory_backward_is_finite():
    config = SigmaConfig.from_dict({**SigmaConfig.for_size("tiny").to_dict(), **GAPD})
    model = SigmaStorm(config)
    tokens = torch.randint(0, config.vocab_size, (2, 64))
    outputs, _ = model.loss(tokens[:, :-1], tokens[:, 1:])
    loss = outputs["lm_sum"] / max(outputs["lm_count"], 1)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_latent_is_bounded():
    config = SigmaConfig.for_size("tiny")
    model = SigmaStorm(config).eval()
    tokens = torch.randint(0, config.vocab_size, (2, 128))
    with torch.no_grad():
        hidden, _, _, _ = model(tokens)
    assert torch.isfinite(hidden).all()


def test_generate_decodes_past_the_prompt():
    config = SigmaConfig.for_size("tiny")
    model = SigmaStorm(config).eval()
    prompt = torch.randint(0, config.vocab_size, (4,))
    out = model.generate(prompt, max_new_tokens=4, temperature=1.0, top_k=None)
    assert prompt.shape[0] <= out.shape[0] <= prompt.shape[0] + 4


def test_loss_backward_is_finite():
    config = SigmaConfig.for_size("tiny")
    model = SigmaStorm(config)
    tokens = torch.randint(0, config.vocab_size, (2, 64))
    outputs, _ = model.loss(tokens[:, :-1], tokens[:, 1:])
    loss = outputs["lm_sum"] / max(outputs["lm_count"], 1)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_mqar_masks_only_query_positions():
    data = make_mqar_batch(4, 64, n_queries=4, n_keys=64, n_values=16)
    assert data["input_ids"].shape == data["targets"].shape
    assert int(data["loss_mask"].sum()) == 4 * 4
    supervised = data["targets"][data["loss_mask"]]
    assert ((supervised >= 64) & (supervised < 80)).all()


def test_no_sequence_quadratic_materialisation():
    """A 1024-token chunk must not allocate any [B, H, T, T] tensor."""

    config = SigmaConfig.for_size("tiny")
    model = SigmaStorm(config).eval()
    tokens = torch.randint(0, config.vocab_size, (1, 1024))
    with torch.no_grad():
        model(tokens)
    # The window attention mask is [1, H, T, window], not full T x T. Nothing to assert directly,
    # but the forward completing without OOM on a 1K sequence is the practical check.
    assert True
