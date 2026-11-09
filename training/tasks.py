"""Synthetic recall tasks. MQAR is the gate any non-attention architecture has to pass.

Sequence layout::

    [k1 v1] [k2 v2] ... [kn vn]  [Q k_q1 v_q1] [Q k_q2 v_q2] ...

The loss applies only at each ``k_q`` position, where the next token must be the value bound to that
key. Dense next-token prediction would be trivial (copy the previous token); this is the real test.
"""

from __future__ import annotations

import torch


def make_mqar_batch(
    batch: int,
    length: int,
    n_queries: int = 4,
    n_keys: int = 63,
    n_values: int = 63,
    device=None,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    if length < 3 * n_queries + 4:
        raise ValueError("length is too short for the requested number of queries")

    query_token = n_keys + n_values
    pad_token = query_token + 1
    vocab = pad_token + 1
    n_pairs = (length - 3 * n_queries) // 2

    if n_pairs > n_keys:
        raise ValueError("n_pairs must not exceed n_keys when keys are drawn without replacement")

    tokens = torch.full((batch, length), pad_token, dtype=torch.long, device=device)
    # Keys are unique within a sequence. Sampling with replacement silently caps accuracy far
    # below 1 (a queried key may have been bound several times), turning recall into guessing.
    key_ids = torch.stack(
        [torch.randperm(n_keys, device=device, generator=generator)[:n_pairs] for _ in range(batch)]
    )
    value_ids = torch.randint(n_keys, n_keys + n_values, (batch, n_pairs), device=device, generator=generator)

    tokens[:, 0 : 2 * n_pairs : 2] = key_ids
    tokens[:, 1 : 2 * n_pairs : 2] = value_ids

    mask = torch.zeros_like(tokens, dtype=torch.bool)
    positions = []
    for query in range(n_queries):
        base = 2 * n_pairs + 3 * query
        choice = torch.randint(0, n_pairs, (batch,), device=device, generator=generator)
        chosen_key = key_ids.gather(1, choice[:, None]).squeeze(1)
        chosen_value = value_ids.gather(1, choice[:, None]).squeeze(1)
        tokens[:, base] = query_token
        tokens[:, base + 1] = chosen_key
        tokens[:, base + 2] = chosen_value
        mask[:, base + 1] = True  # at k_q the model must emit v_q
        positions.append(base + 1)

    inputs = tokens[:, :-1].contiguous()
    targets = tokens[:, 1:].contiguous()
    loss_mask = mask[:, :-1].contiguous()
    return {
        "input_ids": inputs,
        "targets": targets,
        "loss_mask": loss_mask,
        "vocab_size": torch.tensor(vocab),
        "positions": torch.tensor(positions),
        "query_token": query_token,
        "n_keys": n_keys,
    }


def make_delayed_recall_batch(
    batch: int,
    length: int,
    n_pairs: int = 8,
    n_values: int = 16,
    n_queries: int = 4,
    device=None,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Long-distance key → value recall with the queries far from the bindings.

    Layout: ``n_pairs`` bindings are placed in the first quarter of the sequence, then a long stretch
    of filler, then ``n_queries`` ``[Q key value]`` triples. The supervised position is the query
    key, and the correct answer is the bound value. Distances are returned per query so recall can
    be reported as a function of how far back the binding lies — the axis on which exponential
    recurrence, a summary ring and full attention differ.
    """

    if n_pairs > length // 6:
        raise ValueError("too many pairs for this length")
    n_keys = n_pairs
    query_id = n_keys + n_values
    filler_id = query_id + 1
    pad_id = filler_id + 1
    vocab = pad_id + 1

    tokens = torch.full((batch, length), filler_id, dtype=torch.long, device=device)
    keys = torch.stack(
        [torch.randperm(n_keys, device=device, generator=generator)[:n_pairs] for _ in range(batch)]
    )
    values = torch.randint(n_keys, n_keys + n_values, (batch, n_pairs), device=device, generator=generator)
    pair_start = 1
    pair_positions = pair_start + 2 * torch.arange(n_pairs, device=device)
    tokens[:, pair_positions] = keys
    tokens[:, pair_positions + 1] = values

    mask = torch.zeros_like(tokens, dtype=torch.bool)
    first_query = int(pair_positions[-1]) + 8
    if length - first_query < 3 * n_queries + 2:
        raise ValueError("length is too short for the requested number of queries")
    slots = torch.linspace(first_query, length - 3, n_queries, device=device).long()
    distances = torch.zeros((batch, n_queries), dtype=torch.long, device=device)
    for query_index in range(n_queries):
        base = int(slots[query_index])
        choice = torch.randint(0, n_pairs, (batch,), device=device, generator=generator)
        chosen_key = keys.gather(1, choice[:, None]).squeeze(1)
        chosen_value = values.gather(1, choice[:, None]).squeeze(1)
        tokens[:, base] = query_id
        tokens[:, base + 1] = chosen_key
        tokens[:, base + 2] = chosen_value
        mask[:, base + 1] = True  # at the queried key the model must emit the bound value
        pair_index = (keys == chosen_key[:, None]).float().argmax(dim=-1)
        key_position = pair_positions[pair_index]
        distances[:, query_index] = base + 1 - key_position

    inputs = tokens[:, :-1].contiguous()
    return {
        "input_ids": inputs,
        "targets": tokens[:, 1:].contiguous(),
        "loss_mask": mask[:, :-1].contiguous(),
        "vocab_size": torch.tensor(vocab),
        "distances": distances,
        "query_token": query_id,
        "n_keys": n_keys,
    }


def make_presence_recall_batch(
    batch: int,
    length: int,
    n_symbols: int = 16,
    n_present: int = 8,
    device=None,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Did the queried symbol appear in the opening set?

    Layout: ``n_present`` distinct symbols scattered over the first quarter, repeated filler for the
    rest, then ``[Q, symbol, yes/no]`` at the end. This is the capability the slot memory actually
    has — a long-horizon prototype store read by content similarity — rather than key->value
    binding, which is the delta rule's job. Filler is a single repeated token, so the novelty gate
    suppresses it; a single call-wide consolidation never publishes the symbols before the query,
    which is exactly the defect of the old chunk-mean write.
    """

    if not 0 < n_present < n_symbols:
        raise ValueError("require 0 < n_present < n_symbols")
    if length < 8 * n_present:
        raise ValueError("length is too short for the requested number of symbols")
    filler = n_symbols
    query_token = n_symbols + 1
    yes_token = n_symbols + 2
    no_token = n_symbols + 3
    vocab = n_symbols + 4

    tokens = torch.full((batch, length), filler, dtype=torch.long, device=device)
    present = torch.stack(
        [torch.randperm(n_symbols, device=device, generator=generator)[:n_present] for _ in range(batch)]
    )
    region = max(length // 4, n_present + 1)
    positions = torch.stack(
        [torch.randperm(region, device=device, generator=generator)[:n_present] for _ in range(batch)]
    )
    tokens.scatter_(1, positions, present)

    is_present = torch.rand(batch, device=device, generator=generator) < 0.5
    choice = torch.randint(0, n_present, (batch,), device=device, generator=generator)
    from_present = present.gather(1, choice[:, None]).squeeze(1)
    allowed = torch.ones(batch, n_symbols, device=device)
    allowed.scatter_(1, present, 0.0)
    absent = torch.multinomial(allowed, 1, generator=generator).squeeze(1)
    query = torch.where(is_present, from_present, absent)

    tokens[:, -3] = query_token
    tokens[:, -2] = query
    tokens[:, -1] = torch.where(is_present, yes_token, no_token)

    mask = torch.zeros_like(tokens, dtype=torch.bool)
    mask[:, -2] = True
    return {
        "input_ids": tokens[:, :-1].contiguous(),
        "targets": tokens[:, 1:].contiguous(),
        "loss_mask": mask[:, :-1].contiguous(),
        "vocab_size": torch.tensor(vocab),
        "is_present": is_present,
    }


def make_aggregate_recall_batch(
    batch: int,
    length: int,
    max_count: int = 8,
    n_symbols: int = 8,
    device=None,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Count a set of symbols seen far back in the sequence.

    All symbols occur in the first quarter, followed by filler and a query at the end; the answer is
    the *number* of symbols, a statistic that is preserved by a pooled summary but not by a
    key→value binding or a prototype store. This isolates the global aggregate path (Gap C) from the
    exact binding-recall path (Gap A).
    """

    if length < 8 * max_count:
        raise ValueError("length is too short for this count")
    query_id = n_symbols + 1
    answer_base = n_symbols + 2
    vocab = answer_base + max_count + 1

    tokens = torch.zeros((batch, length), dtype=torch.long, device=device)
    counts = torch.randint(0, max_count + 1, (batch,), device=device, generator=generator)
    region = max(length // 4, max_count + 1)
    for row in range(batch):
        count = int(counts[row])
        if count == 0:
            continue
        chosen = torch.randperm(region, device=device, generator=generator)[:count]
        tokens[row, chosen] = torch.randint(1, n_symbols + 1, (count,), device=device, generator=generator)
    tokens[:, -2] = query_id
    tokens[:, -1] = answer_base + counts

    mask = torch.zeros_like(tokens, dtype=torch.bool)
    mask[:, -2] = True
    return {
        "input_ids": tokens[:, :-1].contiguous(),
        "targets": tokens[:, 1:].contiguous(),
        "loss_mask": mask[:, :-1].contiguous(),
        "vocab_size": torch.tensor(vocab),
        "counts": counts,
        "query_token": query_id,
    }


@torch.no_grad()
def mqar_accuracy(model, batch_data: dict[str, torch.Tensor], state=None) -> tuple[float, int]:
    """Fraction of supervised query positions answered correctly."""

    input_ids = batch_data["input_ids"]
    targets = batch_data["targets"]
    mask = batch_data["loss_mask"]
    hidden, _, _, _ = model(input_ids, state)
    logits = model.lm_head(hidden)
    predictions = logits.argmax(dim=-1)
    correct = (predictions[mask] == targets[mask]).float().mean() if bool(mask.any()) else torch.tensor(0.0)
    return float(correct), int(mask.sum())
