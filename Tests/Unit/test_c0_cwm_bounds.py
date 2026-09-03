from __future__ import annotations

from typing import cast

from Core.Continuity.models import (
    MAX_ACTIVE_EPISODE_REFS,
    MAX_FACTS,
    MAX_SLOT_REFS,
    CWMState,
    CWMVecSlot,
    FactItem,
    FactsTable,
    MemoryRef,
    ShardRef,
)


def test_cwm_bounds_caps() -> None:
    memory_refs = [
        MemoryRef(record_id=f"r{i}", span_id=None, sha256="x" * 64, ts_ms=1)
        for i in range(20)
    ]
    refs = cast(list[MemoryRef | ShardRef], memory_refs)
    slot = CWMVecSlot(
        label="x",
        confidence=0.5,
        refs=refs,
        vec_b64="AA==",
    )
    cwm = CWMState(
        goal_slots=[slot, slot],
        entity_slots=[slot, slot, slot],
        constraint_slots=[slot, slot],
        affect_slots=[slot],
        active_episode_refs=memory_refs,
    )

    # slot refs capped
    assert len(cwm.goal_slots[0].refs) <= MAX_SLOT_REFS
    # active refs capped
    assert len(cwm.active_episode_refs) <= MAX_ACTIVE_EPISODE_REFS


def test_facts_table_caps() -> None:
    items = [
        FactItem(
            key=f"k{i}",
            value="v",
            confidence=1.0,
            source_kind="SIM",
            source_ref="r",
            sha256="x" * 64,
            updated_at_ms=i,
        )
        for i in range(200)
    ]
    ft = FactsTable(items=items)
    assert len(ft.items) <= MAX_FACTS
