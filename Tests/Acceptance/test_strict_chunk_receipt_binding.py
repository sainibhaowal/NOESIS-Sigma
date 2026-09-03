from Core.LongTask.checkpointing import checkpoint_hash
from Core.LongTask.task_models import TaskCheckpoint


def test_strict_chunk_receipt_binding():
    ckpt = TaskCheckpoint(
        checkpoint_id="c1",
        task_id="t1",
        chunk_id=1,
        thread_id="th",
        plan_id="p1",
        policy_mode="STRICT",
        sim_snapshot_id="snap",
        scene_hash="scene",
        routing_trace_hash="route",
        plan_version=1,
        plan_hash="planhash",
        created_at_ms=1,
    )
    h1 = checkpoint_hash(ckpt)
    ckpt2 = ckpt.model_copy(update={"plan_hash": "planhash2"})
    h2 = checkpoint_hash(ckpt2)
    assert h1 != h2
