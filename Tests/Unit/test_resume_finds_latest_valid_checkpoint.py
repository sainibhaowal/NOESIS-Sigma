from pathlib import Path

from Core.LongTask.checkpointing import load_latest_checkpoint, write_checkpoint_atomic
from Core.LongTask.task_models import TaskCheckpoint


def test_resume_finds_latest_valid_checkpoint(tmp_path: Path):
    root = tmp_path / "ck"
    ckpt1 = TaskCheckpoint(
        checkpoint_id="c1",
        task_id="t",
        chunk_id=1,
        thread_id="th",
        plan_id="p",
        policy_mode="BALANCED",
        plan_version=1,
        plan_hash="h1",
        created_at_ms=1,
    )
    ckpt2 = TaskCheckpoint(
        checkpoint_id="c2",
        task_id="t",
        chunk_id=2,
        thread_id="th",
        plan_id="p",
        policy_mode="BALANCED",
        plan_version=2,
        plan_hash="h2",
        created_at_ms=2,
    )
    write_checkpoint_atomic(root, ckpt1)
    write_checkpoint_atomic(root, ckpt2)

    # Corrupt latest
    (root / "2.json").write_text("{\"corrupt\":true}")

    latest = load_latest_checkpoint(root)
    assert latest is not None
    assert latest.chunk_id == 1
