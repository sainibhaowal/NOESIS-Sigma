from pathlib import Path

from Core.LongTask.checkpointing import load_latest_checkpoint, write_checkpoint_atomic
from Core.LongTask.task_models import TaskCheckpoint


def test_checkpoint_atomicity_tmp_ignored(tmp_path: Path):
    ckpt = TaskCheckpoint(
        checkpoint_id="c1",
        task_id="t1",
        chunk_id=1,
        thread_id="th",
        plan_id="p",
        policy_mode="BALANCED",
        plan_version=1,
        plan_hash="h",
        created_at_ms=1,
    )
    root = tmp_path / "ck"
    write_checkpoint_atomic(root, ckpt)

    # Create a temp file that should be ignored
    (root / "2.json.next").write_text("{\"bad\":true}")

    latest = load_latest_checkpoint(root)
    assert latest is not None
    assert latest.chunk_id == 1
