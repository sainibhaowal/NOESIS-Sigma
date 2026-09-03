from pathlib import Path

from Core.LongTask.artifact_store_fs import FSArtifactStore


def test_artifact_hashing_manifest(tmp_path: Path):
    store = FSArtifactStore(tmp_path)
    data = b"hello"
    ref = store.put_bytes(kind="REPORT_MD", data=data, task_id="t", chunk_id=1)
    assert ref.sha256
    assert ref.bytes == len(data)
