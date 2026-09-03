from __future__ import annotations

import base64
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest


def test_keygen_raw_and_b64_match(tmp_path: Path) -> None:
    if shutil.which("openssl") is None:
        pytest.skip("openssl is required for keygen test")

    out_dir = tmp_path / "keys"
    subprocess.run(
        ["bash", "Core/Security/keygen.sh", "--dir", str(out_dir), "--force"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    raw = (out_dir / "model.aes.key").read_bytes()
    b64 = (out_dir / "model.aes.key.b64").read_text(encoding="utf-8").strip()
    assert len(raw) == 32
    assert base64.b64decode(b64) == raw

    if os.name == "posix":
        mode = stat.S_IMODE((out_dir / "model.aes.key.b64").stat().st_mode)
        assert mode == 0o600
