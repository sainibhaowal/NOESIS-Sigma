#!/usr/bin/env python3
"""Stage and push the Kaggle kernels for the TinyStories training run.

Kaggle uploads exactly the folder given to ``kernels push``, so the staging directory is assembled
from the live repository instead of keeping duplicate copies of the package under ``kaggle/``.

    python scripts/kaggle_push.py data          # CPU: build the token shards
    python scripts/kaggle_push.py train         # 2xT4: train the 20M config
    python scripts/kaggle_push.py train --watch # push, then poll until it finishes

The data kernel writes ``/kaggle/working/corpus``; download it with::

    python scripts/kaggle_push.py download ravsingh003/noesis-sigma-tinystories-data -p data/

and upload the folder as a private dataset with ``scripts/kaggle_push.py dataset``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = Path("/tmp/noesis_kaggle")


def kaggle() -> list[str]:
    exe = shutil.which("kaggle")
    if exe is None:
        exe = str(Path(sys.executable).with_name("kaggle"))
        if not Path(exe).exists():
            raise SystemExit("kaggle CLI not found; pip install kaggle and run 'kaggle auth login'")
    return [exe]


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("+", " ".join(str(part) for part in command), flush=True)
    return subprocess.run(command, check=True, **kwargs)


def stage_code() -> Path:
    """Package the importable code as a single tar: script kernels only upload their code file."""
    target = STAGE / "code"
    shutil.rmtree(target, ignore_errors=True)
    payload = STAGE / "code_payload"
    shutil.rmtree(payload, ignore_errors=True)
    payload.mkdir(parents=True)
    shutil.copytree(ROOT / "sigmastorm", payload / "sigmastorm", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "training", payload / "training", ignore=shutil.ignore_patterns("__pycache__"))
    (payload / "scripts").mkdir()
    for name in ("train.py", "build_corpus.py", "eval.py", "ladder.py", "gate.py", "gapd_proof.py"):
        shutil.copy2(ROOT / "scripts" / name, payload / "scripts" / name)
    (payload / "configs").mkdir()
    for path in sorted((ROOT / "configs").glob("*.json")):
        shutil.copy2(path, payload / "configs" / path.name)
    (payload / "kaggle" / "train").mkdir(parents=True)
    shutil.copy2(
        ROOT / "kaggle" / "train" / "train_config.json", payload / "kaggle" / "train" / "train_config.json"
    )
    (payload / "kaggle" / "gapd_train").mkdir(parents=True)
    shutil.copy2(
        ROOT / "kaggle" / "gapd_train" / "train_config.json",
        payload / "kaggle" / "gapd_train" / "train_config.json",
    )

    target.mkdir(parents=True)
    subprocess.run(["tar", "-cf", str(target / "code.tar"), "-C", str(payload), "."], check=True)
    (target / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "noesis sigma code tinystories",
                "id": "ravsingh003/noesis-sigma-code-tinystories",
                "licenses": [{"name": "Apache-2.0"}],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return target


def stage_data() -> Path:
    target = STAGE / "data"
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    shutil.copy2(ROOT / "kaggle" / "data_build" / "run.py", target / "run.py")
    shutil.copy2(ROOT / "kaggle" / "data_build" / "kernel-metadata.json", target / "kernel-metadata.json")
    return target


def stage_train() -> Path:
    target = STAGE / "train"
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    shutil.copy2(ROOT / "kaggle" / "train" / "run.py", target / "run.py")
    shutil.copy2(ROOT / "kaggle" / "train" / "kernel-metadata.json", target / "kernel-metadata.json")
    return target


def watch(slug: str, interval: int = 60) -> int:
    while True:
        output = subprocess.run(kaggle() + ["kernels", "status", slug], capture_output=True, text=True).stdout
        print(time.strftime("%H:%M:%S"), output.strip().splitlines()[-1] if output else "?", flush=True)
        lowered = output.lower()
        if "complete" in lowered:
            return 0
        if "error" in lowered or "cancel" in lowered:
            return 1
        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=["code", "data", "train", "status", "output", "download", "dataset"]
    )
    parser.add_argument("extra", nargs="*")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--version", action="store_true", help="write a new version of an existing dataset")
    args = parser.parse_args()

    if args.action == "code":
        folder = stage_code()
        # The dataset holds one code.tar; the kernel extracts it into /kaggle/working.
        if args.version:
            run(kaggle() + ["datasets", "version", "-p", str(folder), "-m", "code refresh"])
        else:
            run(kaggle() + ["datasets", "create", "-p", str(folder)])
        return
    if args.action == "data":
        folder = stage_data()
        run(kaggle() + ["kernels", "push", "-p", str(folder)])
        if args.watch:
            raise SystemExit(watch("ravsingh003/noesis-sigma-tinystories-data"))
        return
    if args.action == "train":
        folder = stage_train()
        run(kaggle() + ["kernels", "push", "-p", str(folder)])
        if args.watch:
            raise SystemExit(watch("ravsingh003/noesis-sigma-20m"))
        return
    if args.action in {"status", "output", "download"}:
        slug = args.extra[0] if args.extra else "ravsingh003/noesis-sigma-20m"
        command = {"status": "status", "output": "output", "download": "output"}[args.action]
        run(kaggle() + ["kernels", command, slug, *args.extra[1:]])
        return
    if args.action == "dataset":
        folders = [part for part in args.extra if not part.startswith("--")]
        folder = Path(folders[0]) if folders else ROOT / "data" / "tinystories_dataset"
        metadata = folder / "dataset-metadata.json"
        if not metadata.exists():
            metadata.write_text(
                json.dumps(
                    {
                        "title": "noesis sigma tinystories",
                        "id": "ravsingh003/noesis-sigma-tinystories",
                        "licenses": [{"name": "other"}],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        if args.version:
            run(
                kaggle()
                + ["datasets", "version", "-p", str(folder), "--dir-mode", "tar", "-m", "corpus refresh"]
            )
        else:
            run(kaggle() + ["datasets", "create", "-p", str(folder), "--dir-mode", "tar"])
        return


if __name__ == "__main__":
    main()
