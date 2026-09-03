from __future__ import annotations

import os
import sys
from pathlib import Path


def _ensure_repo_root() -> None:
    repo = Path(__file__).resolve()
    for _ in range(6):
        if (repo / 'pyproject.toml').exists():
            break
        repo = repo.parent
    sys.path.insert(0, str(repo))

_ensure_repo_root()


import argparse
import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path when run as a script.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from External.WKS.index_build import build_wks_index


def main() -> int:
    p = argparse.ArgumentParser(description="Build WKS HNSW index")
    p.add_argument("--pack-root", required=True, help="Path to WKS pack root")
    p.add_argument("--concept-map", required=True, help="Path to concept->index json")
    p.add_argument("--out-dir", required=True, help="Output directory for index files")
    if len(sys.argv) == 1:
        p.print_help()
        return 0
    args = p.parse_args()

    concept_map = json.loads(Path(args.concept_map).read_text("utf-8"))
    concept_to_idx = {str(k): int(v) for k, v in concept_map.items()}
    build_wks_index(Path(args.pack_root), concept_to_idx, Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
