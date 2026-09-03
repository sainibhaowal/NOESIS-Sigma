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
import hashlib
import html
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, Tuple

import zstandard as zstd

_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-я0-9_]+")


def strip_html(text: str) -> str:
    # Remove script/style blocks
    text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    # Drop tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Unescape HTML entities
    text = html.unescape(text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def hash_bucket(token: str, buckets: int) -> int:
    h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little") % buckets


def make_dist(tokens: Iterable[str], buckets: int, top_k: int = 64) -> list[tuple[str, float]]:
    counts: Dict[int, int] = {}
    total = 0
    for t in tokens:
        idx = hash_bucket(t, buckets)
        counts[idx] = counts.get(idx, 0) + 1
        total += 1
    if total <= 0:
        return []
    # top-k by count
    items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:top_k]
    return [(f"h{idx}", cnt / float(total)) for idx, cnt in items]


def iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file():
            if "_exceptions" in p.parts:
                continue
            yield p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", required=True, help="zimdump output dir")
    ap.add_argument("--out-pack", required=True, help="output pack dir")
    ap.add_argument("--concept-map-out", required=True, help="concept map json output")
    ap.add_argument("--bucket-count", type=int, default=4096)
    ap.add_argument("--span-bytes", type=int, default=2000)
    ap.add_argument("--overlap", type=int, default=200)
    ap.add_argument("--doc-limit", type=int, default=0, help="0 = no limit")
    if len(sys.argv) == 1:
        ap.print_help()
        return 0
    args = ap.parse_args()

    dump_dir = Path(args.dump_dir)
    out_pack = Path(args.out_pack)
    raw_dir = out_pack / "raw_doc"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Concept map
    concept_map = {f"h{i}": i for i in range(args.bucket_count)}
    Path(args.concept_map_out).write_text(json.dumps(concept_map, ensure_ascii=False), encoding="utf-8")

    shard_path = out_pack / "shards.jsonl.zst"
    zctx = zstd.ZstdCompressor(level=3)

    doc_count = 0
    shard_count = 0

    with shard_path.open("wb") as fh:
        with zctx.stream_writer(fh) as writer:
            for p in iter_files(dump_dir):
                rel = str(p.relative_to(dump_dir))
                try:
                    raw = p.read_bytes()
                except Exception:
                    continue

                # HTML to text
                try:
                    html_text = raw.decode("utf-8", errors="ignore")
                except Exception:
                    continue
                text = strip_html(html_text)
                if len(text) < 200:
                    continue

                # doc_id as hash of rel path
                doc_id = hashlib.sha256(rel.encode("utf-8")).hexdigest()
                doc_path = raw_dir / f"{doc_id}.txt"
                doc_bytes = text.encode("utf-8", errors="ignore")
                doc_path.write_bytes(doc_bytes)

                hash_exact = "sha256:" + hashlib.sha256(doc_bytes).hexdigest()

                # token dist
                tokens = (m.group(0) for m in _TOKEN_RE.finditer(text.lower()))
                dist = make_dist(tokens, args.bucket_count, top_k=64)
                if not dist:
                    continue

                # shard spans (byte offsets)
                step = max(1, args.span_bytes - args.overlap)
                for start in range(0, len(doc_bytes), step):
                    end = min(len(doc_bytes), start + args.span_bytes)
                    if end - start < 200:
                        break
                    rec_id = f"{doc_id}:{start}-{end}"
                    shard = {
                        "rec_id": rec_id,
                        "doc_id": doc_id,
                        "span": {"start": start, "end": end},
                        "dist": dist,
                        "meta": {"title": rel},
                        "provenance": {
                            "hash_exact": hash_exact,
                            "source_id": "wikipedia_ru_top_nopic_2025-08",
                            "license": "CC-BY-SA-3.0",
                        },
                    }
                    writer.write((json.dumps(shard, ensure_ascii=False) + "\n").encode("utf-8"))
                    shard_count += 1

                doc_count += 1
                if args.doc_limit and doc_count >= args.doc_limit:
                    break

    manifest = {
        "pack_name": "ru_wikipedia_top_nopic_2025-08",
        "version": "1",
        "created_at": "2026-01-24",
        "shards": ["shards.jsonl.zst"],
        "raw_docs_dir": "raw_doc",
        "vocab_id": f"hash_bucket_{args.bucket_count}",
        "allow_licenses": ["CC-BY-SA-3.0"],
        "hashes": {},
        "stats": {"docs": doc_count, "shards": shard_count},
    }
    out_pack.mkdir(parents=True, exist_ok=True)
    (out_pack / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
