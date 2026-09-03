#!/usr/bin/env python3
"""
Reindex WKS embeddings and upsert into Qdrant.

Safe by default:
- Uses existing Qdrant collection if dimensions match.
- Refuses to proceed on dim mismatch unless --reset-collection is set.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Iterable, List, cast

from dotenv import load_dotenv
from sqlalchemy import func
from sqlmodel import Session, create_engine, select

from External.Sim.Models.wks import WksChunk, WksEmbedding
from External.WKS.embeddings import EmbeddingProvider, get_embedding_provider


def _embed_batch(provider, texts: List[str]) -> List[List[float]]:
    # Use model batch encode if available for speed
    if hasattr(provider, "_load"):
        try:
            provider._load()  # type: ignore[attr-defined]
        except Exception:
            pass
    model = getattr(provider, "_model", None)
    if model is not None:
        vecs = model.encode(texts, normalize_embeddings=True)
        return [[float(x) for x in v] for v in vecs]
    return [provider.embed(t) for t in texts]


def _qdrant_client(url: str):
    from qdrant_client import QdrantClient

    return QdrantClient(url=url)


def _ensure_collection(client, name: str, dim: int, reset: bool) -> None:
    from qdrant_client.http import models as rest

    if client.collection_exists(name):
        info = client.get_collection(name)
        vec_cfg = info.config.params.vectors
        size = getattr(vec_cfg, "size", None)
        if size is None and isinstance(vec_cfg, dict):
            for v in vec_cfg.values():
                size = getattr(v, "size", None)
                if size:
                    break
        if size is None:
            raise RuntimeError("Unable to read Qdrant collection vector size")
        if int(size) != int(dim):
            if not reset:
                raise RuntimeError(
                    f"Qdrant collection dim mismatch: expected {dim}, found {size}. "
                    "Use --reset-collection to recreate."
                )
            client.delete_collection(collection_name=name)
        else:
            return
    client.create_collection(
        collection_name=name,
        vectors_config=rest.VectorParams(size=int(dim), distance=rest.Distance.COSINE),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Reindex WKS embeddings and Qdrant vectors.")
    parser.add_argument("--env", default="Runtime/Config/.env", help="Path to .env")
    parser.add_argument("--tenant-id", default=None, help="Filter by tenant_id")
    parser.add_argument("--user-id", default=None, help="Filter by user_id")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None, help="Max chunks to process")
    parser.add_argument("--reset-collection", action="store_true", help="Recreate Qdrant collection")
    parser.add_argument("--dry-run", action="store_true", help="Scan counts only; no writes")
    args = parser.parse_args()

    load_dotenv(args.env)
    sim_db_url = (os.getenv("SIM_DB_URL") or "").strip()
    if not sim_db_url:
        print("SIM_DB_URL missing", file=sys.stderr)
        return 2

    qdrant_url = (os.getenv("QDRANT_URL") or "").strip()
    qdrant_collection = (os.getenv("QDRANT_COLLECTION") or "noesis_wks").strip()
    if not qdrant_url:
        print("QDRANT_URL missing", file=sys.stderr)
        return 2

    engine = create_engine(sim_db_url)
    if args.dry_run:
        # Avoid model load; just report scope
        provider: EmbeddingProvider | None = None
    else:
        provider = get_embedding_provider()

    with Session(engine) as s:
        stmt = select(func.count()).select_from(WksChunk)
        if args.tenant_id:
            stmt = stmt.where(WksChunk.tenant_id == args.tenant_id)
        if args.user_id:
            stmt = stmt.where(WksChunk.user_id == args.user_id)
        total = s.exec(stmt).one()

    if total == 0:
        print("No chunks found. Nothing to reindex.")
        return 0

    if args.dry_run:
        print(f"Dry run: {total} chunk(s) would be reindexed.")
        if args.tenant_id:
            print(f"  tenant_id={args.tenant_id}")
        if args.user_id:
            print(f"  user_id={args.user_id}")
        print("No writes performed.")
        return 0

    # Determine embedding dimension
    if provider is None:
        print("Embedding provider unavailable", file=sys.stderr)
        return 2
    dim = len(provider.embed("dimension probe"))
    if dim <= 0:
        print("Embedding provider returned empty vectors", file=sys.stderr)
        return 2

    client = _qdrant_client(qdrant_url)
    _ensure_collection(client, qdrant_collection, dim, args.reset_collection)

    processed = 0
    offset = 0
    batch = max(1, int(args.batch_size))
    limit = int(args.limit) if args.limit else None

    while True:
        with Session(engine) as s:
            q = select(WksChunk)
            if args.tenant_id:
                q = q.where(WksChunk.tenant_id == args.tenant_id)
            if args.user_id:
                q = q.where(WksChunk.user_id == args.user_id)
            q = q.order_by(cast(Any, WksChunk.id)).offset(offset).limit(batch)
            rows = s.exec(q).all()
            if not rows:
                break

            texts = [r.text or "" for r in rows]
            vecs = _embed_batch(provider, texts)
            if len(vecs) != len(rows):
                raise RuntimeError("Embedding batch size mismatch")

            # Upsert DB embeddings
            ids = [r.id for r in rows]
            existing = s.exec(
                select(WksEmbedding).where(cast(Any, WksEmbedding.chunk_id).in_(ids))
            ).all()
            emb_by_chunk = {e.chunk_id: e for e in existing}
            for r, vec in zip(rows, vecs):
                emb = emb_by_chunk.get(r.id)
                if emb:
                    emb.vector = vec
                    s.add(emb)
                else:
                    s.add(
                        WksEmbedding(
                            chunk_id=r.id,
                            tenant_id=r.tenant_id,
                            user_id=r.user_id,
                            vector=vec,
                        )
                    )
            s.commit()

            # Upsert Qdrant
            payloads = [
                {"doc_id": str(r.doc_id), "tenant_id": r.tenant_id, "user_id": r.user_id}
                for r in rows
            ]
            from qdrant_client.http import models as rest

            client.upsert(
                collection_name=qdrant_collection,
                points=rest.Batch(
                    ids=[str(r.id) for r in rows],
                    vectors=vecs,
                    payloads=payloads,
                ),
            )

        processed += len(rows)
        offset += len(rows)
        print(f"Indexed {processed}/{total}")
        if limit and processed >= limit:
            break

    print("Reindex complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
