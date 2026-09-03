# Verifier/span_store.py
# NOESIS-S -- Span Store (D4)

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from External.Sim.Security.encryption import Envelope, open_envelope
from Core.Verifier.errors import E_LIM_413, E_POL_401, E_REF_404, VerifierError


@dataclass(frozen=True)
class SpanText:
    text: str
    sha256: str | None = None


class SpanStore(Protocol):
    def fetch(
        self,
        *,
        source: str,
        span_id: str,
        start: int,
        end: int,
        snapshot_id: str | None,
    ) -> SpanText:
        raise NotImplementedError

    def get_snapshot_root_hash(self, *, snapshot_id: str) -> str | None:
        return None


@dataclass(frozen=True)
class SpanLimits:
    max_bytes: int = 64_000
    max_spans: int = 256


def safe_fetch_span(
    store: SpanStore,
    *,
    source: str,
    span_id: str,
    start: int,
    end: int,
    snapshot_id: str | None,
    limits: SpanLimits,
) -> tuple[SpanText | None, VerifierError | None]:
    if end <= start or start < 0:
        return None, VerifierError(
            code=E_REF_404, message="Invalid span range", detail=f"{start}:{end}"
        )

    if (end - start) > limits.max_bytes:
        return None, VerifierError(
            code=E_LIM_413, message="Span too large", detail=f"bytes={(end - start)}"
        )

    try:
        st = store.fetch(
            source=source,
            span_id=span_id,
            start=start,
            end=end,
            snapshot_id=snapshot_id,
        )
        if len(st.text.encode("utf-8")) > limits.max_bytes:
            return None, VerifierError(
                code=E_LIM_413, message="Span exceeds byte limit after fetch"
            )
        return st, None
    except PermissionError as e:
        return None, VerifierError(
            code=E_POL_401, message="Policy/consent violation", detail=str(e)
        )
    except KeyError as e:
        return None, VerifierError(
            code=E_REF_404, message="Missing span id", detail=str(e)
        )
    except Exception as e:
        return None, VerifierError(
            code=E_REF_404, message="Span fetch failed", detail=str(e)
        )


# -----------------
# SIM span store
# -----------------


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _decrypt_row(r) -> tuple[str, str]:
    if int(getattr(r, "enc_v", 0)) == 1 and r.payload_ct_b64:
        env = Envelope(
            enc_v=1,
            dek_wrapped_b64=r.dek_wrapped_b64,
            dek_wrap_nonce_b64=r.dek_wrap_nonce_b64,
            payload_ct_b64=r.payload_ct_b64,
            payload_nonce_b64=r.payload_nonce_b64,
            embed_ct_b64=r.embed_ct_b64,
            embed_nonce_b64=r.embed_nonce_b64,
        )
        payload, embed_b64_plain = open_envelope(
            tenant_id=r.tenant_id,
            user_id=r.user_id,
            memory_type=r.memory_type,
            env=env,
        )
        return payload, embed_b64_plain
    return (r.payload or "", r.embed_b64 or "")


class SimSpanStore:
    def __init__(self, *, warm, snapshot_store, tenant_id: str, user_id: str) -> None:
        self._warm = warm
        self._snapshot_store = snapshot_store
        self._tenant_id = tenant_id
        self._user_id = user_id

    def _load_snapshot_row(self, *, snapshot_id: str, record_id: str) -> dict:
        snap = self._warm.get_snapshot(
            snapshot_id=snapshot_id, tenant_id=self._tenant_id, user_id=self._user_id
        )
        if not snap.folder_path:
            raise KeyError("snapshot folder path missing")
        warm_path = Path(snap.folder_path) / "warm_rows.jsonl"
        if not warm_path.exists():
            raise KeyError("warm_rows.jsonl missing")
        for line in warm_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("id") == record_id:
                return obj
        raise KeyError("record not found in snapshot")

    def fetch(
        self,
        *,
        source: str,
        span_id: str,
        start: int,
        end: int,
        snapshot_id: str | None,
    ) -> SpanText:
        if source.upper() != "SIM":
            raise KeyError("source not SIM")

        if snapshot_id:
            # snapshot mode: pull row from snapshot artifacts
            row = self._load_snapshot_row(snapshot_id=snapshot_id, record_id=span_id)

            # row contains encrypted fields; construct a minimal object
            class _Row:
                pass

            r = _Row()
            for k, v in row.items():
                setattr(r, k, v)
            if (
                getattr(r, "tenant_id", None) != self._tenant_id
                or getattr(r, "user_id", None) != self._user_id
            ):
                raise PermissionError("tenant/user mismatch")
            payload, _ = _decrypt_row(r)
        else:
            r = self._warm.get_record_by_id(record_id=span_id)
            if r is None:
                raise KeyError("record not found")
            r_any = cast(Any, r)
            if r_any.tenant_id != self._tenant_id or r_any.user_id != self._user_id:
                raise PermissionError("tenant/user mismatch")
            payload, _ = _decrypt_row(r_any)

        txt = payload or ""
        s = max(0, min(int(start), len(txt)))
        e = max(s, min(int(end), len(txt)))
        part = txt[s:e]
        return SpanText(text=part, sha256=_sha256_text(part))

    def get_snapshot_root_hash(self, *, snapshot_id: str) -> str | None:
        try:
            snap = self._warm.get_snapshot(
                snapshot_id=snapshot_id,
                tenant_id=self._tenant_id,
                user_id=self._user_id,
            )
            return snap.root_hash
        except Exception:
            return None


# -----------------
# WKS span store
# -----------------


class WKSSpanStore:
    def __init__(self, *, pack_root: Path, index_dir: Path) -> None:
        from External.WKS.pack_store import load_pack

        self._pack = load_pack(pack_root)
        name = self._pack.manifest.pack_name
        meta_path = index_dir / f"{name}.meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"WKS meta missing: {meta_path}")
        meta = json.loads(meta_path.read_text("utf-8"))
        self._rec_ids = meta.get("rec_ids", [])
        self._doc_ids = meta.get("doc_ids", [])
        self._starts = meta.get("starts", [])
        self._ends = meta.get("ends", [])
        self._hashes = meta.get("hashes", [])
        self._licenses = meta.get("licenses", [])
        self._rec_index = {rid: i for i, rid in enumerate(self._rec_ids)}

    def fetch(
        self,
        *,
        source: str,
        span_id: str,
        start: int,
        end: int,
        snapshot_id: str | None,
    ) -> SpanText:
        if source.upper() != "WKS":
            raise KeyError("source not WKS")
        if span_id not in self._rec_index:
            raise KeyError("rec_id not found")
        i = self._rec_index[span_id]
        lic = self._licenses[i]
        if not self._pack.license_policy.is_allowed(lic):
            raise PermissionError("license not allowed")
        doc_id = self._doc_ids[i]
        # provenance check: raw doc hash must match
        if not self._pack.verify_raw_doc_hash(doc_id, self._hashes[i]):
            raise PermissionError("hash mismatch")
        # fetch excerpt
        data = self._pack.get_excerpt_bytes(doc_id, self._starts[i], self._ends[i])
        if data is None:
            raise KeyError("doc not found")
        txt = data.decode("utf-8", errors="replace")
        s = max(0, min(int(start), len(txt)))
        e = max(s, min(int(end), len(txt)))
        part = txt[s:e]
        return SpanText(text=part, sha256=self._hashes[i])


class WKSSpanStoreDB:
    def __init__(self, *, warm, tenant_id: str, user_id: str) -> None:
        self._warm = warm
        self._tenant_id = tenant_id
        self._user_id = user_id

    def fetch(
        self,
        *,
        source: str,
        span_id: str,
        start: int,
        end: int,
        snapshot_id: str | None,
    ) -> SpanText:
        if source.upper() != "WKS":
            raise KeyError("source not WKS")
        from sqlmodel import Session, select

        from External.Sim.Models.wks import WksChunk
        from uuid import UUID

        with Session(self._warm.engine) as s:
            try:
                rid = UUID(str(span_id))
            except Exception:
                raise KeyError("rec_id not found")
            rec = s.exec(select(WksChunk).where(WksChunk.id == rid)).first()
            if rec is None:
                raise KeyError("rec_id not found")
            if rec.tenant_id != self._tenant_id or rec.user_id != self._user_id:
                raise PermissionError("tenant/user mismatch")
            txt = rec.text or ""
            s_idx = max(0, min(int(start), len(txt)))
            e_idx = max(s_idx, min(int(end), len(txt)))
            part = txt[s_idx:e_idx]
            return SpanText(text=part, sha256=rec.sha256)


class MultiSpanStore:
    def __init__(
        self, *, sim_store: SimSpanStore, wks_store: WKSSpanStore | None = None
    ) -> None:
        self._sim = sim_store
        self._wks = wks_store

    def fetch(
        self,
        *,
        source: str,
        span_id: str,
        start: int,
        end: int,
        snapshot_id: str | None,
    ) -> SpanText:
        src = source.upper()
        if src == "SIM":
            return self._sim.fetch(
                source=source,
                span_id=span_id,
                start=start,
                end=end,
                snapshot_id=snapshot_id,
            )
        if src == "WKS":
            if self._wks is None:
                raise KeyError("WKS store not configured")
            return self._wks.fetch(
                source=source,
                span_id=span_id,
                start=start,
                end=end,
                snapshot_id=snapshot_id,
            )
        raise KeyError("unknown source")

    def get_snapshot_root_hash(self, *, snapshot_id: str) -> str | None:
        return self._sim.get_snapshot_root_hash(snapshot_id=snapshot_id)
