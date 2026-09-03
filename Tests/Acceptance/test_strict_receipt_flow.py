from __future__ import annotations

import base64
import os
import tempfile

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import FastAPI
from fastapi.testclient import TestClient

from API.Routers.reason import router as reason_router
from External.Output.receipts import verify_verifier_receipt
from External.Sim.Api.sim_api import router as sim_router


def make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(sim_router)
    app.include_router(reason_router)
    return app


def _write_ed25519_keys(tmpdir: str) -> tuple[str, str]:
    sk = Ed25519PrivateKey.generate()
    pk = sk.public_key()

    priv_pem = sk.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_pem = pk.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    priv_path = os.path.join(tmpdir, "ed25519_private.pem")
    pub_path = os.path.join(tmpdir, "ed25519_public.pem")

    with open(priv_path, "wb") as f:
        f.write(priv_pem)
    with open(pub_path, "wb") as f:
        f.write(pub_pem)

    return priv_path, pub_path


def test_strict_receipt_flow(monkeypatch) -> None:
    # SIM encryption key
    monkeypatch.setenv("SIM_MASTER_KEY_B64", base64.b64encode(b"\x55" * 32).decode("ascii"))
    # Require Postgres; skip if not configured.
    db_url = os.getenv("SIM_DB_URL", "").strip()
    if not db_url:
        pytest.skip("SIM_DB_URL not set; PostgreSQL required.")
    if "sqlite" in db_url.lower():
        pytest.skip("SQLite is not allowed for this test; PostgreSQL required.")

    with tempfile.TemporaryDirectory() as td:
        priv_path, pub_path = _write_ed25519_keys(td)
        monkeypatch.setenv("NOESIS_SNAPSHOT_SIGN_PRIVATE_PEM_PATH", priv_path)
        monkeypatch.setenv("NOESIS_SNAPSHOT_SIGN_PUBLIC_PEM_PATH", pub_path)

        app = make_app()
        c = TestClient(app)

        headers = {"X-Tenant-Id": "t_d4", "X-User-Id": "u_d4", "X-Role": "USER"}

        # write a SIM record
        w = c.post(
            "/memory/write",
            headers=headers,
            json={"memory_type": "semantic", "payload": "hello world", "profile": "STRICT"},
        )
        assert w.status_code == 200
        record_id = w.json()["record_id"]

        span = c.post(
            "/memory/span",
            headers=headers,
            json={"provenance_id": record_id, "start": 0, "end": 5},
        )
        assert span.status_code == 200
        sha256 = span.json()["sha256"]

        # verify with strict endpoint
        payload = {
            "answer": "hello world",
            "statements": [f"SPAN('{record_id}',0,5) == 'hello'"],
            "citations": [{"source": "SIM", "id": record_id, "start": 0, "end": 5, "sha256": sha256}],
            "mode": "strict",
        }
        r = c.post("/reason/strict", headers=headers, json=payload)
        assert r.status_code == 200
        j = r.json()
        assert j["ok"] is True
        assert j["verifier"]["verdict"] == "pass"
        assert j.get("receipt") is not None

        # verify receipt signature
        with open(pub_path, "rb") as f:
            pub_pem = f.read()
        answer_obj = {
            "answer": payload["answer"],
            "citations": payload["citations"],
            "decoder_manifest_sha256": j.get("decoder_manifest_sha256"),
            "tokenizer_manifest_sha256": j.get("tokenizer_manifest_sha256"),
            "evidence_hashes": j.get("evidence_hashes") or [],
        }
        witness_obj = j["witness"]
        assert verify_verifier_receipt(public_pem=pub_pem, receipt=j["receipt"], answer_obj=answer_obj, witness_obj=witness_obj)
