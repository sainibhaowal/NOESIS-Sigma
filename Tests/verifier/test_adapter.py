# Tests/verifier/test_adapter.py
import json

import pytest
import torch
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from Core.OSC.dynamics import OperatorSplitEngine
from Core.OSC.icnn import ICNNDirectGrad
from Core.OSC.params import load_params
from External.Output.receipts import create_receipt
from Core.Verifier.adapter import verify_receipt


@pytest.mark.fast
def test_verify_receipt_signature_and_crosscheck():
    # ephemeral keys
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

    # engine (CPU, tape-free ICNN)
    p = load_params()
    p.device = "cpu"
    p.state_dim = 8
    p.icnn = ICNNDirectGrad(d=8, m=32, dtype=torch.float32, device=torch.device("cpu"))
    p.snapshot_signing_private_pem = priv_pem
    p.snapshot_signing_public_pem = pub_pem
    p.deterministic = True
    eng = OperatorSplitEngine(p)
    eng.set_seed(11)

    xb = torch.randn(8)
    xa = eng.step(xb)

    r = create_receipt(eng, xb, xa, trace_id="t1")
    r_json = json.dumps(r)

    ok = verify_receipt(r_json, pub_pem, engine=eng, x_before=xb, x_after=xa, strict=True)
    assert ok

@pytest.mark.fast
def test_verify_receipt_bad_sig_fails():
    skA = Ed25519PrivateKey.generate()
    pkA = skA.public_key()
    privA = skA.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pubA = pkA.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    skB = Ed25519PrivateKey.generate()
    pubB = skB.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    p = load_params()
    p.device = "cpu"
    p.state_dim = 6
    p.icnn = ICNNDirectGrad(d=6, m=24, dtype=torch.float32, device=torch.device("cpu"))
    p.snapshot_signing_private_pem = privA
    p.snapshot_signing_public_pem = pubA
    p.deterministic = True
    eng = OperatorSplitEngine(p)
    eng.set_seed(23)

    xb = torch.randn(6)
    xa = eng.step(xb)

    r = create_receipt(eng, xb, xa, trace_id="t2")
    r_json = json.dumps(r)

    ok = verify_receipt(r_json, pubB, engine=eng, x_before=xb, x_after=xa, strict=True)
    assert not ok
