# Tests/output/test_receipts.py
# Verifies that create_receipt() produces a verifiable signature and
# that payload fields (energies, hashes) are consistent.

import pytest
import torch
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from Core.OSC.dynamics import OperatorSplitEngine
from Core.OSC.icnn import ICNNDirectGrad
from Core.OSC.params import load_params
from External.Output.receipts import create_receipt, verify_receipt


@pytest.mark.fast
def test_receipt_sign_and_verify_ephemeral_keys(tmp_path):
    # Generate ephemeral Ed25519 keys in-memory for test isolation
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

    # Build a tiny engine (CPU, tape-free ICNN)
    d = 12
    params = load_params()
    params.device = torch.device("cpu")
    params.state_dim = d
    params.icnn = ICNNDirectGrad(d=d, m=48, dtype=torch.float32, device=torch.device("cpu"))
    params.snapshot_signing_private_pem = priv_pem
    params.snapshot_signing_public_pem = pub_pem
    params.deterministic = True

    eng = OperatorSplitEngine(params)
    eng.set_seed(123)

    x0 = torch.randn(d)
    x1 = eng.step(x0)

    # Create and verify receipt
    r = create_receipt(eng, x0, x1, trace_id="unit-test")
    assert "signature" in r and "payload" in r

    ok = verify_receipt(r, pub_pem)
    assert ok, "signature should verify with provided public PEM"

    # Recompute energy/hashes and compare to payload fields
    payload = r["payload"]
    e0 = float(eng.energy(x0))
    e1 = float(eng.energy(x1))
    assert pytest.approx(payload["energy_before"], rel=1e-6, abs=1e-6) == e0
    assert pytest.approx(payload["energy_after"],  rel=1e-6, abs=1e-6) == e1

@pytest.mark.fast
def test_receipt_rejects_bad_signature(tmp_path):
    # Keys A
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

    # Keys B
    skB = Ed25519PrivateKey.generate()
    pubB = skB.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    # Engine with keys A (CPU, tape-free ICNN)
    d = 8
    params = load_params()
    params.device = torch.device("cpu")
    params.state_dim = d
    params.icnn = ICNNDirectGrad(d=d, m=32, dtype=torch.float32, device=torch.device("cpu"))
    params.snapshot_signing_private_pem = privA
    params.snapshot_signing_public_pem = pubA
    params.deterministic = True

    eng = OperatorSplitEngine(params)
    eng.set_seed(7)

    x0 = torch.randn(d)
    x1 = eng.step(x0)

    r = create_receipt(eng, x0, x1, trace_id="bad-sig-test")

    # Verify with wrong public key (B) → must fail
    assert not verify_receipt(r, pubB)
