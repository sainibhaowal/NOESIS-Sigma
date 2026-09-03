"""
Road B Phase 1 — verify the emitter reads the brain's converged final_state
instead of re-encoding graph text.

Three guarantees:
  1. When final_state is provided to NativeDecoder.decode(), the text re-encode
     path (_text_to_xfinal) is NOT called.
  2. The exact tensor passed in reaches the emitter's generate() as x_final.
  3. Backward compat: when final_state is None, the legacy text path runs.
"""
from __future__ import annotations

import torch

from Core.Cognition.thought_graph import ThoughtGraph, ThoughtNode, NodeType
from Core.Native_Decoder.native_decoder import NativeDecoder
from Core.Native_Decoder.schemas import DecodeRequest
from Core.Native_Decoder.sigma_native_emitter import SigmaEmitterConfig, SigmaNativeEmitter


class _StubTokenizer:
    """Minimal tokenizer matching the interface NativeDecoder.generate uses."""

    def token_to_id(self, text: str) -> int:
        return 2  # EOS — force immediate stop so generate is deterministic

    def encode(self, text: str):
        class _Enc:
            ids = [1, 2]  # BOS, EOS

        return _Enc()

    def decode(self, token_ids):
        return "stub"


def _build_decoder() -> NativeDecoder:
    """Build a NativeDecoder with a tiny in-memory emitter (no weights needed)."""
    from unittest.mock import patch

    cfg = SigmaEmitterConfig(
        state_dim=32,
        d_model=16,
        vocab_size=16,
        prefix_len=2,
    )
    emitter = SigmaNativeEmitter(cfg)

    # The real __init__ loads weights from disk; we bypass it and inject a stub
    # emitter + tokenizer so the test runs without GPU or weights.
    dec = NativeDecoder.__new__(NativeDecoder)
    dec._model = emitter
    dec._tokenizer = _StubTokenizer()
    dec._engine = None
    dec._device = torch.device("cpu")
    dec._expression = None
    dec._decoder_mode = None  # type: ignore[assignment]
    return dec


def _build_graph() -> ThoughtGraph:
    g = ThoughtGraph()
    g.add_node(
        ThoughtNode(
            node_id="f1",
            node_type=NodeType.FACT,
            content="Paris is the capital of France",
        )
    )
    return g


def test_final_state_skips_text_reencode(monkeypatch):
    """Guarantee 1: providing final_state means _text_to_xfinal is never called."""
    dec = _build_decoder()

    called = {"text_to_xfinal": 0}

    def _spy_text_to_xfinal(self, text):
        called["text_to_xfinal"] += 1
        return torch.zeros(1, 32)

    monkeypatch.setattr(
        NativeDecoder, "_text_to_xfinal", _spy_text_to_xfinal, raising=True
    )

    final_state = torch.randn(1, 32)
    dec.decode(_build_graph(), mode="chat", final_state=final_state)

    assert called["text_to_xfinal"] == 0, (
        "Road B violation: _text_to_xfinal was called even though final_state was provided. "
        "The emitter must read the brain's converged state directly."
    )


def test_final_state_tensor_reaches_emitter(monkeypatch):
    """Guarantee 2: the exact tensor passed reaches emitter.generate()."""
    dec = _build_decoder()

    captured = {}

    def _spy_generate(self, x_final, tokenizer, **kwargs):
        captured["x_final"] = x_final.clone()
        return "stub"

    monkeypatch.setattr(SigmaNativeEmitter, "generate", _spy_generate, raising=True)

    final_state = torch.randn(1, 32)
    dec.decode(_build_graph(), mode="chat", final_state=final_state)

    assert "x_final" in captured, "emitter.generate was never reached"
    assert torch.allclose(captured["x_final"], final_state.to(captured["x_final"].device)), (
        "final_state tensor did not reach the emitter unchanged"
    )


def test_legacy_path_runs_when_no_final_state(monkeypatch):
    """Guarantee 3: backward compat — no final_state ⇒ legacy text re-encode runs."""
    dec = _build_decoder()

    called = {"text_to_xfinal": 0}

    def _spy_text_to_xfinal(self, text):
        called["text_to_xfinal"] += 1
        return torch.zeros(1, 32)

    monkeypatch.setattr(
        NativeDecoder, "_text_to_xfinal", _spy_text_to_xfinal, raising=True
    )

    dec.decode(_build_graph(), mode="chat", final_state=None)
    assert called["text_to_xfinal"] >= 1, (
        "Legacy path did not run when final_state=None — backward compat broken"
    )


def test_decode_request_carries_optional_final_state():
    """DecodeRequest schema accepts final_state and defaults to None."""
    req = DecodeRequest(graph_dict={}, mode="chat", trace_id="t1")
    assert req.final_state is None, "default final_state must be None"

    req2 = DecodeRequest(
        graph_dict={}, mode="chat", trace_id="t2", final_state=[0.1, 0.2, 0.3]
    )
    assert req2.final_state == [0.1, 0.2, 0.3], "final_state must round-trip"


def test_coerce_final_state_handles_inputs():
    """The _coerce_final_state helper accepts tensors, lists, and None safely."""
    dec = _build_decoder()

    # None → None
    assert dec._coerce_final_state(None) is None

    # list → tensor [1, N]
    t = dec._coerce_final_state([0.1, 0.2, 0.3])
    assert t is not None and tuple(t.shape) == (1, 3)

    # 1-D tensor → [1, N]
    t2 = dec._coerce_final_state(torch.tensor([0.5, 0.5]))
    assert t2 is not None and tuple(t2.shape) == (1, 2)

    # bad input → None (no crash)
    assert dec._coerce_final_state("not a tensor") is None
