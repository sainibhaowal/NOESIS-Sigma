"""
Phase A deterministic decoder. Zero VRAM, zero model.
Linearizes ThoughtGraph nodes directly into structured text.
The output is grounded by construction — every word comes from a graph node.
"""
from __future__ import annotations

import time

from Core.Cognition.thought_graph import ThoughtGraph
from Core.Native_Decoder.modes import normalize_mode
from Core.Native_Decoder.renderer import render
from Core.Native_Decoder.schemas import DecodeRequest, DecodeResponse


class StubDecoder:
    """
    Phase A renderer. Replaced in Phase C by a trained NativeDecoder.
    """

    MODEL_ID = "stub_decoder_v1"
    LANE = "native"

    def decode(self, graph: ThoughtGraph, mode: str = "chat") -> str:
        mode = normalize_mode(mode)
        t0 = time.monotonic()
        text = render(graph, mode)
        elapsed = (time.monotonic() - t0) * 1000
        _ = elapsed  # available for tracing if needed
        return text

    def decode_full(self, req: DecodeRequest) -> DecodeResponse:
        graph = ThoughtGraph.from_dict(req.graph_dict)
        mode = normalize_mode(req.mode)
        t0 = time.monotonic()
        text = render(graph, mode)
        elapsed = (time.monotonic() - t0) * 1000
        return DecodeResponse(
            text=text,
            mode=mode,
            trace_id=req.trace_id,
            grounded=True,
            elapsed_ms=elapsed,
        )
