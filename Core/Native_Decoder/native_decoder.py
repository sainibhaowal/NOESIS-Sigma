"""
NativeDecoder — production inference using SigmaNativeEmitter (causal conv head).

Interface matches StubDecoder:
    decode(graph: ThoughtGraph, mode: str, final_state=None) -> str
    decode_full(req: DecodeRequest) -> DecodeResponse

Flow (Road B):
    1. Accept converged brain state (LoopResult.final_state) directly, OR
       fall back to serializing ThoughtGraph text → ConceptStateEncoder → x_final
    2. SigmaNativeEmitter generates discrete tokens (no Transformer attention)
    3. Generation halts when the brain state stabilizes (convergence-based)

state_dim is deploy-time configurable via NOESIS_STATE_DIM env var. Falls back
to the emitter config's state_dim, then 1024 legacy default.

Weights:
    Core/Native_Decoder/weights/sigma_native_emitter.pt
    Core/Native_Decoder/weights/sigma_native_emitter_config.json (optional sidecar)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch

_ROOT = Path(__file__).parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Core.Cognition.thought_graph import ThoughtGraph
from Core.Native_Decoder.config import DecoderConfig, DecoderConfigManager, DecoderMode
from Core.Native_Decoder.modes import normalize_mode
from Core.Native_Decoder.schemas import DecodeRequest, DecodeResponse
from Core.Native_Decoder.sigma_native_emitter import (
    SigmaEmitterConfig,
    SigmaNativeEmitter,
    is_sigma_checkpoint_file,
)

_WEIGHTS_DIR = Path(__file__).parent / "weights"
_EMITTER_PT = "sigma_native_emitter.pt"
_EMITTER_CFG = "sigma_native_emitter_config.json"
_TOK_DIR = _WEIGHTS_DIR / "tokenizer"


class NativeDecoder:
    """OSC-state-conditioned token emitter (sigma native conv stack)."""

    MODEL_ID = "sigma_native_emitter_v1"
    LANE = "sigma_native"

    def __init__(
        self, weights_dir: str | Path = _WEIGHTS_DIR, config: Optional[DecoderConfig] = None
    ) -> None:
        self._weights_dir = Path(weights_dir)
        self._config = config or DecoderConfigManager().get_config()
        self._model: Optional[SigmaNativeEmitter] = None
        self._tokenizer = None
        self._engine = None
        self._expression = self._load_expression_layer()
        self._device = self._pick_device()
        self._decoder_mode = self._config.get_decoder_mode()
        self._load()

    def decode(
        self,
        graph: ThoughtGraph,
        mode: str = "chat",
        trace_id: Optional[str] = None,
        final_state: Optional[torch.Tensor] = None,
    ) -> str:
        mode = normalize_mode(mode)
        if self._decoder_mode == DecoderMode.EXPRESSION_LAYER and self._expression is not None:
            return self._expression.render(graph, mode=mode)
        if self._model is not None:
            return self._generate(graph, mode, final_state=final_state)
        return self._fallback(graph, mode)

    def decode_full(self, req: DecodeRequest) -> DecodeResponse:
        graph = ThoughtGraph.from_dict(req.graph_dict)
        mode = normalize_mode(req.mode)
        t0 = time.monotonic()

        # Road B: prefer the brain's converged final_state when provided.
        final_state_tensor = self._coerce_final_state(req.final_state)

        if self._decoder_mode == DecoderMode.EXPRESSION_LAYER and self._expression is not None:
            text = self._expression.render(graph, mode=mode)
            decoder_used = "expression_layer"
        elif self._model is not None:
            text = self._generate(graph, mode, final_state=final_state_tensor)
            decoder_used = "sigma_native_emitter"
        else:
            text = self._fallback(graph, mode)
            decoder_used = "stub"

        elapsed = (time.monotonic() - t0) * 1000
        try:
            from loguru import logger

            logger.info(
                f"[decode] trace_id={req.trace_id} decoder={decoder_used} elapsed_ms={elapsed:.1f}"
            )
        except Exception:
            pass

        return DecodeResponse(
            text=text,
            mode=mode,
            trace_id=req.trace_id,
            grounded=bool(text and len(text.strip()) > 0),
            elapsed_ms=elapsed,
        )

    @classmethod
    def is_available(cls, weights_dir: str | Path = _WEIGHTS_DIR) -> bool:
        d = Path(weights_dir)
        pt = d / _EMITTER_PT
        return pt.is_file() and is_sigma_checkpoint_file(pt)

    def _load(self) -> None:
        pt_path = self._weights_dir / _EMITTER_PT
        if not self.is_available(self._weights_dir):
            return
        try:
            cfg_path = self._weights_dir / _EMITTER_CFG
            if cfg_path.is_file():
                _ = SigmaEmitterConfig.load(cfg_path)
            self._model = SigmaNativeEmitter.load(pt_path, device=self._device)
            self._model.eval()

            # Road B: state_dim comes from the emitter config (single source of truth),
            # falling back to env NOESIS_STATE_DIM, then 1024 legacy default.
            # This ensures the decoder's engine matches the emitter's state_dim
            # when deployed at non-1024 dimensions.
            self._state_dim = int(os.environ.get("NOESIS_STATE_DIM", self._model.cfg.state_dim))

            try:
                from tokenizers import Tokenizer
                self._tokenizer = Tokenizer.from_file(str(self._weights_dir / "tokenizer" / "tokenizer.json"))
            except Exception:
                self._tokenizer = None

            from Core.OSC.dynamics import EngineParams, OperatorSplitEngine

            params = EngineParams(
                state_dim=self._state_dim,
                dt=0.005,
                max_norm=20.0,
                implicit_iters=3,
                implicit_tol=1e-5,
                device=self._device,
                dtype=torch.float32,
            )
            self._engine = OperatorSplitEngine(params)

            # Verify emitter and engine agree on state_dim
            if self._model.cfg.state_dim != self._state_dim:
                from loguru import logger
                logger.warning(
                    f"NativeDecoder: emitter state_dim={self._model.cfg.state_dim} "
                    f"!= engine state_dim={self._state_dim}. "
                    "This will corrupt state passing. Set NOESIS_STATE_DIM to match."
                )

            total = sum(p.numel() for p in self._model.parameters())
            from loguru import logger

            logger.info(f"NativeDecoder (sigma emitter) loaded: {total:,} params on {self._device} state_dim={self._state_dim}")
        except Exception as exc:
            from loguru import logger

            logger.warning(f"NativeDecoder: failed to load ({exc}), using stub fallback")
            self._model = None

    def _graph_to_text(self, graph: ThoughtGraph) -> str:
        nodes = getattr(graph, "_nodes", {})
        if isinstance(nodes, dict):
            nodes = list(nodes.values())
        parts = []
        for n in nodes:
            content = getattr(n, "content", None) or str(n)
            if content:
                parts.append(str(content).strip())
        return " ".join(parts[:8]) if parts else "unknown"

    def _text_to_xfinal(self, text: str) -> torch.Tensor:
        from External.WorldModel.concept_state_encoder import ConceptStateEncoder

        # Road B: use the resolved state_dim, not a hardcoded 1024.
        state_dim = getattr(self, "_state_dim", 1024)
        encoder = ConceptStateEncoder(state_dim=state_dim)
        x0 = encoder.encode_query(text, device=self._device).to(torch.float32).unsqueeze(0)
        with torch.no_grad():
            x_final = self._engine.step_many(x0, n_steps=50)
        if x_final.dim() == 1:
            x_final = x_final.unsqueeze(0)
        return x_final

    def _coerce_final_state(self, final_state) -> Optional[torch.Tensor]:
        """
        Convert a raw sequence/list/tensor of floats into a [1, state_dim] tensor
        on the decoder's device. Returns None if input is None or invalid, so the
        caller falls back to text re-encoding safely.
        """
        if final_state is None:
            return None
        try:
            if isinstance(final_state, torch.Tensor):
                t = final_state.detach().clone().to(torch.float32)
            else:
                t = torch.as_tensor(list(final_state), dtype=torch.float32)
            if t.dim() == 1:
                t = t.unsqueeze(0)
            return t.to(self._device)
        except Exception:
            return None

    def _generate(self, graph: ThoughtGraph, mode: str, final_state: Optional[torch.Tensor] = None) -> str:
        if self._tokenizer is None:
            return self._fallback(graph, mode)
        # Road B: prefer the brain's actual converged state. Only re-encode
        # graph text as a fallback when no final_state was threaded through.
        if final_state is not None:
            x_final = final_state.to(self._device)
            if x_final.dim() == 1:
                x_final = x_final.unsqueeze(0)
        else:
            query_text = self._graph_to_text(graph)
            x_final = self._text_to_xfinal(query_text)

        text = self._model.generate(
            x_final,
            self._tokenizer,
            max_new_tokens=None,
            temperature=0.8,
            top_p=0.9,
            engine=self._engine,
            # F3: Dropped hard logit masking in favor of soft grounding loss
            # allowed_token_ids=allowed_tokens,
        )
        from loguru import logger
        logger.debug(f"[DEBUG NATIVE DECODER] RAW TEXT GENERATED: {text!r}")
        if not text or len(text.strip()) < 3:
            logger.debug("[DEBUG NATIVE DECODER] Text too short, falling back to stub!")
            return self._fallback(graph, mode)
        
        # Clean BPE visual placeholders (e.g. Ġ -> space, Ċ -> newline, ĉ -> tab)
        cleaned = text.replace("Ġ", " ").replace("Ċ", "\n").replace("ĉ", "\t")
        
        # F2: Hard Logical Verification Gate
        from Core.Verifier.graph_verifier import ThoughtGraphVerifier
        verifier = ThoughtGraphVerifier()
        allowed, reason = verifier.gate_output(graph, cleaned)
        if not allowed:
            from loguru import logger
            logger.debug(f"[DEBUG NATIVE DECODER] Verifier blocked output: {reason}. Falling back to Expression Layer!")
            return self._fallback(graph, mode)
        
        # Collapse multiple spaces while preserving formatting
        import re
        parts = []
        for line in cleaned.splitlines():
            collapsed = re.sub(r" +", " ", line)
            parts.append(collapsed)
        cleaned_text = "\n".join(parts).strip()
        
        return cleaned_text

    def _fallback(self, graph: ThoughtGraph, mode: str) -> str:
        from Core.Cognition.thought_graph import NodeType
        
        nodes = getattr(graph, "_nodes", {})
        if isinstance(nodes, dict):
            nodes = list(nodes.values())
            
        facts = [n.content for n in nodes if getattr(n, "node_type", None) == NodeType.FACT and getattr(n, "content", None)]
        plans = [n.content for n in nodes if getattr(n, "node_type", None) == NodeType.PLAN and getattr(n, "content", None)]
        
        fact_str = " | ".join(facts) if facts else "None"
        plan_str = " | ".join(plans) if plans else "None"
        
        return f"Based on: {fact_str}. Reasoning: {plan_str}."

    @staticmethod
    def _load_expression_layer():
        import os

        enabled = (os.getenv("NOESIS_NATIVE_EXPRESSION_LAYER") or "0").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if not enabled:
            return None
        try:
            from Core.Native_Decoder.expression_layer import StatefulExpressionLayer

            return StatefulExpressionLayer()
        except Exception:
            return None

    @staticmethod
    def _pick_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
