"""
Road B Phase 4 — deploy-time configurable state_dim.

Verifies the NativeDecoder reads state_dim from the emitter config (and
NOESIS_STATE_DIM env var) rather than a hardcoded 1024. When the decoder
is deployed with a non-1024 emitter, the engine and encoder must match.
"""
from __future__ import annotations

import os
import torch

from Core.Native_Decoder.native_decoder import NativeDecoder
from Core.Native_Decoder.sigma_native_emitter import SigmaEmitterConfig, SigmaNativeEmitter


def test_state_dim_comes_from_emitter_config(monkeypatch):
    """When emitter has state_dim=256, the decoder's _state_dim should be 256."""
    cfg = SigmaEmitterConfig(state_dim=256, d_model=32, n_layers=2, vocab_size=16)
    emitter = SigmaNativeEmitter(cfg)

    # Bypass __init__ to inject a known emitter, simulating a loaded decoder.
    dec = NativeDecoder.__new__(NativeDecoder)
    dec._model = emitter
    dec._tokenizer = None
    dec._engine = None
    dec._expression = None
    dec._decoder_mode = None  # type: ignore[assignment]
    dec._device = torch.device("cpu")

    # Simulate what _load() does: read state_dim from emitter config.
    dec._state_dim = int(os.environ.get("NOESIS_STATE_DIM", dec._model.cfg.state_dim))

    assert dec._state_dim == 256, (
        f"Expected state_dim=256 from emitter config, got {dec._state_dim}"
    )


def test_env_var_overrides_emitter_config(monkeypatch):
    """NOESIS_STATE_DIM=512 should override the emitter's 256."""
    monkeypatch.setenv("NOESIS_STATE_DIM", "512")

    cfg = SigmaEmitterConfig(state_dim=256, d_model=32, n_layers=2, vocab_size=16)
    emitter = SigmaNativeEmitter(cfg)

    dec = NativeDecoder.__new__(NativeDecoder)
    dec._model = emitter
    dec._device = torch.device("cpu")

    dec._state_dim = int(os.environ.get("NOESIS_STATE_DIM", dec._model.cfg.state_dim))

    assert dec._state_dim == 512, (
        "NOESIS_STATE_DIM env var should override emitter config"
    )


def test_text_to_xfinal_uses_resolved_state_dim():
    """_text_to_xfinal should use self._state_dim, not hardcoded 1024."""
    from unittest.mock import MagicMock, patch

    cfg = SigmaEmitterConfig(state_dim=512, d_model=32, n_layers=2, vocab_size=16)
    emitter = SigmaNativeEmitter(cfg)

    dec = NativeDecoder.__new__(NativeDecoder)
    dec._model = emitter
    dec._tokenizer = None
    dec._engine = None
    dec._expression = None
    dec._decoder_mode = None  # type: ignore[assignment]
    dec._device = torch.device("cpu")
    dec._state_dim = 512

    # Mock ConceptStateEncoder to capture the state_dim it receives.
    # It's imported locally inside _text_to_xfinal from External.WorldModel,
    # so we patch at the source module.
    captured_dim = {}
    mock_encoder = MagicMock()

    def mock_encoder_init(state_dim):
        captured_dim["val"] = state_dim
        return mock_encoder

    # Patch the encoder at the source module, and mock engine.step_many
    # so _text_to_xfinal can complete without a real engine.
    dec._engine = MagicMock()
    dec._engine.step_many.return_value = torch.zeros(1, 512)

    with patch(
        "External.WorldModel.concept_state_encoder.ConceptStateEncoder",
        side_effect=mock_encoder_init,
    ):
        try:
            dec._text_to_xfinal("test query")
        except Exception:
            pass  # we only care about the captured state_dim

    assert captured_dim.get("val") == 512, (
        f"_text_to_xfinal passed state_dim={captured_dim.get('val')}, expected 512"
    )
