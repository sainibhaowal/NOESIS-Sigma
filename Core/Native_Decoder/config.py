"""
Core/Native_Decoder/config.py

Runtime configuration for decoder selection (native sigma emitter vs expression layer).

Feature flags:
  - use_expression_layer: Prefer deterministic graph renderer
  - fallback_to_sigma_emitter: If expression unavailable, use sigma conv emitter path
  - acceptance_suite_passed: Gate for switching defaults
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DecoderMode(str, Enum):
    """Decoder selection mode."""

    SIGMA_EMITTER = "sigma_emitter"
    EXPRESSION_LAYER = "expression_layer"
    AUTO = "auto"


@dataclass
class DecoderConfig:
    """Runtime decoder selection configuration."""

    use_expression_layer: bool = False
    fallback_to_sigma_emitter: bool = True
    acceptance_suite_passed: bool = False
    decoder_mode: DecoderMode = DecoderMode.AUTO
    sigma_emitter_path: Optional[str] = None
    expression_layer_path: Optional[str] = None
    tenant_expression_layer_enabled: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.tenant_expression_layer_enabled is None:
            self.tenant_expression_layer_enabled = {}

    def is_expression_layer_enabled_for_tenant(self, tenant_id: str) -> bool:
        if tenant_id in self.tenant_expression_layer_enabled:
            return bool(self.tenant_expression_layer_enabled[tenant_id])
        return self.use_expression_layer

    def get_decoder_mode(self, tenant_id: Optional[str] = None) -> DecoderMode:
        if self.decoder_mode in {DecoderMode.SIGMA_EMITTER, DecoderMode.EXPRESSION_LAYER}:
            return self.decoder_mode
        if self.acceptance_suite_passed:
            return DecoderMode.EXPRESSION_LAYER
        if tenant_id and self.is_expression_layer_enabled_for_tenant(tenant_id):
            return DecoderMode.EXPRESSION_LAYER
        if self.use_expression_layer:
            return DecoderMode.EXPRESSION_LAYER
        return DecoderMode.SIGMA_EMITTER


class DecoderConfigManager:
    _instance: Optional["DecoderConfigManager"] = None
    _config: Optional[DecoderConfig] = None

    def __new__(cls) -> "DecoderConfigManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if not getattr(self, "_initialized", False):
            self._config = self._load_from_env()
            self._initialized = True

    @staticmethod
    def _load_from_env() -> DecoderConfig:
        use_expr = os.getenv("USE_EXPRESSION_LAYER", "false").lower() == "true"
        # Legacy env name still honored
        fallback_sigma = os.getenv("DECODER_FALLBACK_TO_SIGMA_EMITTER", "").strip()
        if not fallback_sigma:
            legacy = os.getenv("DECODER_FALLBACK_TO_BRIDGE", "true").lower() == "true"
            fallback_sigma = "true" if legacy else "false"
        fallback = fallback_sigma.lower() == "true"
        acceptance = os.getenv("ACCEPTANCE_SUITE_PASSED", "false").lower() == "true"

        mode_str = os.getenv("DECODER_MODE", "auto").lower()
        if mode_str in {"sigma_emitter", "sigma", "native_emitter"}:
            mode = DecoderMode.SIGMA_EMITTER
        elif mode_str == "expression_layer":
            mode = DecoderMode.EXPRESSION_LAYER
        elif mode_str in {"bridge", "osc_native"}:
            mode = DecoderMode.SIGMA_EMITTER
        else:
            mode = DecoderMode.AUTO

        sigma_path = os.getenv("SIGMA_EMITTER_PATH") or os.getenv("BRIDGE_DECODER_PATH")
        expr_path = os.getenv("EXPRESSION_LAYER_PATH")

        return DecoderConfig(
            use_expression_layer=use_expr,
            fallback_to_sigma_emitter=fallback,
            acceptance_suite_passed=acceptance,
            decoder_mode=mode,
            sigma_emitter_path=sigma_path,
            expression_layer_path=expr_path,
        )

    def get_config(self) -> DecoderConfig:
        return self._config  # type: ignore[return-value]

    def set_config(self, config: DecoderConfig) -> None:
        self._config = config

    def get_decoder_mode(self, tenant_id: Optional[str] = None) -> DecoderMode:
        return self.get_config().get_decoder_mode(tenant_id)

    def enable_expression_layer_for_tenant(self, tenant_id: str) -> None:
        self.get_config().tenant_expression_layer_enabled[tenant_id] = True

    def disable_expression_layer_for_tenant(self, tenant_id: str) -> None:
        self.get_config().tenant_expression_layer_enabled[tenant_id] = False


def get_decoder_config() -> DecoderConfig:
    return DecoderConfigManager().get_config()


def get_decoder_mode(tenant_id: Optional[str] = None) -> DecoderMode:
    return DecoderConfigManager().get_decoder_mode(tenant_id)
