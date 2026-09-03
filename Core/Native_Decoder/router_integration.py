"""
Core/Native_Decoder/router_integration.py

Phase 5: Integration utilities for API/Agent/router.py to support decoder selection.

This module provides helper functions for:
  1. Selecting the correct decoder based on tenant and feature flags
  2. Accepting or rejecting task requests based on acceptance suite status
  3. Recording decoder choice in trace metadata for audit trail
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

from Core.Native_Decoder.config import (
    DecoderConfigManager,
    DecoderMode,
    get_decoder_mode,
)

logger = logging.getLogger(__name__)


class DecoderSelector:
    """Select and validate decoder for a given request."""
    
    @staticmethod
    def get_decoder_for_request(
        tenant_id: str,
        user_id: str,
        session_id: Optional[str] = None,
    ) -> Tuple[DecoderMode, str]:
        """
        Determine which decoder to use for a request.
        
        Args:
            tenant_id: Tenant isolation key
            user_id: User making the request
            session_id: Optional session context
        
        Returns:
            (decoder_mode, reason_string) tuple for audit trail
        
        Logic:
          1. Check if expression layer is available and enabled
          2. If acceptance suite has passed, use expression layer by default
          3. Otherwise, use sigma native emitter path
        """
        config = DecoderConfigManager().get_config()
        mode = config.get_decoder_mode(tenant_id)
        
        # Validate expression layer is available before using
        if mode == DecoderMode.EXPRESSION_LAYER:
            from Core.Native_Decoder.expression_layer import StatefulExpressionLayer
            try:
                # Quick availability check (class import succeeded)
                _ = StatefulExpressionLayer
                reason = "using_expression_layer (acceptance_suite_passed=True)" if config.acceptance_suite_passed else "using_expression_layer (feature_enabled=True)"
            except ImportError:
                logger.warning(
                    f"Expression layer unavailable for tenant={tenant_id}; "
                    f"falling back to sigma_emitter"
                )
                mode = DecoderMode.SIGMA_EMITTER
                reason = "fallback_to_sigma_emitter (expression_layer_unavailable)"
        else:
            reason = "using_sigma_emitter (default)"
        
        return mode, reason
    
    @staticmethod
    def record_decoder_choice_in_trace(
        trace_metadata: dict,
        decoder_mode: DecoderMode,
        reason: str,
    ) -> None:
        """Record decoder choice in trace metadata for audit trail."""
        trace_metadata["decoder_selection"] = {
            "mode": decoder_mode.value,
            "reason": reason,
            "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
        }


class AcceptanceSuiteValidator:
    """Validate requests against acceptance suite criteria."""
    
    # Placeholder for acceptance suite configuration
    _ACCEPTANCE_CRITERIA = {
        "verifier_pass_rate_min": 0.95,      # Must achieve ≥95% verifier pass rate
        "grounding_quality_min": 0.90,        # Must achieve ≥90% grounding quality
        "runtime_stability_min": 0.98,        # Must achieve ≥98% runtime stability
        "api_compatibility_min": 1.0,         # Must maintain 100% API compatibility
        "tenant_isolation_required": True,    # Must preserve tenant isolation
        "regression_test_required": True,     # Must pass regression test
    }
    
    @staticmethod
    def is_acceptance_suite_passed() -> bool:
        """
        Check if acceptance suite has passed.
        
        In production, this would check:
          1. Verifier acceptance rate ≥ threshold
          2. Grounding quality ≥ threshold
          3. Runtime stability ≥ threshold
          4. No API compatibility regressions
          5. Tenant isolation preserved
          6. Regression test suite passed
        
        For now, this reads from the config flag set by testing infrastructure.
        """
        config = DecoderConfigManager().get_config()
        return config.acceptance_suite_passed
    
    @staticmethod
    def validate_request_compatibility(
        decoder_mode: DecoderMode,
        request_data: dict,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that a request is compatible with the selected decoder.
        
        Args:
            decoder_mode: DecoderMode selected for this request
            request_data: Request dict (from TaskRunRequest)
        
        Returns:
            (is_valid, error_message) tuple
        """
        # All modes must support core fields
        required_fields = {"text", "mode"}
        if not all(field in request_data for field in required_fields):
            return False, f"Missing required fields: {required_fields}"
        
        if decoder_mode == DecoderMode.SIGMA_EMITTER:
            return True, None
        
        # Expression layer: validate mode is in supported list
        if decoder_mode == DecoderMode.EXPRESSION_LAYER:
            supported_modes = {"chat", "plan", "analysis", "code", "auto"}
            if request_data.get("mode") not in supported_modes:
                return False, f"Expression layer does not support mode={request_data.get('mode')}"
        
        return True, None


class APICompatibilityGate:
    """Ensure API responses remain compatible across decoder migration."""
    
    @staticmethod
    def get_response_shape_for_decoder(
        decoder_mode: DecoderMode,
    ) -> dict:
        """
        Get the expected response shape for a decoder.
        
        Both decoders must produce the same response schema:
          {
            "task_id": str,
            "status": "complete" | "failed",
            "output": str,
            "phases": [...],
            "session_id": str,
            "trace_id": str,
            "elapsed_ms": int,
          }
        
        Args:
            decoder_mode: DecoderMode selected
        
        Returns:
            Expected response shape template
        """
        return {
            "task_id": "string",
            "status": "complete | failed | capped",
            "output": "string",
            "phases": [
                {
                    "phase_index": "int",
                    "phase_goal": "string",
                    "output_preview": "string",
                    "n_steps": "int",
                    "converged": "bool",
                    "elapsed_ms": "int",
                }
            ],
            "session_id": "string",
            "trace_id": "string",
            "elapsed_ms": "int",
            "decoder_mode": decoder_mode.value,
        }
    
    @staticmethod
    def validate_response_compatibility(
        response_data: dict,
        decoder_mode: DecoderMode,
    ) -> Tuple[bool, Optional[str]]:
        """Validate that response matches expected shape."""
        required_fields = {
            "task_id",
            "status",
            "output",
            "phases",
            "session_id",
            "trace_id",
            "elapsed_ms",
        }
        if not all(field in response_data for field in required_fields):
            missing = required_fields - set(response_data.keys())
            return False, f"Response missing required fields: {missing}"
        
        # Validate phases shape
        if not isinstance(response_data.get("phases"), list):
            return False, "Response.phases must be a list"
        
        return True, None
