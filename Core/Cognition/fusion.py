"""
NOESIS-Σ :: Core/Cognition/fusion.py

Context fusion — converts heterogeneous evidence (SIM memories, WKS results,
request text) into a single u(t) tensor injected into OSC as sim_graft.

The OSC engine expects sim_graft shape [state_dim] or [B, state_dim].
text_encoder produces dim=128 vectors. This module projects 128 → state_dim
using a deterministic random projection matrix seeded from a stable hash,
so the projection is reproducible across restarts without storing weights.

Design:
- ContextBundle carries all evidence fields (typed, optional)
- build_context_tensor() fuses everything into one float32 tensor
- infer_mode() heuristically picks "chat" / "code" / "plan" / "analysis"
- No ML weights — fully deterministic, no GPU required for projection
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from External.Sim.Encoders.text_encoder import EncoderConfig, encode_text

# Encoder produces 128-dim vectors; project up to OSC state_dim
_ENCODER_DIM = 128
_PROJECTION_SEED = "noesis-fusion-proj-v1"


def _build_projection_matrix(src_dim: int, tgt_dim: int) -> torch.Tensor:
    """
    Deterministic random projection matrix (src_dim → tgt_dim).

    Seeded from _PROJECTION_SEED so the matrix is identical on every process
    restart. Uses the hash to seed a manual LCG — no torch.manual_seed() call
    that could perturb global RNG state.
    """
    seed_bytes = hashlib.blake2b(
        _PROJECTION_SEED.encode(), digest_size=8
    ).digest()
    seed_int = int.from_bytes(seed_bytes, "little")

    # LCG parameters (Knuth)
    a, c, m = 6364136223846793005, 1442695040888963407, 2**64
    state = seed_int & (m - 1)

    values: List[float] = []
    for _ in range(src_dim * tgt_dim):
        state = (a * state + c) & (m - 1)
        # map to [-1, 1] via unit normal approximation (Box-Muller pair later is
        # overkill; use scaled uniform which has the same Johnson-Lindenstrauss
        # guarantee up to constant factors)
        values.append((state / (m - 1)) * 2.0 - 1.0)

    P = torch.tensor(values, dtype=torch.float32).reshape(src_dim, tgt_dim)
    # Column-normalize so each projected dimension has unit expected norm
    norms = P.norm(dim=0, keepdim=True).clamp(min=1e-8)
    P = P / norms
    return P


# Module-level cache (built once per (src, tgt) pair per process)
_proj_cache: Dict[tuple, torch.Tensor] = {}


def _project(vec128: List[float], state_dim: int) -> torch.Tensor:
    key = (_ENCODER_DIM, state_dim)
    if key not in _proj_cache:
        _proj_cache[key] = _build_projection_matrix(_ENCODER_DIM, state_dim)
    P = _proj_cache[key]
    v = torch.tensor(vec128, dtype=torch.float32)  # [128]
    projected = v @ P  # [state_dim]
    # L2 normalize so injection magnitude stays comparable to OSC norms
    norm = projected.norm().clamp(min=1e-8)
    return projected / norm


def _encode_and_project(text: str, state_dim: int) -> torch.Tensor:
    vec = encode_text(text, cfg=EncoderConfig(dim=_ENCODER_DIM))
    return _project(vec, state_dim)


# ------------------------------------------------------------------ bundle


@dataclass
class ContextBundle:
    """
    All context sources that feed into the OSC step.

    Fields:
        request_text:    raw user request
        session_summary: short summary of session history from SIM
        sim_memories:    list of (content, source_key) from SIM hot/warm
        wks_results:     list of (chunk_text, doc_id) from WKS retrieval
        focus_hint:      optional focus string from prior OSC state
        request_class:   coarse class e.g. "chat", "code", "plan", "analysis"
        state_dim:       target OSC state dimension (default 1024)
        include_sim_memories: if False, SIM memories stay external-only and are
                              excluded from the OSC fusion path
        world_concepts:   list of (description, concept_id) from World Model
        world_facts:      list of (content, fact_id) from World Model temporal facts
        world_relations:  list of relation summary strings from World Model
        skill_procedures: list of (description, skill_id) from Skills planner
    """

    request_text: str = ""
    session_summary: str = ""
    sim_memories: List[tuple] = field(default_factory=list)   # (content, key)
    wks_results: List[tuple] = field(default_factory=list)    # (chunk, doc_id)
    focus_hint: str = ""
    request_class: str = "chat"
    state_dim: int = 1024
    include_sim_memories: bool = True
    world_concepts: List[tuple] = field(default_factory=list)    # (description, concept_id)
    world_facts: List[tuple] = field(default_factory=list)      # (content, fact_id)
    world_relations: List[str] = field(default_factory=list)    # relation summaries
    skill_procedures: List[tuple] = field(default_factory=list) # (description, skill_id)


# ------------------------------------------------------------------ fusion


def build_context_tensor(bundle: ContextBundle) -> torch.Tensor:
    """
    Fuse all evidence in bundle into a single [state_dim] float32 tensor u(t).

    Weighting scheme:
    - request_text:       weight 1.0  (primary)
    - session_summary:    weight 0.6
    - focus_hint:         weight 0.4
    - each SIM memory:    weight 0.3 (decaying by index)
    - each WKS chunk:     weight 0.25 (decaying by index)
    - each WM concept:    weight 0.35 × 0.88^i
    - each WM fact:       weight 0.30 × 0.85^i
    - each WM relation:   weight 0.20 (flat)
    - each skill step:    weight 0.40 (flat, high-value procedural context)

    All components are projected independently then summed and renormalized.
    This is a weighted additive fusion — no cross-attention, O(1) per component.
    """
    d = bundle.state_dim
    accumulator = torch.zeros(d, dtype=torch.float32)
    total_weight = 0.0

    def _add(text: str, weight: float) -> None:
        nonlocal total_weight
        if not text or not text.strip():
            return
        v = _encode_and_project(text, d)
        accumulator.add_(v, alpha=weight)
        total_weight += weight

    _add(bundle.request_text, 1.0)
    _add(bundle.session_summary, 0.6)
    _add(bundle.focus_hint, 0.4)

    if bundle.include_sim_memories:
        for i, (content, _key) in enumerate(bundle.sim_memories):
            w = 0.3 * (0.85 ** i)  # mild decay for older memories
            _add(str(content), w)

    for i, (chunk, _doc_id) in enumerate(bundle.wks_results):
        w = 0.25 * (0.9 ** i)
        _add(str(chunk), w)

    for i, (description, _concept_id) in enumerate(bundle.world_concepts):
        w = 0.35 * (0.88 ** i)
        _add(str(description), w)

    for i, (content, _fact_id) in enumerate(bundle.world_facts):
        w = 0.30 * (0.85 ** i)
        _add(str(content), w)

    for rel_summary in bundle.world_relations:
        _add(str(rel_summary), 0.20)

    for _i, (description, _skill_id) in enumerate(bundle.skill_procedures):
        _add(str(description), 0.40)  # skills are high-value procedural context

    if total_weight < 1e-8:
        # nothing to inject — return zero tensor (engine treats sim_graft=0 as no-op)
        return accumulator

    # Final L2 normalize
    norm = accumulator.norm().clamp(min=1e-8)
    return accumulator / norm


# ------------------------------------------------------------------ mode


_CODE_KEYWORDS = frozenset(
    ["code", "function", "def ", "class ", "implement", "write", "script",
     "bug", "error", "fix", "debug", "test", "refactor", "python", "javascript",
     "typescript", "sql", "bash", "rust", "go ", "java ", "c++"]
)
_PLAN_KEYWORDS = frozenset(
    ["plan", "steps", "roadmap", "how to", "design", "architect", "strategy",
     "sequence", "phases", "approach", "outline", "schedule"]
)
_ANALYSIS_KEYWORDS = frozenset(
    ["analyze", "analysis", "explain", "why", "compare", "difference", "evaluate",
     "assess", "review", "understand", "what is", "how does", "summarize"]
)


def infer_mode(text: str, request_class: str = "chat") -> str:
    """
    Heuristically infer decode mode from text content.

    Returns one of: "chat", "code", "plan", "analysis"
    """
    if request_class in {"code", "plan", "analysis"}:
        return request_class

    lower = text.lower()
    code_score = sum(1 for kw in _CODE_KEYWORDS if kw in lower)
    plan_score = sum(1 for kw in _PLAN_KEYWORDS if kw in lower)
    analysis_score = sum(1 for kw in _ANALYSIS_KEYWORDS if kw in lower)

    best = max(code_score, plan_score, analysis_score)
    if best == 0:
        return "chat"
    if best == code_score:
        return "code"
    if best == plan_score:
        return "plan"
    return "analysis"
