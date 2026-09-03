"""
SigmaNativeEmitter — NOESIS-Σ Ultra-Frontier Foundation Edition.

Architecture Upgrades:
  - RMSNorm: Replaced standard LayerNorm for frontier-scale stability.
  - SwiGLU Gating: Implemented SiLU Gated Linear Units for high-precision linguistic selection.
  - Geometric Residual Scaling: Implemented 1/sqrt(N) scaling for signal preservation in 24+ layer stacks.
  - Causal Conv GLU: Integrated gating directly into the convolutional mixer.

This is a post-transformer foundation-scale emitter designed for 
deterministic, high-nuance vocalization of OSC attractor states.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

@dataclass
class SigmaEmitterConfig:
    state_dim: int = 1024
    d_model: int = 1024
    n_layers: int = 24
    kernel_size: int = 7
    prefix_len: int = 24
    vocab_size: int = 32_768
    # Legacy safety cap only; the emitter now uses streaming state-space mixing
    # rather than a fixed receptive-field window.
    max_seq_len: int = 512
    dropout: float = 0.0

    # Retrograde coupling params
    retro_alpha: float = 0.1
    retro_decay: float = 0.95
    retro_micro_steps: int = 5
    retro_u_max: float = 1.0

    # Road B: brain-controlled generation halt.
    # Generation continues while the brain state keeps changing; halts once
    # the state stabilizes (delta < threshold) for `convergence_window`
    # consecutive tokens. This makes the brain - not a fixed token cap -
    # decide when the answer is complete. The token budget from
    # _estimate_generation_budget remains as a safety ceiling only.
    convergence_threshold: float = 1e-3
    convergence_window: int = 3

    @classmethod
    def small(cls) -> "SigmaEmitterConfig":
        return cls(d_model=512, n_layers=8, kernel_size=5, vocab_size=16_384)

    @classmethod
    def medium(cls) -> "SigmaEmitterConfig":
        return cls(d_model=768, n_layers=12, kernel_size=5, vocab_size=32_768)

    @classmethod
    def foundation(cls) -> "SigmaEmitterConfig":
        """The 'Ultra-Level' Foundation preset."""
        return cls(d_model=1024, n_layers=24, kernel_size=7, vocab_size=32768)

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "SigmaEmitterConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

class RMSNorm(nn.Module):
    """Llama-class Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class SwiGLU(nn.Module):
    """SiLU Gated Linear Unit — the engine of modern LLM intelligence."""
    def __init__(self, d_model: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_model * 4, bias=False)
        self.w2 = nn.Linear(d_model * 4, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_model * 4, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class UltraFrontierBlock(nn.Module):
    """
    Ultra-Frontier mixer with a streaming state-space path.

    The old causal-conv path stays in place for checkpoint compatibility,
    but the block now also carries a recurrent scan so the model can process
    arbitrarily long sequences without a fixed receptive-field ceiling.
    """
    def __init__(self, d_model: int, kernel_size: int, dilation: int = 1, layer_idx: int = 1):
        super().__init__()
        self.layer_idx = layer_idx
        self.norm = RMSNorm(d_model)
        
        self.dw = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            groups=d_model, bias=False, dilation=dilation
        )
        self.mixer_gate = nn.Linear(d_model, d_model, bias=False)
        
        self.ffn = SwiGLU(d_model)
        self.ffn_norm = RMSNorm(d_model)

        # Streaming state-space path for unbounded sequence length.
        self.ssm_norm = RMSNorm(d_model)
        self.ssm_in = nn.Linear(d_model, d_model * 2, bias=False)
        self.ssm_state = nn.Linear(d_model, d_model, bias=False)
        self.ssm_out = nn.Linear(d_model, d_model, bias=False)
        self.ssm_decay_raw = nn.Parameter(torch.tensor(-0.5))

    def forward(self, x: Tensor) -> Tensor:
        # 1. Convolutional Mixing with Gating
        residual = x
        h = self.norm(x)
        hc = h.transpose(1, 2)
        pad = (self.dw.kernel_size[0] - 1) * self.dw.dilation[0]
        hc = F.pad(hc, (pad, 0))
        y = self.dw(hc).transpose(1, 2)
        
        y = y * torch.sigmoid(self.mixer_gate(h))
        
        # Geometric Residual Scaling (1/sqrt(depth))
        x = residual + y * (1.0 / math.sqrt(self.layer_idx + 1))

        # 2. Frontier SwiGLU FFN
        residual = x
        y = self.ffn(self.ffn_norm(x))
        x = residual + y * (1.0 / math.sqrt(self.layer_idx + 1))

        # 3. Streaming SSM path: recurrent scan across the whole sequence.
        # This is the part that removes the finite receptive-field ceiling.
        ssm_in = self.ssm_norm(x)
        decay = torch.sigmoid(self.ssm_decay_raw).clamp(0.0, 0.9999)
        state = torch.zeros(x.shape[0], x.shape[-1], device=x.device, dtype=x.dtype)
        ssm_out: list[Tensor] = []
        for t in range(ssm_in.shape[1]):
            step = ssm_in[:, t, :]
            gate, drive = self.ssm_in(step).chunk(2, dim=-1)
            proposal = torch.tanh(self.ssm_state(step)) * torch.sigmoid(gate)
            state = decay * state + proposal
            mixed = self.ssm_out(torch.tanh(state) * torch.sigmoid(drive))
            ssm_out.append(mixed.unsqueeze(1))
        ssm = torch.cat(ssm_out, dim=1) if ssm_out else torch.zeros_like(x)
        return x + ssm * (1.0 / math.sqrt(self.layer_idx + 1))

class StructuredPrefixProjector(nn.Module):
    """Ultra-Frontier Projector mapping x* to distinct semantic regions."""
    def __init__(self, state_dim: int, d_model: int, prefix_len: int):
        super().__init__()
        self.prefix_len = int(prefix_len)
        self.d_model = int(d_model)
        hidden = max(self.d_model * 2, 128)
        self.fact_proj = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.SiLU(), nn.Linear(hidden, self.prefix_len * self.d_model)
        )
        self.plan_proj = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.SiLU(), nn.Linear(hidden, self.prefix_len * self.d_model)
        )
        self.out_proj  = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.SiLU(), nn.Linear(hidden, self.prefix_len * self.d_model)
        )

    def forward(self, x_final: Tensor) -> Tensor:
        B = x_final.shape[0]
        f = self.fact_proj(x_final).view(B, self.prefix_len, self.d_model)
        p = self.plan_proj(x_final).view(B, self.prefix_len, self.d_model)
        o = self.out_proj(x_final).view(B, self.prefix_len, self.d_model)
        return (f + p + o) / 3.0

class SigmaNativeEmitter(nn.Module):
    """The 'Ultra-Frontier' Voice: 24-Layer Foundation-Scale Emitter."""

    def __init__(self, cfg: SigmaEmitterConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.state_proj = StructuredPrefixProjector(cfg.state_dim, cfg.d_model, cfg.prefix_len)
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        
        self.blocks = nn.ModuleList([
            UltraFrontierBlock(
                cfg.d_model, cfg.kernel_size, 
                dilation=2**(i % 8),
                layer_idx=i
            )
            for i in range(cfg.n_layers)
        ])
        
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        from Core.OSC.retrograde import RetrogradeController
        self.retrograde = RetrogradeController(
            d_model=cfg.d_model, state_dim=cfg.state_dim,
            alpha=cfg.retro_alpha, decay=cfg.retro_decay,
            micro_steps=cfg.retro_micro_steps, u_max=cfg.retro_u_max,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02 / math.sqrt(self.cfg.n_layers))
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self, x_final: Tensor, input_ids: Tensor, 
        labels: Optional[Tensor] = None, target_nodes: Optional[Tensor] = None
    ) -> dict:
        pl = self.cfg.prefix_len
        prefix = self.state_proj(x_final)
        tok_emb = self.embed(input_ids.clamp(min=0))
        
        x = torch.cat([prefix, tok_emb], dim=1)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        token_logits = logits[:, pl:, :]

        loss = None
        if labels is not None:
            loss_ce = F.cross_entropy(
                token_logits[:, :-1, :].contiguous().view(-1, self.cfg.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
            mean_prefix = prefix.mean(dim=1)
            mean_soft = target_nodes.mean(dim=1) if target_nodes is not None else tok_emb.mean(dim=1)
            common_dim = min(mean_soft.shape[-1], mean_prefix.shape[-1])
            l_ground = 1.0 - F.cosine_similarity(
                mean_soft[..., :common_dim], mean_prefix[..., :common_dim], dim=-1
            ).mean()
            loss = loss_ce + 0.1 * l_ground

        return {"logits": token_logits, "loss": loss}

    @torch.inference_mode()
    def generate(
        self, x_final: Tensor, tokenizer: object, 
        max_new_tokens: Optional[int] = None, temperature: float = 0.7, top_p: float = 0.9,
        bos_token_id: int = 1, eos_token_id: int = 2,
        engine: Any = None, **kwargs
    ) -> str:
        device = x_final.device
        curr_x = x_final.clone()
        # Ensure input_ids is [B, L]
        input_ids = torch.tensor([[bos_token_id]], device=device)
        generated = [bos_token_id]

        if max_new_tokens is None:
            max_new_tokens = self._estimate_generation_budget(x_final, input_ids)

        # Road B: track consecutive tokens where the brain state barely moved.
        # When it reaches convergence_window, the brain has settled - halt.
        convergence_streak = 0

        for _ in range(max_new_tokens):
            outputs = self.forward(curr_x, input_ids)
            next_token_logits = outputs["logits"][:, -1, :] / max(temperature, 1e-5)

            allowed_token_ids = kwargs.get("allowed_token_ids")
            if allowed_token_ids is not None:
                allowed = torch.zeros_like(next_token_logits, dtype=torch.bool)
                allowed_ids = [int(i) for i in allowed_token_ids]
                if allowed_ids:
                    allowed[:, allowed_ids] = True
                    next_token_logits = next_token_logits.masked_fill(~allowed, -float("Inf"))

            # Nucleus Sampling
            probs = F.softmax(next_token_logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[0, indices_to_remove] = -float("Inf")

            next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
            token_id = next_token.item()

            if token_id == eos_token_id:
                break

            generated.append(token_id)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Retrograde Feedback Loop: Drive the Brain with the Voice
            token_emb = self.embed(next_token).squeeze(1) # [B, d_model]
            u_retro = self.retrograde.step(token_emb) # [B, state_dim]

            prev_x = curr_x
            if engine is not None:
                # Re-converge the brain for M micro-steps using the retrograde force
                curr_x = engine.step_many(curr_x, n_steps=self.cfg.retro_micro_steps, sim_graft=u_retro)
            else:
                # Fallback: simple Euler step if engine not provided
                curr_x = curr_x + 0.01 * u_retro

            # Road B: brain-controlled halt. Measure how much the converged
            # state changed this token. When the brain stops moving (delta <
            # threshold) for convergence_window consecutive tokens, the brain
            # has settled on its answer - stop vocalizing. This is what makes
            # generation length model-decided rather than a fixed cap.
            with torch.no_grad():
                delta = float(torch.linalg.vector_norm(curr_x - prev_x).detach().cpu().item())
            if delta < self.cfg.convergence_threshold:
                convergence_streak += 1
                if convergence_streak >= self.cfg.convergence_window:
                    break
            else:
                convergence_streak = 0

        # Handle different tokenizer interfaces (tokenizers.Tokenizer vs others)
        if hasattr(tokenizer, "decode"):
            if hasattr(tokenizer, "encode"): # tokenizers.Tokenizer
                return tokenizer.decode(generated)
            else: # transformers.PreTrainedTokenizer
                try:
                    return tokenizer.decode(generated, skip_special_tokens=True)
                except TypeError:
                    return tokenizer.decode(generated)
        return str(generated)

    def _estimate_generation_budget(self, x_final: Tensor, input_ids: Tensor) -> int:
        """
        Dynamic generation budget.

        This is a safety ceiling, not a receptive-field window. The model still
        stops early on EOS or confidence collapse; this only prevents runaway
        loops when the decoder cannot self-terminate.
        """
        hidden_energy = float(torch.linalg.norm(x_final).detach().cpu().item())
        prompt_len = int(input_ids.shape[1])
        base = 128 + int(hidden_energy * 48.0) + prompt_len * 4
        safety_cap = int(max(512, min(4096, self.cfg.max_seq_len * 8)))
        return max(64, min(base, safety_cap))

    def save_pretrained(self, path: Path | str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), p / "sigma_native_emitter.pt")
        self.cfg.save(p / "sigma_native_emitter_config.json")

    @classmethod
    def load(cls, path: Path | str, device: str = "cpu") -> "SigmaNativeEmitter":
        p = Path(path)
        if p.is_file():
            # Loading from single .pt file, look for config sidecar
            cfg_path = p.parent / "sigma_native_emitter_config.json"
            cfg = SigmaEmitterConfig.load(cfg_path) if cfg_path.is_file() else SigmaEmitterConfig.foundation()
            model = cls(cfg)
            model.load_state_dict(torch.load(p, map_location=device, weights_only=True), strict=False)
        else:
            # Loading from directory
            cfg = SigmaEmitterConfig.load(p / "sigma_native_emitter_config.json")
            model = cls(cfg)
            model.load_state_dict(torch.load(p / "sigma_native_emitter.pt", map_location=device, weights_only=True), strict=False)
        return model.to(device)

def is_sigma_checkpoint_file(path: Path | str) -> bool:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        state = ckpt.get("model") if isinstance(ckpt, dict) else ckpt
        if not isinstance(state, dict):
            return False
        if any(k.endswith("qkv.weight") or ".qkv." in k for k in state.keys()):
            return False
        return True
    except Exception:
        return False
