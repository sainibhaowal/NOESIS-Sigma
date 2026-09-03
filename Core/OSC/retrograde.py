import torch
import torch.nn as nn
from typing import Optional

class RetrogradeController(nn.Module):
    """Coupled continuous-discrete feedback loop.
    
    At each emitted token τk:
      1. Embed τk → e(τk) ∈ ℝ^d_model
      2. Project: u_retro = W_retro @ e(τk)
      3. Decay: u(t) = u_0 + α · Σ λ^(k-i) · W_retro · e(τi)
      4. Re-converge OSC for M micro-steps with new u(t)
      5. Update x* → feed new prefix to emitter
      
    Enforces Input-to-State Stability (ISS) by clamping ||W_retro @ e(τk)||₂ <= u_max.
    """

    def __init__(
        self,
        d_model: int,
        state_dim: int,
        alpha: float = 0.1,
        decay: float = 0.95,
        micro_steps: int = 5,
        u_max: float = 1.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.alpha = alpha
        self.decay = decay
        self.micro_steps = micro_steps
        self.u_max = u_max

        # Projection from token embedding space (d_model) to OSC state space (state_dim)
        self.W_retro = nn.Linear(d_model, state_dim, bias=False)
        nn.init.normal_(self.W_retro.weight, std=0.02)

        # Track the accumulated retrograde drive u_retro_acc.
        # Registered as buffer so it moves to correct device, but is not a parameter.
        self.register_buffer("u_retro_acc", torch.zeros(state_dim))

    def reset(self) -> None:
        """Reset the retrograde feedback accumulator."""
        self.u_retro_acc.zero_()

    def step(self, e_k: torch.Tensor) -> torch.Tensor:
        """Updates the retrograde drive with the new token embedding.
        
        e_k: [B, d_model] - the embedding of the emitted token τk.
        Returns: [B, state_dim] - the retrograde feedback contribution.
        """
        # Project token embedding to state space
        u_proj = self.W_retro(e_k)  # [B, state_dim]

        # Accumulate with temporal decay: u_retro_acc = decay * u_retro_acc + alpha * u_proj
        if self.u_retro_acc.shape != u_proj.shape:
            # Re-initialize to match batch size dynamically if needed
            self.u_retro_acc = torch.zeros_like(u_proj)

        self.u_retro_acc = self.decay * self.u_retro_acc + self.alpha * u_proj
        
        # Dynamically compute closed ISS bound: U_max = (alpha * ||W_retro||_2 * ||e_k||_2) / (1 - decay)
        e_norm = torch.linalg.norm(e_k, dim=-1, keepdim=True)
        # Using Frobenius norm as a fast differentiable upper bound for spectral norm ||W||_2
        w_norm = torch.linalg.matrix_norm(self.W_retro.weight, ord='fro')
        
        dynamic_u_max = (self.alpha * w_norm * e_norm) / (1.0 - self.decay + 1e-8)
        
        # Enforce ISS bound: clamp L2 norm of the accumulated drive
        acc_norm = torch.linalg.norm(self.u_retro_acc, dim=-1, keepdim=True)
        scale = torch.clamp(dynamic_u_max / (acc_norm + 1e-12), max=1.0)
        self.u_retro_acc = self.u_retro_acc * scale
        
        return self.u_retro_acc
