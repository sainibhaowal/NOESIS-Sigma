import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1dEncoderBlock(nn.Module):
    """1D depthwise-separable conv block for sequence encoding without self-attention."""
    def __init__(self, d_model: int, kernel_size: int, dropout: float = 0.1, dilation: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.norm = nn.LayerNorm(d_model)
        self.dw = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            groups=d_model,
            bias=False,
            dilation=dilation,
            padding="same"
        )
        self.pw = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        hc = h.transpose(1, 2)
        y = self.dw(hc).transpose(1, 2)
        y = self.pw(y)
        y = F.silu(y)
        return residual + self.drop(y)

class SigmaNativeEncoder(nn.Module):
    """
    Attention-Free Sequence Encoder.
    Replaces external Transformer APIs (e.g. Nomic AI) with a purely native 
    1D convolutional architecture to map input text to the continuous state space.
    """
    def __init__(
        self, 
        vocab_size: int = 16384, 
        d_model: int = 512, 
        state_dim: int = 1024, 
        n_layers: int = 6, 
        kernel_size: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.drop_emb = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            CausalConv1dEncoderBlock(d_model, kernel_size, dropout, dilation=2**i)
            for i in range(n_layers)
        ])
        
        self.norm_final = nn.LayerNorm(d_model)
        self.state_proj = nn.Linear(d_model, state_dim, bias=False)
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [Batch, SequenceLength]
        Returns:
            continuous_vector: [Batch, state_dim]
        """
        x = self.drop_emb(self.embed(input_ids.clamp(min=0)))
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm_final(x)
        
        # Global Max Pooling to extract strongest semantic features across sequence length
        x_pooled, _ = torch.max(x, dim=1)  # [Batch, d_model]
        
        # Project to ICNN semantic space
        continuous_vector = self.state_proj(x_pooled)  # [Batch, state_dim]
        return continuous_vector
