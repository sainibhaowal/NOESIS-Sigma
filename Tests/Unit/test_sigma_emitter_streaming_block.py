from __future__ import annotations

import torch


def test_ultra_frontier_block_handles_variable_sequence_lengths():
    from Core.Native_Decoder.sigma_native_emitter import UltraFrontierBlock

    block = UltraFrontierBlock(d_model=16, kernel_size=3, layer_idx=0)
    x_short = torch.randn(2, 1, 16)
    x_long = torch.randn(2, 11, 16)

    y_short = block(x_short)
    y_long = block(x_long)

    assert y_short.shape == x_short.shape
    assert y_long.shape == x_long.shape
