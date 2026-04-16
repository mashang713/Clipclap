"""
Minimal SNN frontend: fixed-size mel (or flat features) -> 1024-D for existing audio O_enc.
Uses the same SNN_EmbeddingNet as the rest of ClipClap (snntorch Leaky LIF).
"""

from __future__ import annotations

import torch.nn as nn

from src.clipclap_model import SNN_EmbeddingNet


class PhaseBAudioEncoderSNN(nn.Module):
    """
    Maps a flattened mel / spectrogram patch (``mel_flat_dim``) to ``out_dim`` (default 1024),
    matching ``ClipClap_model`` audio branch ``O_enc`` input size.

    This is a skeleton: one or two LIF stacks, not a full HTSAT/AST replacement.
    """

    def __init__(
        self,
        mel_flat_dim: int = 4096,
        out_dim: int = 1024,
        dropout: float = 0.1,
        hidden_size: int = 2048,
        num_steps: int = 8,
        beta: float = 0.9,
        threshold: float = 1.0,
    ):
        super().__init__()
        self.mel_flat_dim = mel_flat_dim
        self.out_dim = out_dim
        self.net = SNN_EmbeddingNet(
            input_size=mel_flat_dim,
            output_size=out_dim,
            dropout=dropout,
            use_bn=False,
            hidden_size=hidden_size,
            num_steps=num_steps,
            beta=beta,
            threshold=threshold,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (B, mel_flat_dim) or (B, T, mel_flat_dim)
            If 3-D, time is mean-pooled before the encoder (skeleton behaviour).
        """
        import torch

        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.net(x)
