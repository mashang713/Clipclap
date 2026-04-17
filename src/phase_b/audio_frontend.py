"""
Minimal SNN frontend: fixed-size mel (or flat features) -> 1024-D for existing audio O_enc.
Uses the same SNN_EmbeddingNet as the rest of ClipClap (snntorch Leaky LIF).
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from src.clipclap_model import SNN_EmbeddingNet

_logger = logging.getLogger(__name__)


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
        collect_frontend_diag: bool = False,
        diag_log_interval: int = 50,
    ):
        super().__init__()
        self.mel_flat_dim = mel_flat_dim
        self.out_dim = out_dim
        self.collect_frontend_diag = bool(collect_frontend_diag)
        self.diag_log_interval = max(1, int(diag_log_interval))
        self._diag_accum: dict[str, float] = {}
        self._diag_count = 0
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
        if x.dim() == 3:
            x = x.mean(dim=1)
        spike_diag = {} if self.collect_frontend_diag and self.training else None
        y = self.net(x, spike_diag)
        if spike_diag:
            self._merge_frontend_diag(spike_diag)
        return y

    def _merge_frontend_diag(self, spike_diag: dict) -> None:
        """Running average of spike stats; periodic INFO log."""
        self._diag_count += 1
        for k, v in spike_diag.items():
            self._diag_accum[k] = self._diag_accum.get(k, 0.0) + float(v)
        if self._diag_count < self.diag_log_interval:
            return
        n = float(self._diag_count)
        parts = [f"PhaseB-frontend_diag (last {self._diag_count} forwards):"]
        for k in sorted(self._diag_accum.keys()):
            parts.append(f" {k}={self._diag_accum[k] / n:.6g}")
        _logger.info(" ".join(parts))
        self._diag_count = 0
        self._diag_accum.clear()

    def flush_frontend_diag(self) -> None:
        """Call at epoch end to log any remainder."""
        if not self.collect_frontend_diag or self._diag_count == 0:
            return
        n = float(self._diag_count)
        parts = [f"PhaseB-frontend_diag (flush, n={int(n)}):"]
        for k in sorted(self._diag_accum.keys()):
            parts.append(f" {k}={self._diag_accum[k] / n:.6g}")
        _logger.info(" ".join(parts))
        self._diag_count = 0
        self._diag_accum.clear()
