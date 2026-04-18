"""
Phase B audio SNN frontends: mel-flat (or shaped audio) -> 1024-D for existing audio O_enc.

v1: single ``SNN_EmbeddingNet`` (two LIF layers when hidden_size > 0), time mean-pooled then encoded.

v2: **per-frame** stem SNN on (B*T, D), temporal **mean || max** aggregation, ``LayerNorm`` on the
aggregate, then a **tail** ``SNN_EmbeddingNet`` -> 1024. Improves temporal expressivity vs collapsing
time with a single mean before any nonlinearity.

Firing diagnostics (when ``collect_frontend_diag`` and ``training``): accumulate per-forward
metrics from ``SNN_EmbeddingNet``; **one log line per epoch** via ``flush_frontend_diag`` (called
from ``train.py`` after each epoch).
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

from src.clipclap_model import SNN_EmbeddingNet

_logger = logging.getLogger(__name__)


def _merge_spike_dict(dst: Dict[str, float], src: Optional[Dict], prefix: str) -> None:
    if not src:
        return
    for k, v in src.items():
        dst[f"{prefix}{k}"] = float(v)


class _EpochFiringDiag:
    """Train-only, epoch-summarized firing stats (mean over training forwards in the epoch)."""

    __slots__ = ("enabled", "_sum", "_n")

    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self._sum: Dict[str, float] = {}
        self._n = 0

    def accumulate(self, spike_diag: Optional[Dict[str, float]]) -> None:
        if not self.enabled or not spike_diag:
            return
        self._n += 1
        for k, v in spike_diag.items():
            self._sum[k] = self._sum.get(k, 0.0) + float(v)

    def flush(self, tag: str) -> None:
        if not self.enabled or self._n == 0:
            self._sum.clear()
            self._n = 0
            return
        n = float(self._n)
        parts = [f"PhaseB-frontend_diag epoch summary ({tag}, n_forwards={int(self._n)}):"]
        for k in sorted(self._sum.keys()):
            parts.append(f" {k}={self._sum[k] / n:.6g}")
        _logger.info(" ".join(parts))
        self._sum.clear()
        self._n = 0


class PhaseBAudioEncoderSNN(nn.Module):
    """
    Maps a flattened mel / spectrogram patch (``mel_flat_dim``) to ``out_dim`` (default 1024),
    matching ``ClipClap_model`` audio branch ``O_enc`` input size.

    v1 path: optional time mean-pool, then one ``SNN_EmbeddingNet`` (mel_flat_dim -> 1024).
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
        # Kept for API compatibility; firing stats are logged once per epoch (flush), not by interval.
        self.diag_log_interval = max(1, int(diag_log_interval))
        self._diag = _EpochFiringDiag(collect_frontend_diag)
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

    @property
    def collect_frontend_diag(self) -> bool:
        return self._diag.enabled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (B, mel_flat_dim) or (B, T, mel_flat_dim)
            If 3-D, time is mean-pooled before the encoder (v1 behaviour).
        """
        if x.dim() == 3:
            x = x.mean(dim=1)
        spike_diag: Optional[dict] = {} if self._diag.enabled and self.training else None
        y = self.net(x, spike_diag)
        if spike_diag is not None:
            self._diag.accumulate(spike_diag)
        return y

    def flush_frontend_diag(self) -> None:
        self._diag.flush(self.__class__.__name__)


class PhaseBAudioEncoderSNN_V2(nn.Module):
    """
    Stronger audio frontend: stem SNN on each time slice, temporal mean||max, LayerNorm, tail SNN -> 1024.

    - **Stem**: ``SNN_EmbeddingNet(mel_flat_dim -> stem_out_dim)`` on (B*T, mel_flat_dim).
    - **Temporal agg**: concat(mean_t, max_t) over T -> (B, 2 * stem_out_dim).
    - **Light projection**: ``LayerNorm`` on the aggregate (stabilizes scale for the tail SNN).
    - **Tail**: ``SNN_EmbeddingNet(2 * stem_out_dim -> out_dim)``.
    """

    #: ``ClipClapPhaseB_AudioWrapper`` keeps ``(B, T, F)`` instead of mean-pooling time before this module.
    phase_b_use_temporal_audio = True

    def __init__(
        self,
        mel_flat_dim: int = 4096,
        out_dim: int = 1024,
        stem_out_dim: int = 512,
        dropout: float = 0.1,
        stem_hidden: int = 2048,
        tail_hidden: int = 2048,
        num_steps: int = 8,
        beta: float = 0.9,
        threshold: float = 1.0,
        collect_frontend_diag: bool = False,
        diag_log_interval: int = 50,
    ):
        super().__init__()
        self.mel_flat_dim = int(mel_flat_dim)
        self.out_dim = int(out_dim)
        self.stem_out_dim = int(stem_out_dim)
        self.diag_log_interval = max(1, int(diag_log_interval))
        self._diag = _EpochFiringDiag(collect_frontend_diag)

        self.stem = SNN_EmbeddingNet(
            input_size=self.mel_flat_dim,
            output_size=self.stem_out_dim,
            dropout=dropout,
            use_bn=False,
            hidden_size=stem_hidden,
            num_steps=num_steps,
            beta=beta,
            threshold=threshold,
        )
        agg_dim = 2 * self.stem_out_dim
        self.agg_norm = nn.LayerNorm(agg_dim)
        self.tail = SNN_EmbeddingNet(
            input_size=agg_dim,
            output_size=self.out_dim,
            dropout=dropout,
            use_bn=False,
            hidden_size=tail_hidden,
            num_steps=num_steps,
            beta=beta,
            threshold=threshold,
        )

    @property
    def collect_frontend_diag(self) -> bool:
        return self._diag.enabled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3:
            raise ValueError(f"Expected audio tensor of dim 2 or 3, got shape {tuple(x.shape)}")

        b, t, d = x.shape
        if d != self.mel_flat_dim:
            raise ValueError(f"Last dim {d} != mel_flat_dim {self.mel_flat_dim}")

        diag_on = self._diag.enabled and self.training
        x_flat = x.reshape(b * t, d)

        d_stem: Optional[dict] = {} if diag_on else None
        stem_y = self.stem(x_flat, d_stem)
        seq = stem_y.view(b, t, self.stem_out_dim)
        agg = torch.cat([seq.mean(dim=1), seq.max(dim=1).values], dim=-1)
        agg = self.agg_norm(agg)

        d_tail: Optional[dict] = {} if diag_on else None
        out = self.tail(agg, d_tail)

        if diag_on and d_stem is not None and d_tail is not None:
            merged: Dict[str, float] = {}
            _merge_spike_dict(merged, d_stem, "stem_")
            _merge_spike_dict(merged, d_tail, "tail_")
            self._diag.accumulate(merged)
        return out

    def flush_frontend_diag(self) -> None:
        self._diag.flush(self.__class__.__name__)
