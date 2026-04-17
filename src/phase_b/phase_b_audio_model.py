"""
Wrap a trained ``ClipClap_model`` (``modality='audio'``) with ``PhaseBAudioEncoderSNN`` so that
``optimize_params`` / ``get_embeddings`` receive raw batched audio tensors that are first
mapped to 1024-D to match existing ``O_enc`` / ``W_enc`` inputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _pool_audio_batch(a: torch.Tensor) -> torch.Tensor:
    """(B, T, F) -> (B, F); (B, F) unchanged."""
    if a.dim() == 3:
        return a.mean(dim=1)
    if a.dim() == 2:
        return a
    raise ValueError(f"Expected audio tensor of dim 2 or 3, got shape {tuple(a.shape)}")


class ClipClapPhaseB_AudioWrapper(nn.Module):
    """
    Frontend: mel-flat or pooled sequence -> 1024-D.
    Inner: existing ``ClipClap_model`` (audio branch, SNN or ANN).
    """

    def __init__(self, inner: nn.Module, frontend: nn.Module, audio_input_scale: float = 1.0):
        super().__init__()
        self.inner = inner
        self.frontend = frontend
        self.register_buffer(
            "_audio_input_scale",
            torch.tensor(float(audio_input_scale), dtype=torch.float32),
            persistent=True,
        )

    @property
    def optimizer_gen(self):
        return self.inner.optimizer_gen

    def _encode_audio(self, a: torch.Tensor) -> torch.Tensor:
        a = _pool_audio_batch(a)
        s = self._audio_input_scale.to(device=a.device, dtype=a.dtype)
        a = a * s
        return self.frontend(a)

    def flush_frontend_diag(self) -> None:
        fe = getattr(self, "frontend", None)
        if fe is not None and hasattr(fe, "flush_frontend_diag"):
            fe.flush_frontend_diag()

    def forward(self, a, v, w, masks, timesteps):
        a = self._encode_audio(a)
        return self.inner.forward(a, v, w, masks, timesteps)

    def optimize_params(self, audio, video, cls_numeric, cls_embedding, masks, timesteps, embedding_crossentropy, optimize=False):
        audio = self._encode_audio(audio)
        return self.inner.optimize_params(
            audio, video, cls_numeric, cls_embedding, masks, timesteps, embedding_crossentropy, optimize=optimize
        )

    def get_embeddings(self, a, v, w, masks, timesteps):
        a = self._encode_audio(a)
        return self.inner.get_embeddings(a, v, w, masks, timesteps)

    def optimize_scheduler(self, value):
        return self.inner.optimize_scheduler(value)

    # train.py / check_best_* expect a scheduler on the student
    @property
    def scheduler_learning_rate(self):
        return getattr(self.inner, "scheduler_learning_rate", None)
