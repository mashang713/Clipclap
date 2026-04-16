"""
Phase B (audio): optional dataset path that feeds a fixed-size mel-flat (or dummy) tensor
through the SNN frontend, without changing UCFDataset or default main.py.

Text / class embeddings still come from the same UCF processed pickle as Phase A.
Video features are left as in the underlying ContrastiveDataset (unchanged Phase A path).
Only the audio branch is replaced before the collator by reshaping offline features or noise.
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

import numpy as np

from src.dataset import ContrastiveDataset

if TYPE_CHECKING:
    pass


class PhaseBAudioSource(str, enum.Enum):
    """How to build the pseudo-mel flat vector fed to PhaseBAudioEncoderSNN."""

    DUMMY_RANDOM = "dummy"  # reproducible Gaussian (fixed seed per idx optional)
    OFFLINE_AS_MEL = "offline_as_mel"  # crop/pad flattened offline CLAP audio feats → mel_flat_dim


def _offline_row_to_mel_flat(x_a: np.ndarray, mel_flat_dim: int, rng: np.random.Generator) -> np.ndarray:
    """Flatten offline audio features (T, F) or (F,) to ``mel_flat_dim``."""
    flat = np.asarray(x_a, dtype=np.float64).reshape(-1)
    if flat.size >= mel_flat_dim:
        out = flat[:mel_flat_dim].astype(np.float32)
    else:
        out = np.zeros(mel_flat_dim, dtype=np.float32)
        out[: flat.size] = flat.astype(np.float32)
    return out


def _dummy_mel_flat(mel_flat_dim: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(mel_flat_dim).astype(np.float32)


class ContrastivePhaseBAudio(ContrastiveDataset):
    """
    Same contrastive sampling as ``ContrastiveDataset``, but replaces each audio sequence
    with a 2-D array ``(1, mel_flat_dim)`` so the default collator pads along time like Phase A.

    Parameters
    ----------
    mel_flat_dim : int
        Must match ``PhaseBAudioEncoderSNN.mel_flat_dim`` (default 4096).
    source : PhaseBAudioSource
        ``DUMMY_RANDOM`` or ``OFFLINE_AS_MEL`` (reuses offline audio rows as a long vector).
    rng_seed : int
        Seed for dummy noise (offline path ignores it except for reproducibility hooks).
    """

    def __init__(
        self,
        zsl_dataset,
        source: PhaseBAudioSource = PhaseBAudioSource.OFFLINE_AS_MEL,
        mel_flat_dim: int = 4096,
        rng_seed: int = 42,
    ):
        super().__init__(zsl_dataset)
        self.source = PhaseBAudioSource(source)
        self.mel_flat_dim = int(mel_flat_dim)
        self._rng = np.random.default_rng(rng_seed)

    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        for side in ("positive", "negative"):
            if self.source == PhaseBAudioSource.DUMMY_RANDOM:
                flat = _dummy_mel_flat(self.mel_flat_dim, self._rng)
            else:
                flat = _offline_row_to_mel_flat(data[side]["audio"], self.mel_flat_dim, self._rng)
            data[side]["audio"] = np.expand_dims(flat, axis=0).astype(np.float32)
        return data, target


# Alias for docs / explicit naming
UCFPhaseBAudioDataset = ContrastivePhaseBAudio
