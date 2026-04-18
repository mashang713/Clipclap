"""
Phase B (skeleton): optional audio frontend + UCF contrastive bridge.
Phase A: ``main.py`` + offline features + SNN head (``config/clipclap_snn_baseline.yaml``).

Phase B train entry: ``python scripts/phase_b_train_audio.py -c config/phase_b_ucf_audio.yaml ...``
"""

from .audio_frontend import PhaseBAudioEncoderSNN, PhaseBAudioEncoderSNN_V2
from .phase_b_audio_model import ClipClapPhaseB_AudioWrapper
from .ucf_phase_b_audio_dataset import ContrastivePhaseBAudio, PhaseBAudioSource, UCFPhaseBAudioDataset

__all__ = [
    "PhaseBAudioEncoderSNN",
    "PhaseBAudioEncoderSNN_V2",
    "ClipClapPhaseB_AudioWrapper",
    "ContrastivePhaseBAudio",
    "UCFPhaseBAudioDataset",
    "PhaseBAudioSource",
]
