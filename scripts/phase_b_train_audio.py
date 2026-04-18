#!/usr/bin/env python
"""
Phase B entry (UCF, audio-only): mel-flat -> PhaseBAudioEncoderSNN (v1 or v2) -> existing SNN head.

Does not change main.py or Phase A defaults. Example:

  python scripts/phase_b_train_audio.py -c config/phase_b_ucf_audio.yaml \\
    --root_dir /path/to/UCF --log_dir ./runs --device cuda:0 \\
    --snn_init_ann_path /path/to/ANN_ckpt.pt

Use --phase_b_audio_source dummy for a quick test without meaningful audio signal.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase_b.run_phase_b_ucf_audio import run_phase_b_ucf_audio  # noqa: E402

if __name__ == "__main__":
    run_phase_b_ucf_audio()
