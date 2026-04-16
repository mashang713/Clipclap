#!/usr/bin/env python
"""
GZSL val+test for a Phase B audio run (weights must include ``ClipClapPhaseB_AudioWrapper`` + frontend).

  python scripts/phase_b_eval_ucf.py --run_dir /path/to/runs/ExpTimestamp_host \\
    --root_dir /path/to/UCF --device cuda:0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase_b.eval_phase_b_ucf import run_phase_b_ucf_eval  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path, required=True, help="Experiment folder (contains args.pkl, train.log)")
    ap.add_argument("--root_dir", type=Path, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--ckpt", type=Path, default=None)
    args = ap.parse_args()
    run_phase_b_ucf_eval(
        args.run_dir,
        root_dir=args.root_dir,
        device=args.device,
        ckpt_path=args.ckpt,
    )


if __name__ == "__main__":
    main()
