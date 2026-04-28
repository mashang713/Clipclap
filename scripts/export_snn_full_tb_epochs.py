#!/usr/bin/env python3
"""
Export per-epoch validation metrics from TensorBoard for one training stage folder
(subdirectory that contains events.out.tfevents.*).

Call twice (stage1 dir, stage2 dir) or use --stage1_dir and --stage2_dir to merge
into one CSV with a 'stage' column.

Metrics (from src/train.py val + add_logs_tensorboard, which_stage='val'):
  metric_val/both_seen, both_unseen, both_hm, both_zsl
  Loss/original_loss_val, Loss/proto_kd_val, Loss/feature_mse_val
  Diag/proto_topk_overlap_val, Diag/proto_top1_agree_val
  Diag/snn_clip_ratio_val, Diag/snn_zero_ratio_val
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


TAGS = [
    ("metric_val/both_seen", "val_Seen"),
    ("metric_val/both_unseen", "val_Unseen"),
    ("metric_val/both_hm", "val_HM"),
    ("metric_val/both_zsl", "val_ZSL"),
    ("Loss/original_loss_val", "Loss/original_loss"),
    ("Loss/proto_kd_val", "Loss/proto_kd"),
    ("Loss/feature_mse_val", "Loss/feature_mse"),
    ("Diag/proto_topk_overlap_val", "Diag/proto_topk_overlap"),
    ("Diag/proto_top1_agree_val", "Diag/proto_top1_agree"),
    ("Diag/snn_clip_ratio_val", "Diag/snn_clip_ratio"),
    ("Diag/snn_zero_ratio_val", "Diag/snn_zero_ratio"),
]


def _load_ea(log_dir: Path) -> Any:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    ea.Reload()
    return ea


def _series(ea: Any, tag: str) -> List[Tuple[int, float]]:
    if tag not in ea.Tags().get("scalars", []):
        return []
    return [(e.step, float(e.value)) for e in ea.Scalars(tag)]


def _epochs_from_hm(hm_series: List[Tuple[int, float]]) -> List[int]:
    """One row per validation end: use index order as epoch within this stage."""
    return list(range(len(hm_series)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--stage1_dir",
        type=Path,
        required=True,
        help="TB subfolder for stage 1 (must exist; contains events.out.tfevents.*).",
    )
    ap.add_argument(
        "--stage2_dir",
        type=Path,
        required=True,
        help="TB subfolder for stage 2 (must exist; contains events.out.tfevents.*).",
    )
    ap.add_argument("--out_csv", type=Path, default=None, help="Output CSV path.")
    ap.add_argument(
        "--scale_percent",
        action="store_true",
        help="Multiply val_Seen/HM/Unseen/ZSL by 100 (TB stores 0..1).",
    )
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    out = args.out_csv or (repo / "export_snn_full_tb_epochs.csv")
    mult = 100.0 if args.scale_percent else 1.0

    rows_out: List[Dict[str, Any]] = []
    for stage_label, log_dir in (("stage1", args.stage1_dir), ("stage2", args.stage2_dir)):
        log_dir = log_dir.resolve()
        if not log_dir.is_dir():
            raise SystemExit(
                f"Not a directory (use the real TB folder that contains events.out.tfevents.*, not placeholders): {log_dir}"
            )
        ea = _load_ea(log_dir)
        hm_series = _series(ea, "metric_val/both_hm")
        if not hm_series:
            raise SystemExit(f"No metric_val/both_hm in {log_dir}")
        hm_series = sorted(hm_series, key=lambda x: x[0])
        n = len(hm_series)
        # Align all tags by step to same length as HM (use HM steps as reference)
        steps = [s for s, _ in hm_series]

        def val_at_step(tag: str, step: int) -> float:
            ser = sorted(_series(ea, tag), key=lambda x: x[0])
            m = {s: v for s, v in ser}
            return float(m.get(step, float("nan")))

        for ep in range(n):
            step = steps[ep]
            row: Dict[str, Any] = {"stage": stage_label, "epoch": ep}
            row["tb_step"] = step
            for tb_tag, col in TAGS:
                v = val_at_step(tb_tag, step)
                if col.startswith("val_"):
                    v = v * mult
                row[col] = v
            rows_out.append(row)

    fieldnames = ["stage", "epoch", "tb_step"] + [c for _, c in TAGS]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    print(f"Wrote {len(rows_out)} rows to {out.resolve()}")


if __name__ == "__main__":
    main()
