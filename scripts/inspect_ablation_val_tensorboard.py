#!/usr/bin/env python3
"""
Summarize per-epoch validation metrics from a *single* training stage log directory
(the subfolder under log_dir that contains TensorBoard events for that stage, e.g. stage-1).

Usage (on server, after identifying the stage-1 run folder for snn_proto_only):
  python scripts/inspect_ablation_val_tensorboard.py \\
    --log_dir /home/ubuntu/ClipClap-GZSL/logs/ablation_snn_proto_only_T8_p099/snn_proto_only_T8_p099_<timestamp>_vm-mary

Requires: tensorboard package (EventAccumulator).

best_epoch (when best_model_criterion=score):
  In src/utils.py::check_best_score, the checkpoint with the highest val HM is kept.
  Val HM is logged as scalar tag: metric_val/both_hm
  So best_epoch = argmax over epochs of metric_val/both_hm on the validation split.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_scalars(log_dir: Path) -> "Any":
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    ea.Reload()
    return ea


def _series(ea: Any, tag: str) -> List[Tuple[int, float]]:
    if tag not in ea.Tags().get("scalars", []):
        return []
    return [(e.step, float(e.value)) for e in ea.Scalars(tag)]


def _align_by_step(tags: Dict[str, List[Tuple[int, float]]]) -> List[Dict[str, float]]:
    """One row per unique step, merging tags (validation logs one step per epoch end)."""
    all_steps = sorted({s for series in tags.values() for s, _ in series})
    rows = []
    for st in all_steps:
        row: Dict[str, float] = {"step": float(st)}
        for name, series in tags.items():
            m = {s: v for s, v in series}
            if st in m:
                row[name] = m[st]
        rows.append(row)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--log_dir",
        type=Path,
        required=True,
        help="Path to one experiment directory containing events.out.tfevents.* (e.g. stage-1 subfolder).",
    )
    p.add_argument(
        "--epoch_from",
        choices=("step", "index"),
        default="index",
        help="How to label epochs: by order of val snapshots (index), or derive from step/(len val loader) (step).",
    )
    p.add_argument(
        "--scale_percent",
        action="store_true",
        help="Multiply both_* metrics by 100 for display (TB stores 0..1).",
    )
    args = p.parse_args()

    log_dir = args.log_dir.resolve()
    if not log_dir.is_dir():
        print(f"ERROR: not a directory: {log_dir}", file=sys.stderr)
        sys.exit(1)

    ea = _load_scalars(log_dir)
    want = {
        "metric_val/both_seen": "both_seen",
        "metric_val/both_unseen": "both_unseen",
        "metric_val/both_hm": "both_hm",
        "metric_val/both_zsl": "both_zsl",
        "Loss/proto_kd_val": "Loss/proto_kd",
        "Loss/original_loss_val": "Loss/original_loss",
        "Diag/proto_topk_overlap_val": "proto_topk_overlap",
        "Diag/proto_top1_agree_val": "proto_top1_agree",
        "Diag/snn_clip_ratio_val": "snn_clip_ratio",
        "Diag/snn_zero_ratio_val": "snn_zero_ratio",
    }
    raw: Dict[str, List[Tuple[int, float]]] = {}
    missing = []
    for tb_tag, _short in want.items():
        s = _series(ea, tb_tag)
        if not s:
            missing.append(tb_tag)
        raw[tb_tag] = s

    if not raw.get("metric_val/both_hm"):
        print("ERROR: no metric_val/both_hm in this log_dir. Is this a training (val) run folder with TB events?", file=sys.stderr)
        sys.exit(1)

    # Use intersection of steps that appear in both_hm (primary)
    hm_steps = {s for s, _ in raw["metric_val/both_hm"]}
    rows_out: List[Dict[str, Any]] = []
    for i, st in enumerate(sorted(hm_steps)):
        row: Dict[str, Any] = {"epoch": i if args.epoch_from == "index" else None, "step": st}
        for tb_tag, short in want.items():
            m = {s: v for s, v in raw[tb_tag]}
            row[short] = m.get(st, float("nan"))
        rows_out.append(row)

    # Optional: set epoch from step if val len known
    if args.epoch_from == "step" and len(rows_out) >= 2:
        steps = sorted(hm_steps)
        diffs = [steps[i + 1] - steps[i] for i in range(len(steps) - 1)]
        n_val = int(round(sum(diffs) / max(len(diffs), 1)))  # rough
        if n_val > 0 and all(abs(d - n_val) < 1e-6 for d in diffs if len(set(diffs)) == 1):
            for j, st in enumerate(sorted(hm_steps)):
                rows_out[j]["epoch"] = int(round(st / n_val)) - 1

    print(f"# log_dir: {log_dir}")
    if missing:
        print(f"# missing tags (empty column): {', '.join(missing)}", file=sys.stderr)
    print(
        "\t".join(
            [
                "epoch",
                "both_seen",
                "both_unseen",
                "both_hm",
                "both_zsl",
                "Loss/proto_kd",
                "Loss/original_loss",
                "proto_topk_overlap",
                "proto_top1_agree",
                "snn_clip_ratio",
                "snn_zero_ratio",
            ]
        )
    )
    best_i = 0
    best_hm = -1.0
    s = 100.0 if args.scale_percent else 1.0
    for j, r in enumerate(rows_out):
        hm = r.get("both_hm", float("nan"))
        if not math.isnan(hm) and hm > best_hm:
            best_hm = hm
            best_i = j
    for j, r in enumerate(rows_out):
        ep = r.get("epoch", j)
        if ep is None:
            ep = j
        mark = "  <-- best HM (this would be best_epoch if score criterion)" if j == best_i else ""
        print(
            f"{ep}\t{r.get('both_seen', float('nan')) * s:.6f}\t{r.get('both_unseen', float('nan')) * s:.6f}\t"
            f"{r.get('both_hm', float('nan')) * s:.6f}\t{r.get('both_zsl', float('nan')) * s:.6f}\t"
            f"{r.get('Loss/proto_kd', float('nan')):.8g}\t{r.get('Loss/original_loss', float('nan')):.8g}\t"
            f"{r.get('proto_topk_overlap', float('nan')):.8g}\t{r.get('proto_top1_agree', float('nan')):.8g}\t"
            f"{r.get('snn_clip_ratio', float('nan')):.8g}\t{r.get('snn_zero_ratio', float('nan')):.8g}{mark}"
        )
    print(
        f"\n# best_epoch (0-based) under best_model_criterion=score: {best_i}  (val HM = {best_hm * s:.6f})",
        file=sys.stderr,
    )
    print(
        "# Criterion: src/utils.py::check_best_score — maximize hm_score from val MeanClassAccuracy (metric_val/both_hm).",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
