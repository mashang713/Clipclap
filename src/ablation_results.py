"""Append GZSL ablation results to a CSV; reads last *validation* TensorBoard scalars from stage-2 run."""
from __future__ import annotations

import csv
import pathlib
from typing import Any, Dict, List, Optional

# TensorBoard log tags (suffix = validation step in src/train.py val_step, which_stage="val")
_TB_VAL_TAGS = {
    "proto_kd": "Loss/proto_kd_val",
    "feature_mse": "Loss/feature_mse_val",
    "proto_topk_overlap": "Diag/proto_topk_overlap_val",
    "proto_top1_agree": "Diag/proto_top1_agree_val",
    "snn_clip_ratio": "Diag/snn_clip_ratio_val",
    "snn_zero_ratio": "Diag/snn_zero_ratio_val",
    "theta_o_negative_ratio": "Diag/theta_o_negative_ratio_val",
}

_CSV_FIELDNAMES: List[str] = [
    "run_name",
    "best_epoch",
    "stage2_epochs",
    "stage1_epochs",
    "n_batches",
    "seed",
    "cfg_path",
    "use_snn_conversion",
    "snn_timesteps",
    "snn_conv_threshold_percentile",
    "lambda_proto",
    "lambda_feat",
    "proto_temperature",
    "Seen",
    "Unseen",
    "HM",
    "ZSL",
    "proto_topk_overlap",
    "proto_top1_agree",
    "snn_clip_ratio",
    "snn_zero_ratio",
    "theta_o_negative_ratio",
    "proto_kd",
    "feature_mse",
    "final_checkpoint_path",
]


def _last_scalar_in_tb(log_dir: pathlib.Path, tag: str) -> Optional[float]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return None
    if not log_dir or not log_dir.is_dir():
        return None
    try:
        ea = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
        ea.Reload()
    except Exception:
        return None
    if tag not in ea.Tags().get("scalars", []):
        return None
    try:
        events = ea.Scalars(tag)
    except Exception:
        return None
    if not events:
        return None
    return float(events[-1].value)


def read_val_diagnostics_from_tb(log_dir: pathlib.Path) -> Dict[str, Optional[float]]:
    return {k: _last_scalar_in_tb(log_dir, tag) for k, tag in _TB_VAL_TAGS.items()}


def append_results_ablation_csv(
    csv_path: pathlib.Path,
    *,
    run_name: str,
    config: Any,
    results_both: Dict[str, Any],
    final_checkpoint_path: pathlib.Path,
    log_dir_for_tb: pathlib.Path,
) -> None:
    """
    results_both: from test_evaluation['both'] (seen, unseen, hm, zsl in [0,1]).
    """
    diags = read_val_diagnostics_from_tb(log_dir_for_tb)
    seen = float(results_both.get("seen", 0.0)) * 100.0
    unseen = float(results_both.get("unseen", 0.0)) * 100.0
    hm = float(results_both.get("hm", 0.0)) * 100.0
    zsl = float(results_both.get("zsl", 0.0)) * 100.0

    row: Dict[str, Any] = {
        "run_name": run_name,
        "best_epoch": _fmt_int_or_str(getattr(config, "ablation_best_epoch", None)),
        "stage2_epochs": _fmt_int_or_str(getattr(config, "ablation_stage2_epochs", None)),
        "stage1_epochs": _fmt_int_or_str(getattr(config, "ablation_stage1_epochs", None)),
        "n_batches": _fmt_int_or_str(getattr(config, "n_batches", None)),
        "seed": _fmt_int_or_str(getattr(config, "seed", None)),
        "cfg_path": _fmt_cfg(getattr(config, "cfg", None)),
        "use_snn_conversion": getattr(config, "use_snn_conversion", False),
        "snn_timesteps": getattr(config, "snn_timesteps", 4),
        "snn_conv_threshold_percentile": getattr(config, "snn_conv_threshold_percentile", 0.99),
        "lambda_proto": getattr(config, "lambda_proto", 0.5),
        "lambda_feat": getattr(config, "lambda_feat", 0.1),
        "proto_temperature": getattr(config, "proto_temperature", 2.0),
        "Seen": f"{seen:.4f}",
        "Unseen": f"{unseen:.4f}",
        "HM": f"{hm:.4f}",
        "ZSL": f"{zsl:.4f}",
        "proto_topk_overlap": _fmt(diags.get("proto_topk_overlap")),
        "proto_top1_agree": _fmt(diags.get("proto_top1_agree")),
        "snn_clip_ratio": _fmt(diags.get("snn_clip_ratio")),
        "snn_zero_ratio": _fmt(diags.get("snn_zero_ratio")),
        "theta_o_negative_ratio": _fmt(diags.get("theta_o_negative_ratio")),
        "proto_kd": _fmt(diags.get("proto_kd")),
        "feature_mse": _fmt(diags.get("feature_mse")),
        "final_checkpoint_path": str(pathlib.Path(final_checkpoint_path).resolve()),
    }
    _write_row(csv_path, row)


def _fmt(x: Optional[float]) -> str:
    if x is None:
        return ""
    return f"{x:.8g}"


def _fmt_int_or_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def _fmt_cfg(x: Any) -> str:
    if x is None or x == "":
        return ""
    try:
        return str(pathlib.Path(x).resolve())
    except Exception:
        return str(x)


def _write_row(csv_path: pathlib.Path, row: Dict[str, Any]) -> None:
    csv_path = pathlib.Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.is_file() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)
