#!/usr/bin/env python3
"""
Formal validation-HM-selected evaluation for seed44 proto tuning runs (no training).

For each run_name, discovers two timestamped dirs under logs/<run_name>_*/ (stage-1 then stage-2
by lexicographic order of folder name), then calls the same logic as select_stage2_ckpt_by_val.py.

Writes reports/seed44_proto_tuning_val_selected_summary.csv with deltas vs fixed baselines.

Example:
  python scripts/seed44_proto_tuning_val_selected_summary.py \\
    --log_root ~/ClipClap-GZSL/logs \\
    --root_dir /path/to/UCF \\
    --device cuda:0
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_spec = importlib.util.spec_from_file_location(
    "_eval_stage2_ckpts",
    _REPO / "scripts" / "evaluate_stage2_checkpoints.py",
)
assert _spec and _spec.loader
_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval)  # type: ignore[attr-defined]


RUNS: List[Dict[str, Any]] = [
    {
        "run_name": "snn_proto_msetop10_l005_warm2_seed44",
        "lambda_proto": 0.05,
        "proto_topk": 10,
        "proto_warmup_epochs": 2,
    },
    {
        "run_name": "snn_proto_msetop5_l01_warm2_seed44",
        "lambda_proto": 0.1,
        "proto_topk": 5,
        "proto_warmup_epochs": 2,
    },
    {
        "run_name": "snn_proto_msetop10_l01_warm3_seed44",
        "lambda_proto": 0.1,
        "proto_topk": 10,
        "proto_warmup_epochs": 3,
    },
]


def _find_stage_dirs(log_root: Path, run_name: str) -> Tuple[Path, Path]:
    prefix = run_name + "_"
    dirs = sorted([p for p in log_root.iterdir() if p.is_dir() and p.name.startswith(prefix)])
    if len(dirs) < 2:
        raise SystemExit(
            f"Expected 2 log dirs under {log_root} with prefix {prefix!r}, "
            f"found {len(dirs)}: {[p.name for p in dirs]}"
        )
    dirs_sorted = sorted(dirs, key=lambda p: p.name)
    return dirs_sorted[0], dirs_sorted[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log_root", type=Path, default=_REPO / "logs", help="Directory containing run_* folders.")
    ap.add_argument("--root_dir", type=str, required=True, help="Dataset root (UCF).")
    ap.add_argument("--dataset_name", type=str, default="UCF")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--selection_metric", type=str, default="HM", choices=("HM", "Seen", "Unseen", "ZSL"))
    ap.add_argument("--best_model_criterion", type=str, default="score")
    ap.add_argument(
        "--output_csv",
        type=Path,
        default=_REPO / "reports" / "seed44_proto_tuning_val_selected_summary.csv",
    )
    ap.add_argument("--d_seed44_hm", type=float, default=52.69, help="Baseline D test HM (percent scale).")
    ap.add_argument("--p4_seed44_hm", type=float, default=51.85, help="Baseline P4 warm2 test HM (percent scale).")
    args = ap.parse_args()

    log_root = args.log_root.resolve()
    if not log_root.is_dir():
        raise SystemExit(f"Not a directory: {log_root}")

    out_rows: List[Dict[str, str]] = []
    fieldnames = [
        "run_name",
        "lambda_proto",
        "proto_topk",
        "proto_warmup_epochs",
        "best_ckpt_by_val",
        "val_HM",
        "test_Seen",
        "test_Unseen",
        "test_HM",
        "test_ZSL",
        "test_HM_minus_D_seed44",
        "test_HM_minus_P4_seed44",
    ]

    for spec in RUNS:
        run_name = spec["run_name"]
        stage_a, stage_b = _find_stage_dirs(log_root, run_name)
        cfg_path = stage_b / "args.pkl"
        cfg = pickle.load(cfg_path.open("rb"))
        cfg.root_dir = Path(args.root_dir)
        cfg.dataset_name = args.dataset_name
        if args.device is not None:
            cfg.device = args.device

        pack = _eval.evaluate_pick_best_and_test(
            stage_a,
            stage_b,
            cfg,
            selection_metric=args.selection_metric,
            best_model_criterion=args.best_model_criterion,
            device=args.device,
        )
        s = pack["summary"]
        test_hm = float(s["test_HM"])
        out_rows.append(
            {
                "run_name": run_name,
                "lambda_proto": str(spec["lambda_proto"]),
                "proto_topk": str(spec["proto_topk"]),
                "proto_warmup_epochs": str(spec["proto_warmup_epochs"]),
                "best_ckpt_by_val": str(s["best_epoch"]),
                "val_HM": s["best_val_HM"],
                "test_Seen": s["test_Seen"],
                "test_Unseen": s["test_Unseen"],
                "test_HM": s["test_HM"],
                "test_ZSL": s["test_ZSL"],
                "test_HM_minus_D_seed44": f"{test_hm - args.d_seed44_hm:.4f}",
                "test_HM_minus_P4_seed44": f"{test_hm - args.p4_seed44_hm:.4f}",
            }
        )
        print(
            f"{run_name}: ckpt={s['best_epoch']} val_HM={s['best_val_HM']} test_HM={s['test_HM']} "
            f"(ΔD={test_hm - args.d_seed44_hm:+.4f} ΔP4={test_hm - args.p4_seed44_hm:+.4f})"
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in out_rows:
            w.writerow(row)
    print(f"Wrote {len(out_rows)} rows to {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
