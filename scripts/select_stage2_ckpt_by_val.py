#!/usr/bin/env python3
"""
Select the best stage-B checkpoint by **validation** HM (or Seen/Unseen/ZSL), then report test metrics for that ckpt only.

Does not use test scores for selection. Does not modify training code.

Loads args from --cfg or stage2_dir/args.pkl; applies --root_dir / --dataset_name / --device overrides when set.

Requires:
  --stage_a_dir : stage-1 experiment folder containing *_{score|loss}.pt
  --stage2_dir  : stage-2 experiment folder (args.pkl, checkpoints/)

Appends one summary row to --output_csv (use --no_append to overwrite).
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Load sibling module without packaging scripts/
_spec = importlib.util.spec_from_file_location(
    "_eval_stage2_ckpts",
    _REPO / "scripts" / "evaluate_stage2_checkpoints.py",
)
assert _spec and _spec.loader
_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval)  # type: ignore[attr-defined]


def _load_cfg(cfg_path: Path) -> Any:
    import pickle

    return pickle.load(cfg_path.open("rb"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage2_dir", type=Path, required=True, help="Stage-2 run directory (contains args.pkl, checkpoints/).")
    ap.add_argument("--stage_a_dir", type=Path, required=True, help="Stage-1 run directory containing best *_{score}.pt for model_A.")
    ap.add_argument("--cfg", type=Path, default=None, help="Optional args.pkl path (default: stage2_dir/args.pkl).")
    ap.add_argument("--root_dir", type=str, default=None)
    ap.add_argument("--dataset_name", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--selection_metric", type=str, default="HM", choices=("HM", "Seen", "Unseen", "ZSL"))
    ap.add_argument("--best_model_criterion", type=str, default="score")
    ap.add_argument("--output_csv", type=Path, required=True)
    ap.add_argument("--append", action="store_true", default=True, help="Append to CSV if it exists (default: on).")
    ap.add_argument("--no_append", action="store_true", help="Overwrite output_csv.")
    args = ap.parse_args()
    append = args.append and not args.no_append

    cfg_path = args.cfg or (args.stage2_dir / "args.pkl")
    cfg = _load_cfg(cfg_path)
    if args.root_dir is not None:
        cfg.root_dir = Path(args.root_dir)
    if args.dataset_name is not None:
        cfg.dataset_name = args.dataset_name
    if args.device is not None:
        cfg.device = args.device

    pack = _eval.evaluate_pick_best_and_test(
        args.stage_a_dir,
        args.stage2_dir,
        cfg,
        selection_metric=args.selection_metric,
        best_model_criterion=args.best_model_criterion,
        device=args.device,
    )
    s = pack["summary"]
    run_name = (getattr(cfg, "ablation_run_name", None) or getattr(cfg, "exp_name", "") or "").strip()
    seed = getattr(cfg, "seed", "")

    row = {
        "run_name": run_name,
        "seed": seed,
        "best_ckpt_by_val": s["best_epoch"],
        "best_val_Seen": s["best_val_Seen"],
        "best_val_Unseen": s["best_val_Unseen"],
        "best_val_HM": s["best_val_HM"],
        "best_val_ZSL": s["best_val_ZSL"],
        "test_Seen": s["test_Seen"],
        "test_Unseen": s["test_Unseen"],
        "test_HM": s["test_HM"],
        "test_ZSL": s["test_ZSL"],
        "stage2_dir": str(args.stage2_dir.resolve()),
        "checkpoint_path": s["best_checkpoint"],
    }

    fieldnames = list(row.keys())
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = args.output_csv.is_file()
    mode = "a" if append and file_exists else "w"
    with args.output_csv.open(mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w" or not file_exists:
            w.writeheader()
        w.writerow(row)

    print(
        f"Wrote summary row to {args.output_csv.resolve()}\n"
        f"  best_ckpt_by_val (epoch index)={s['best_epoch']}  val_HM={s['best_val_HM']}  test_HM={s['test_HM']}"
    )


if __name__ == "__main__":
    main()
