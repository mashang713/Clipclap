#!/usr/bin/env python3
"""
Print alignment between stage-1/stage-2 training args and evaluation checkpoint naming.

Reads:
  stage_a_dir/args.pkl   (stage-1 experiment — retrain_all False when saved at stage1 start — actually pickled per stage)
  stage_b_dir/args.pkl   (stage-2 experiment — contains ablation_best_epoch, ablation_stage2_epochs)

Also resolves final evaluation checkpoint rule used by get_evaluation.py:
  stage_A score.pt -> epoch_A from pickle inside file
  stage_B checkpoints/ClipClap_model_score_ckpt_{epoch_A - 1}.pt

Optional --checkpoint_path overrides printed final ckpt path for sanity-check vs CSV row.
"""
from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage_a_dir", type=Path, required=True)
    ap.add_argument("--stage_b_dir", type=Path, required=True)
    ap.add_argument("--best_model_criterion", type=str, default="score")
    ap.add_argument("--checkpoint_path", type=Path, default=None, help="Optional path from CSV row for comparison.")
    args = ap.parse_args()

    stage_a = args.stage_a_dir.resolve()
    stage_b = args.stage_b_dir.resolve()

    cfg_b = pickle.load((stage_b / "args.pkl").open("rb"))
    cfg_a_path = stage_a / "args.pkl"
    if cfg_a_path.is_file():
        cfg_a = pickle.load(cfg_a_path.open("rb"))
    else:
        cfg_a = None

    best_epoch_s1 = getattr(cfg_b, "ablation_best_epoch", None)
    stage2_eps = getattr(cfg_b, "ablation_stage2_epochs", None)
    stage1_eps = getattr(cfg_b, "ablation_stage1_epochs", None)

    score_files = list(stage_a.glob(f"*_{args.best_model_criterion}.pt"))
    if not score_files:
        raise SystemExit(f"No *_{args.best_model_criterion}.pt under {stage_a}")
    weights_a = sorted(score_files)[0]

    import torch

    ck_a = torch.load(weights_a, map_location="cpu")
    epoch_saved_a = ck_a.get("epoch")
    print("=== Saved in stage-A checkpoint pickle ===")
    print(f"  file: {weights_a}")
    print(f"  epoch (inside pickle): {epoch_saved_a}")

    # Evaluation selects ckpt index epoch_A - 1 for stage B (see get_evaluation.py)
    epoch_a_for_rule = epoch_saved_a
    ix_b = int(epoch_a_for_rule) - 1 if epoch_a_for_rule is not None else None
    ckpt_eval = stage_b / "checkpoints" / f"ClipClap_model_score_ckpt_{ix_b}.pt"
    print("\n=== Derived evaluation mapping (same as get_evaluation.py) ===")
    print(f"  stage1_best_epoch (from stage-B args pickle): {best_epoch_s1}")
    print(f"  stage1_epochs (training epochs in stage-1): {stage1_eps}")
    print(f"  stage2_epochs (training epochs in stage-2): {stage2_eps}")
    print(f"  epoch value inside stage-A *_score.pt (epoch_A): {epoch_a_for_rule}")
    print(f"  stage-B checkpoint index used at eval: epoch_A - 1 = {ix_b}")
    print(f"  expected evaluation weights path:\n    {ckpt_eval}")
    print(f"  exists: {ckpt_eval.is_file()}")

    if args.checkpoint_path:
        p = args.checkpoint_path.resolve()
        print("\n=== CSV / manual checkpoint_path ===")
        print(f"  {p}")
        print(f"  exists: {p.is_file()}")
        m = re.search(r"_ckpt_(\d+)\.pt", str(p))
        if m:
            idx = int(m.group(1))
            print(f"  parsed ckpt index: {idx}  (= stage2 epoch index if ckpts named ckpt_0..ckpt_{stage2_eps or '?'} )")


if __name__ == "__main__":
    main()
