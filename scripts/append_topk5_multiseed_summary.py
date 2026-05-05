#!/usr/bin/env python3
"""
Merge stableproto D and topk5-P4 (mse_topk, k=5, warmup=2) rows from select_stage2_ckpt_by_val.py.

Expected run_name patterns:
  D:      snn_feat_only_T8_p099_seed{seed}_stableproto_D
  topk5:  snn_proto_msetop5_l01_warm2_seed{seed}

Output columns:
  seed, D_HM, topk5_HM, topk5_minus_D,
  D_Seen, D_Unseen, topk5_Seen, topk5_Unseen, D_ZSL, topk5_ZSL,
  best_ckpt_by_val   (from topk5 run only)

Use --no_append to overwrite output_csv.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional


def _float(s: str) -> float:
    return float(str(s).strip())


def _row_for_seed(rows: List[Dict[str, str]], seed: int, kind: str) -> Optional[Dict[str, str]]:
    s = str(seed)
    want_d = f"snn_feat_only_T8_p099_seed{seed}_stableproto_D"
    want_t5 = f"snn_proto_msetop5_l01_warm2_seed{seed}"
    for r in rows:
        if str(r.get("seed", "")).strip() != s:
            continue
        name = (r.get("run_name") or "").strip()
        if kind == "D" and name == want_d:
            return r
        if kind == "topk5" and name == want_t5:
            return r
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input_csv", type=Path, required=True)
    ap.add_argument(
        "--output_csv",
        type=Path,
        default=Path("reports/topk5_multiseed_summary.csv"),
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--no_append", action="store_true")
    args = ap.parse_args()

    text = args.input_csv.read_text(encoding="utf-8")
    rows = list(csv.DictReader(text.splitlines()))

    out_fields = [
        "seed",
        "D_HM",
        "topk5_HM",
        "topk5_minus_D",
        "D_Seen",
        "D_Unseen",
        "topk5_Seen",
        "topk5_Unseen",
        "D_ZSL",
        "topk5_ZSL",
        "best_ckpt_by_val",
    ]

    merged: List[Dict[str, str]] = []
    for seed in args.seeds:
        d_r = _row_for_seed(rows, seed, "D")
        t5 = _row_for_seed(rows, seed, "topk5")
        if not d_r or not t5:
            missing = []
            if not d_r:
                missing.append("D")
            if not t5:
                missing.append("topk5")
            print(f"[WARN] seed {seed}: missing {missing}; skip.")
            continue
        d_hm = _float(d_r["test_HM"])
        t_hm = _float(t5["test_HM"])
        merged.append(
            {
                "seed": str(seed),
                "D_HM": d_r["test_HM"],
                "topk5_HM": t5["test_HM"],
                "topk5_minus_D": f"{t_hm - d_hm:.4f}",
                "D_Seen": d_r["test_Seen"],
                "D_Unseen": d_r["test_Unseen"],
                "topk5_Seen": t5["test_Seen"],
                "topk5_Unseen": t5["test_Unseen"],
                "D_ZSL": d_r["test_ZSL"],
                "topk5_ZSL": t5["test_ZSL"],
                "best_ckpt_by_val": str(t5["best_ckpt_by_val"]),
            }
        )

    if not merged:
        raise SystemExit("No rows written (missing D/topk5 pairs).")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    append = not args.no_append and args.output_csv.is_file()
    mode = "a" if append else "w"
    with args.output_csv.open(mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        if not append:
            w.writeheader()
        for row in merged:
            w.writerow(row)

    print(f"Wrote {len(merged)} row(s) to {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
