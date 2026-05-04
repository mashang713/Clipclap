#!/usr/bin/env python3
"""
Merge D (feat-only, lambda_proto=0) and P4 (mse_topk+warmup) rows from select_stage2_ckpt_by_val.py.

Expected run_name patterns:
  D:  snn_feat_only_T8_p099_seed{seed}_stableproto_D
  P4: snn_proto_msetop10_l01_warm2_seed{seed}

Output columns:
  seed, D_HM, P4_HM, P4_minus_D,
  D_Seen, D_Unseen, P4_Seen, P4_Unseen, D_ZSL, P4_ZSL,
  D_best_ckpt, P4_best_ckpt

Use --no_append to overwrite output_csv (recommended when regenerating the full table).
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
    want_p4 = f"snn_proto_msetop10_l01_warm2_seed{seed}"
    for r in rows:
        if str(r.get("seed", "")).strip() != s:
            continue
        name = (r.get("run_name") or "").strip()
        if kind == "D" and name == want_d:
            return r
        if kind == "P4" and name == want_p4:
            return r
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input_csv",
        type=Path,
        required=True,
        help="CSV containing select_stage2_ckpt_by_val rows.",
    )
    ap.add_argument(
        "--output_csv",
        type=Path,
        default=Path("reports/stable_proto_multiseed_summary.csv"),
    )
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
        help="Seeds to emit (each needs both D and P4 rows in input_csv).",
    )
    ap.add_argument("--no_append", action="store_true", help="Overwrite output_csv.")
    args = ap.parse_args()

    text = args.input_csv.read_text(encoding="utf-8")
    rows = list(csv.DictReader(text.splitlines()))

    out_fields = [
        "seed",
        "D_HM",
        "P4_HM",
        "P4_minus_D",
        "D_Seen",
        "D_Unseen",
        "P4_Seen",
        "P4_Unseen",
        "D_ZSL",
        "P4_ZSL",
        "D_best_ckpt",
        "P4_best_ckpt",
    ]

    merged: List[Dict[str, str]] = []
    for seed in args.seeds:
        d_r = _row_for_seed(rows, seed, "D")
        p4 = _row_for_seed(rows, seed, "P4")
        if not d_r or not p4:
            missing = []
            if not d_r:
                missing.append("D")
            if not p4:
                missing.append("P4")
            print(f"[WARN] seed {seed}: missing {missing}; skip.")
            continue
        d_hm = _float(d_r["test_HM"])
        p4_hm = _float(p4["test_HM"])
        merged.append(
            {
                "seed": str(seed),
                "D_HM": d_r["test_HM"],
                "P4_HM": p4["test_HM"],
                "P4_minus_D": f"{p4_hm - d_hm:.4f}",
                "D_Seen": d_r["test_Seen"],
                "D_Unseen": d_r["test_Unseen"],
                "P4_Seen": p4["test_Seen"],
                "P4_Unseen": p4["test_Unseen"],
                "D_ZSL": d_r["test_ZSL"],
                "P4_ZSL": p4["test_ZSL"],
                "D_best_ckpt": str(d_r["best_ckpt_by_val"]),
                "P4_best_ckpt": str(p4["best_ckpt_by_val"]),
            }
        )

    if not merged:
        raise SystemExit("No rows written (missing D/P4 pairs).")

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
