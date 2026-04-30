#!/usr/bin/env python3
"""
Build one row per seed: D_HM vs P4 (mse_topk+warmup) from rows produced by select_stage2_ckpt_by_val.py.

Expected run_name patterns (same protocol as seed43 stable proto):
  D:  snn_feat_only_T8_p099_seed{seed}_stableproto_D
  P4: snn_proto_msetop10_l01_warm2_seed{seed}

Appends merged rows to --output_csv (default: reports/stable_proto_multiseed_summary.csv).
Use --no_append to overwrite the output file.

Does not run training or evaluation.
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
        help="CSV containing select_stage2_ckpt_by_val rows (can concatenate multiple runs).",
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
        help="Seeds to emit rows for (must have both D and P4 rows in input_csv).",
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
        "Seen",
        "Unseen",
        "HM",
        "ZSL",
        "best_ckpt_by_val",
        "val_HM",
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
                "Seen": p4["test_Seen"],
                "Unseen": p4["test_Unseen"],
                "HM": p4["test_HM"],
                "ZSL": p4["test_ZSL"],
                "best_ckpt_by_val": str(p4["best_ckpt_by_val"]),
                "val_HM": p4["best_val_HM"],
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
