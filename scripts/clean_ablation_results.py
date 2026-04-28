#!/usr/bin/env python3
"""
Build a deduplicated ablation CSV without deleting the original.

Rule per run_name: keep one row — prefer checkpoint paths containing 'Apr28'
(case-insensitive); otherwise prefer the lexicographically largest path (later
timestamp in …_AprDD_HH-MM-SS_… folder names usually sorts correctly).

Output: results_ablation_clean_apr28.csv (default paths next to repo root).
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


OUT_COLUMNS = [
    "run_name",
    "seed",
    "Seen",
    "Unseen",
    "HM",
    "ZSL",
    "lambda_proto",
    "lambda_feat",
    "snn_timesteps",
    "snn_conv_threshold_percentile",
    "proto_topk_overlap",
    "proto_top1_agree",
    "snn_clip_ratio",
    "snn_zero_ratio",
    "final_checkpoint_path",
]


def _path_priority(path_str: str) -> Tuple[int, str]:
    """Higher tuple sorts later (prefer Apr28, then larger path string)."""
    p = path_str or ""
    low = p.lower()
    has_apr28 = 1 if "apr28" in low else 0
    # Secondary: full path string (later experiment folders usually sort higher)
    return (has_apr28, p)


def _pick_best_row(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return max(rows, key=lambda r: _path_priority(str(r.get("final_checkpoint_path", "") or "")))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input CSV (default: <repo>/results_ablation.csv)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default: <repo>/results_ablation_clean_apr28.csv)",
    )
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    inp = args.input or (repo / "results_ablation.csv")
    out = args.output or (repo / "results_ablation_clean_apr28.csv")

    if not inp.is_file():
        raise SystemExit(f"Missing input file: {inp}")

    with inp.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames_in = reader.fieldnames or []
        by_run: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in reader:
            rn = (row.get("run_name") or "").strip()
            if not rn:
                continue
            by_run[rn].append(row)

    cleaned: List[Dict[str, Any]] = []
    for run_name in sorted(by_run.keys()):
        rows = by_run[run_name]
        best = _pick_best_row(rows)
        out_row: Dict[str, Any] = {}
        for col in OUT_COLUMNS:
            out_row[col] = best.get(col, "").strip() if isinstance(best.get(col), str) else best.get(col, "")
        cleaned.append(out_row)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in cleaned:
            w.writerow(row)

    print(f"Wrote {len(cleaned)} rows to {out.resolve()} (from {inp.resolve()})")


if __name__ == "__main__":
    main()
