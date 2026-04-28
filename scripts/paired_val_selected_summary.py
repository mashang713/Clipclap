#!/usr/bin/env python3
"""
Build paired comparison rows from reports/stage2_val_selected_summary.csv (rows produced by select_stage2_ckpt_by_val.py).

Matches runs by seed using run_name patterns:
  A: ann_budget_baseline
  D: snn_feat_only
  E: snn_full

Outputs reports/paired_val_selected_summary.csv with per-seed rows plus mean/std rows.
Columns: seed, A_HM, D_HM, E_HM, E_minus_D, E_minus_A   (HM values from test_HM column).
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _method(run_name: str) -> Optional[str]:
    n = (run_name or "").lower()
    if "ann_budget_baseline" in n:
        return "A"
    if "snn_feat_only" in n:
        return "D"
    if "snn_full" in n:
        return "E"
    return None


def _float_cell(x: str) -> float:
    return float(x.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[1]
    ap.add_argument("--input_csv", type=Path, default=repo / "reports" / "stage2_val_selected_summary.csv")
    ap.add_argument("--output_csv", type=Path, default=repo / "reports" / "paired_val_selected_summary.csv")
    args = ap.parse_args()

    if not args.input_csv.is_file():
        raise SystemExit(f"Missing input: {args.input_csv}")

    rows_by_seed_method: Dict[str, Dict[str, float]] = {}
    with args.input_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rn = row.get("run_name") or ""
            meth = _method(rn)
            if meth is None:
                continue
            seed_key = str(row.get("seed", "")).strip()
            if not seed_key:
                continue
            hm = row.get("test_HM")
            if hm is None:
                continue
            rows_by_seed_method.setdefault(seed_key, {})[meth] = _float_cell(hm)

    seeds_sorted = sorted(rows_by_seed_method.keys(), key=lambda s: int(s) if s.isdigit() else s)

    out_rows: List[Dict[str, str]] = []
    cols = ["seed", "A_HM", "D_HM", "E_HM", "E_minus_D", "E_minus_A"]
    stacks: Dict[str, List[float]] = {"A_HM": [], "D_HM": [], "E_HM": [], "E_minus_D": [], "E_minus_A": []}

    for seed in seeds_sorted:
        m = rows_by_seed_method.get(seed, {})
        if not all(k in m for k in ("A", "D", "E")):
            print(f"[WARN] Skip seed {seed}: missing methods A/D/E -> have {sorted(m.keys())}", file=sys.stderr)
            continue
        a_, d_, e_ = m["A"], m["D"], m["E"]
        emd = e_ - d_
        ema = e_ - a_
        out_rows.append(
            {
                "seed": seed,
                "A_HM": f"{a_:.4f}",
                "D_HM": f"{d_:.4f}",
                "E_HM": f"{e_:.4f}",
                "E_minus_D": f"{emd:.4f}",
                "E_minus_A": f"{ema:.4f}",
            }
        )
        stacks["A_HM"].append(a_)
        stacks["D_HM"].append(d_)
        stacks["E_HM"].append(e_)
        stacks["E_minus_D"].append(emd)
        stacks["E_minus_A"].append(ema)

    def _mean_std(xs: List[float]) -> Tuple[float, float]:
        if not xs:
            return float("nan"), float("nan")
        if len(xs) == 1:
            return xs[0], 0.0
        return statistics.mean(xs), statistics.stdev(xs)

    mean_row = {"seed": "mean"}
    std_row = {"seed": "std"}
    for c in cols[1:]:
        m, s = _mean_std(stacks[c])
        mean_row[c] = f"{m:.4f}" if not math.isnan(m) else ""
        std_row[c] = f"{s:.4f}" if not math.isnan(s) else ""

    if out_rows:
        out_rows.append(mean_row)
        out_rows.append(std_row)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in cols})

    print(f"Wrote {len(out_rows)} rows to {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
