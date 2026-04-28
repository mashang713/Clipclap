#!/usr/bin/env python3
"""
Paired summary over seeds for ANN baseline (A), feat-only (D), full (E).

Reads results_ablation_clean_apr28.csv (or path you pass), matches rows by run_name:
  A: ann_budget_baseline_seed{seed}
  D: snn_feat_only_T8_p099_seed{seed} optional _apr28
  E: snn_full_T8_p099_seed{seed} optional _apr28

Prints TSV table and mean ± std over seeds present for all three.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _parse_seed(run_name: str) -> Optional[int]:
    m = re.search(r"seed(\d+)", run_name)
    return int(m.group(1)) if m else None


def _kind(run_name: str) -> Optional[str]:
    if re.match(r"^ann_budget_baseline_seed\d+", run_name):
        return "A"
    if re.search(r"^snn_feat_only_T8_p099_seed\d+", run_name):
        return "D"
    if re.search(r"^snn_full_T8_p099_seed\d+", run_name):
        return "E"
    return None


def _hm(row: Dict[str, str]) -> float:
    return float(row.get("HM") or row.get("hm") or "nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Clean CSV (default: <repo>/results_ablation_clean_apr28.csv)",
    )
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    inp = args.input or (repo / "results_ablation_clean_apr28.csv")
    if not inp.is_file():
        raise SystemExit(f"Missing {inp}. Run scripts/clean_ablation_results.py first.")

    by_key: Dict[Tuple[int, str], Dict[str, str]] = {}
    with inp.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rn = (row.get("run_name") or "").strip()
            k = _kind(rn)
            s = _parse_seed(rn)
            if k is None or s is None:
                continue
            by_key[(s, k)] = row

    seeds = sorted({s for (s, _) in by_key.keys()})
    rows_out: List[Tuple[int, float, float, float, float, float]] = []
    for s in seeds:
        a = by_key.get((s, "A"))
        d = by_key.get((s, "D"))
        e = by_key.get((s, "E"))
        if not (a and d and e):
            print(f"# skip seed {s}: missing A={a is not None} D={d is not None} E={e is not None}", flush=True)
            continue
        ahm, dhm, ehm = _hm(a), _hm(d), _hm(e)
        rows_out.append((s, ahm, dhm, ehm, ehm - dhm, ehm - ahm))

    print("seed\tA_HM\tD_HM\tE_HM\tE_minus_D\tE_minus_A")
    for tup in rows_out:
        print("\t".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in tup))

    if not rows_out:
        print("# No seed with A, D, E all present.")
        return

    def col_mean_std(j: int) -> str:
        xs = [t[j] for t in rows_out]
        m = statistics.mean(xs)
        st = statistics.stdev(xs) if len(xs) > 1 else 0.0
        return f"{m:.4f} ± {st:.4f}"

    print(
        "\nmean±std\t"
        + "\t".join(col_mean_std(j) for j in range(1, 6))
    )


if __name__ == "__main__":
    main()
