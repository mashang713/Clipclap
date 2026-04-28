#!/usr/bin/env python3
"""
Dedupe results_ablation.csv -> results_ablation_clean_dedup.csv (never deletes original).

Rules:
  - Exact duplicate run_name: keep one row — prefer path containing 'apr28' (case-insensitive),
    else lexicographically largest final_checkpoint_path (later runs usually sort later).

  - Typo warnings (no auto-merge): if two distinct run_name strings map to the same
    'canonical' key (snnfeat -> snn_feat, collapse double underscores), print WARNING
    for human review.

Outputs:
  - results_ablation_clean_dedup.csv (default next to repo root)
  - dedup_report.txt next to output with duplicate groups + typo suspects
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _path_priority(path_str: str) -> Tuple[int, str]:
    p = path_str or ""
    low = p.lower()
    has_apr28 = 1 if "apr28" in low else 0
    return (has_apr28, p)


def _pick_best_row(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return max(rows, key=lambda r: _path_priority(str(r.get("final_checkpoint_path", "") or "")))


def _canonical_run_name(name: str) -> str:
    """Normalize obvious typo variants for equality checks only (NOT for merging rows)."""
    s = (name or "").strip().lower()
    s = s.replace("snnfeat", "snn_feat")
    s = re.sub(r"_+", "_", s)
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    inp = args.input or (repo / "results_ablation.csv")
    out_csv = args.output or (repo / "results_ablation_clean_dedup.csv")
    out_report = out_csv.with_name(out_csv.stem + "_report.txt")

    if not inp.is_file():
        raise SystemExit(
            f"Missing input: {inp}\n"
            "Place results_ablation.csv in the repo root, or pass --input PATH_TO_CSV."
        )

    with inp.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames_in = reader.fieldnames or []
        all_rows = list(reader)

    by_run: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        rn = (row.get("run_name") or "").strip()
        if not rn:
            continue
        by_run[rn].append(row)

    duplicate_groups: List[Tuple[str, int]] = []
    cleaned: List[Dict[str, Any]] = []
    for rn in sorted(by_run.keys()):
        rows = by_run[rn]
        if len(rows) > 1:
            duplicate_groups.append((rn, len(rows)))
        cleaned.append(_pick_best_row(rows))

    # Typo suspects: distinct run_name -> same canonical
    canon_map: Dict[str, List[str]] = defaultdict(list)
    for rn in by_run.keys():
        canon_map[_canonical_run_name(rn)].append(rn)
    typo_suspects: List[Tuple[str, ...]] = []
    for _canon, names in canon_map.items():
        uniq = sorted(set(names))
        if len(uniq) > 1:
            typo_suspects.append(tuple(uniq))

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_in, extrasaction="ignore")
        w.writeheader()
        for row in cleaned:
            w.writerow(row)

    lines = [
        f"Input: {inp.resolve()}",
        f"Output: {out_csv.resolve()}",
        f"Rows read: {len(all_rows)}, unique run_name: {len(cleaned)}",
        "",
        "=== DUPLICATE run_name (count > 1 before dedup) ===",
    ]
    if duplicate_groups:
        for rn, c in sorted(duplicate_groups, key=lambda x: (-x[1], x[0])):
            lines.append(f"  {rn}  ({c} rows) -> kept 1 row (see path_priority: Apr28 > lex path)")
    else:
        lines.append("  (none)")
    lines += [
        "",
        "=== SUSPECTED typo / alias (same canonical key, different run_name) — DO NOT AUTO-MERGE ===",
    ]
    if typo_suspects:
        for tup in typo_suspects:
            lines.append("  " + " | ".join(tup))
    else:
        lines.append("  (none)")

    out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if typo_suspects:
        print("WARNING: suspected typo/alias groups (see report file). Do not merge without review:")
        for tup in typo_suspects:
            print("   ", " | ".join(tup))

    print(f"Wrote {len(cleaned)} rows to {out_csv.resolve()}")
    print(f"Report: {out_report.resolve()}")


if __name__ == "__main__":
    main()
