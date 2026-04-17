#!/usr/bin/env python
"""
Phase B working-point grid: snn_threshold x phase_b_audio_input_scale only.

Runs ``scripts/phase_b_train_audio.py`` nine times (3x3), writes ``phase_b_thr_scale_sweep.csv``
and prints a Markdown table. Parses ``train.log`` for:
  - final test line (Seen/Unseen/GZSL/ZSL in %), else last VALID line (metrics in 0..1 -> x100)
  - last ``PhaseB-frontend_diag`` aggregate (firing stats)
  - TRAIN losses per epoch for a coarse stability hint

Example::

  python scripts/phase_b_sweep_thr_scale.py \\
    --cfg config/phase_b_ucf_audio.yaml \\
    --root_dir D:/data/UCF \\
    --log_base runs/phase_b_grid_20260417

Requires the same env/deps as Phase B training (CUDA, data paths, etc.).
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TEST_RE = re.compile(
    r"Seen performance=([\d.]+),\s*Unseen performance=([\d.]+),\s*"
    r"GZSL performance=([\d.]+),\s*ZSL performance=([\d.]+)"
)
VALID_RE = re.compile(
    r"VALID\s+Epoch:\s*(\d+)/(\d+)\s+.*?"
    r"ZSL score:\s*([\d.]+)\s+.*?"
    r"Seen score:\s*([\d.]+)\s+.*?"
    r"Unseen score:\s*([\d.]+)\s+.*?"
    r"HM:\s*([\d.]+)"
)
TRAIN_LOSS_RE = re.compile(r"TRAIN\s+Epoch:\s*(\d+)/\d+.*?Loss:\s*([\d.]+)")
DIAG_RE = re.compile(r"PhaseB-frontend_diag[^\n]+")


def _parse_log(train_log: Path) -> dict:
    text = train_log.read_text(encoding="utf-8", errors="replace")
    seen = unseen = hm = zsl = None
    m_test = None
    for m in TEST_RE.finditer(text):
        m_test = m
    if m_test:
        seen, unseen, hm, zsl = (float(m_test.group(i)) for i in range(1, 5))
        source = "test_log"
    else:
        m_val = None
        for m in VALID_RE.finditer(text):
            m_val = m
        if m_val:
            zsl, seen, unseen, hm = (float(m_val.group(i)) * 100.0 for i in range(3, 7))
            source = "val_log_x100"
        else:
            source = "missing"

    train_losses: list[tuple[int, float]] = []
    for m in TRAIN_LOSS_RE.finditer(text):
        train_losses.append((int(m.group(1)), float(m.group(2))))
    stability = ""
    if train_losses:
        by_ep: dict[int, list[float]] = {}
        for ep, lo in train_losses:
            by_ep.setdefault(ep, []).append(lo)
        means = [sum(v) / len(v) for _, v in sorted(by_ep.items())]
        if len(means) >= 2:
            stability = f"loss_mean_last={means[-1]:.4f} drift={means[-1]-means[0]:+.4f}"
        else:
            stability = f"loss_mean={means[0]:.4f}" if means else ""

    diag_line = ""
    for m in DIAG_RE.finditer(text):
        diag_line = m.group(0).strip()
    diag_short = diag_line.replace("PhaseB-frontend_diag ", "")[:220]

    return {
        "seen": seen,
        "unseen": unseen,
        "hm": hm,
        "zsl": zsl,
        "metrics_source": source,
        "train_stability": stability,
        "diag_snippet": diag_short,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=Path, default=ROOT / "config" / "phase_b_ucf_audio.yaml")
    p.add_argument("--root_dir", type=Path, required=True)
    p.add_argument("--log_base", type=Path, required=True, help="Directory under which per-run folders are created")
    p.add_argument("--dry_run", action="store_true", help="Print commands only")
    args = p.parse_args()

    thrs = [0.08, 0.10, 0.12]
    scales = [1.0, 2.0, 4.0]
    args.log_base.mkdir(parents=True, exist_ok=True)
    out_csv = args.log_base / "phase_b_thr_scale_sweep.csv"
    rows: list[dict] = []

    for thr in thrs:
        for sc in scales:
            tag = f"thr{thr:g}_scale{sc:g}".replace(".", "p")
            run_dir = args.log_base / tag
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "phase_b_train_audio.py"),
                "-c",
                str(args.cfg),
                "--root_dir",
                str(args.root_dir),
                "--log_dir",
                str(run_dir),
                "--epochs",
                "5",
                "--n_batches",
                "300",
                "--eval_modality",
                "audio",
                "--phase_b_audio_source",
                "offline_as_mel",
                "--snn_threshold",
                str(thr),
                "--phase_b_audio_input_scale",
                str(sc),
                "--phase_b_frontend_diag",
                "true",
                "--phase_b_diag_log_interval",
                "50",
            ]
            print("RUN:", " ".join(cmd), flush=True)
            if not args.dry_run:
                r = subprocess.run(cmd, cwd=str(ROOT))
                if r.returncode != 0:
                    print(f"WARNING: exit code {r.returncode} for {tag}", flush=True)

            train_log = run_dir / "train.log"
            parsed = _parse_log(train_log) if train_log.is_file() else {
                "seen": None,
                "unseen": None,
                "hm": None,
                "zsl": None,
                "metrics_source": "no_log",
                "train_stability": "",
                "diag_snippet": "",
            }
            row = {
                "snn_threshold": thr,
                "phase_b_audio_input_scale": sc,
                "run_dir": str(run_dir),
                **parsed,
            }
            rows.append(row)

    fieldnames = [
        "snn_threshold",
        "phase_b_audio_input_scale",
        "seen",
        "unseen",
        "hm",
        "zsl",
        "metrics_source",
        "train_stability",
        "diag_snippet",
        "run_dir",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"\nWrote {out_csv}\n")
    # Markdown table (copy-friendly)
    hdr = "| thr | scale | Seen | Unseen | HM | ZSL | stability | diag (trim) |"
    sep = "|---|---:|---:|---:|---:|---:|---|---|"
    print(hdr)
    print(sep)
    for row in rows:
        def fmt(x):
            return "" if x is None else f"{x:.2f}"

        d = (row.get("diag_snippet") or "").replace("|", "/")
        print(
            f"| {row['snn_threshold']} | {row['phase_b_audio_input_scale']} | "
            f"{fmt(row.get('seen'))} | {fmt(row.get('unseen'))} | {fmt(row.get('hm'))} | {fmt(row.get('zsl'))} | "
            f"{row.get('train_stability', '')} | {d[:80]} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
