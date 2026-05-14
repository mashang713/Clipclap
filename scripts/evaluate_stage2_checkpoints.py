#!/usr/bin/env python3
"""
Sweep every stage-2 checkpoint with the same evaluation pipeline as get_evaluation.py:

  model_A weights from stage_a_dir (*_{score|loss}.pt per criterion)
  For each checkpoints/ClipClap_model_score_ckpt_*.pt under stage_b_dir:
    load model_B from that file
    Same beta tuning as src.test.test: tune on val with model_A, then evaluate model_B on --split.

Outputs CSV columns:
  checkpoint, epoch, split, Seen, Unseen, HM, ZSL   (percentages 0-100)

--selection_metric is informational only here (selection happens in select_stage2_ckpt_by_val.py).

Does NOT append to results_ablation.csv.

Requires stage_b_dir/args.pkl unless --cfg points elsewhere.

Example:
  python scripts/evaluate_stage2_checkpoints.py \\
    --stage_a_dir ~/ClipClap-GZSL/logs/.../stage1_folder \\
    --stage_b_dir ~/ClipClap-GZSL/logs/snn_full_T8_p099_seed43_apr28/snn_full_..._Apr28_... \\
    --split val \\
    --output_csv ~/ClipClap-GZSL/reports/sweep_seed43_val.csv
"""
from __future__ import annotations

import argparse
import copy
import sys
import csv
import logging
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from torch.utils import data

from src.clipclap_model import build_clipclap_model
from src.dataset import ActivityNetDataset, AudioSetZSLDataset, ContrastiveDataset, DefaultCollator, UCFDataset, VGGSoundDataset
from src.utils import evaluate_dataset_baseline, load_model_parameters, load_model_weights
from src.utils_improvements import get_model_params


def _load_ckpts_sorted(stage_b: Path):
    ck_dir = stage_b / "checkpoints"
    if not ck_dir.is_dir():
        raise FileNotFoundError(f"No checkpoints dir: {ck_dir}")
    paths = sorted(ck_dir.glob("ClipClap_model_score_ckpt_*.pt"), key=lambda p: int(re.search(r"_ckpt_(\d+)\.pt", p.name).group(1)))
    return paths


def _datasets(config, eval_bs: int, eval_num_workers: int, batch_seqlen_test: str, maxlen: int, trim: str):
    dataset_name = config.dataset_name
    if dataset_name == "AudioSetZSL":
        val_all_dataset = AudioSetZSLDataset(args=config, dataset_split="val", zero_shot_mode="all")
        test_dataset = AudioSetZSLDataset(args=config, dataset_split="test", zero_shot_mode="all")
    elif dataset_name == "VGGSound":
        val_all_dataset = VGGSoundDataset(args=config, dataset_split="val", zero_shot_mode=None)
        test_dataset = VGGSoundDataset(args=config, dataset_split="test", zero_shot_mode=None)
    elif dataset_name == "UCF":
        val_all_dataset = UCFDataset(args=config, dataset_split="val", zero_shot_mode=None)
        test_dataset = UCFDataset(args=config, dataset_split="test", zero_shot_mode=None)
    elif dataset_name == "ActivityNet":
        val_all_dataset = ActivityNetDataset(args=config, dataset_split="val", zero_shot_mode=None)
        test_dataset = ActivityNetDataset(args=config, dataset_split="test", zero_shot_mode=None)
    else:
        raise NotImplementedError(dataset_name)

    contrastive_val = ContrastiveDataset(val_all_dataset)
    contrastive_test = ContrastiveDataset(test_dataset)
    collator = DefaultCollator(mode=batch_seqlen_test, max_len=maxlen, trim=trim)
    val_loader = data.DataLoader(contrastive_val, collate_fn=collator, batch_size=eval_bs, num_workers=eval_num_workers)
    test_loader = data.DataLoader(contrastive_test, collate_fn=collator, batch_size=eval_bs, num_workers=eval_num_workers)
    return (val_all_dataset, val_loader), (test_dataset, test_loader)


def _metrics_for_split(
    val_dataset: tuple,
    test_dataset: tuple,
    model_A: torch.nn.Module,
    model_B: torch.nn.Module,
    device: Union[str, torch.device],
    args: Any,
    split: str,
) -> Dict[str, float]:
    """Match src.test._get_test_performance: val betas from model_A; then model_B on val or test."""
    if split not in ("val", "test"):
        raise ValueError(split)
    val_evaluation = evaluate_dataset_baseline(
        val_dataset,
        model_A,
        device,
        args.distance_fn,
        new_model_sequence=getattr(args, "new_model_sequence", False),
        args=args,
        save_performances=False,
    )
    best_beta_combined = (1.0 / 3.0) * (
        val_evaluation["audio"]["beta"]
        + val_evaluation["video"]["beta"]
        + val_evaluation["both"]["beta"]
        + 1e-10
    )
    target = test_dataset if split == "test" else val_dataset
    out = evaluate_dataset_baseline(
        target,
        model_B,
        device,
        args.distance_fn,
        best_beta=best_beta_combined,
        new_model_sequence=getattr(args, "new_model_sequence", False),
        args=args,
        save_performances=False,
    )
    return out["both"]


def _ensure_root_dir_path(cfg: Any) -> None:
    """Dataset code uses cfg.root_dir / subpaths; argparse and overrides often pass str."""
    rd = getattr(cfg, "root_dir", None)
    if rd is not None and not isinstance(rd, Path):
        cfg.root_dir = Path(rd)


def model_params_from_cfg(cfg: Any):
    return get_model_params(
        cfg.lr,
        cfg.reg_loss,
        cfg.embedding_dropout,
        cfg.decoder_dropout,
        cfg.additional_dropout,
        cfg.embeddings_hidden_size,
        cfg.decoder_hidden_size,
        cfg.embeddings_batch_norm,
        cfg.rec_loss,
        cfg.cross_entropy_loss,
        cfg.transformer_use_embedding_net,
        cfg.transformer_dim,
        cfg.transformer_depth,
        cfg.transformer_heads,
        cfg.transformer_dim_head,
        cfg.transformer_mlp_dim,
        cfg.transformer_dropout,
        cfg.transformer_embedding_dim,
        cfg.transformer_embedding_time_len,
        cfg.transformer_embedding_dropout,
        cfg.transformer_embedding_time_embed_type,
        cfg.transformer_embedding_fourier_scale,
        cfg.transformer_embedding_embed_augment_position,
        cfg.lr_scheduler,
        cfg.optimizer,
        cfg.use_self_attention,
        cfg.use_cross_attention,
        cfg.transformer_average_features,
        cfg.audio_only,
        cfg.video_only,
        cfg.transformer_use_class_token,
        cfg.transformer_embedding_modality,
        cfg.modality,
        cfg.word_embeddings,
        getattr(cfg, "model_backend", "ann"),
        getattr(cfg, "snn_num_steps", 10),
        getattr(cfg, "snn_beta", 0.9),
        getattr(cfg, "snn_threshold", 1.0),
        getattr(cfg, "use_snn_conversion", False),
        getattr(cfg, "snn_timesteps", 4),
        getattr(cfg, "lambda_proto", 1.0),
        getattr(cfg, "lambda_feat", 1.0),
        getattr(cfg, "proto_temperature", 1.0),
        getattr(cfg, "snn_conv_threshold_percentile", 0.99),
        getattr(cfg, "proto_kd_type", "kl_all"),
        getattr(cfg, "proto_topk", 10),
        getattr(cfg, "proto_warmup_epochs", 0),
        getattr(cfg, "proto_conf_margin", 0.0),
        getattr(cfg, "debug_print_shapes", False),
        getattr(cfg, "use_teacher_parallel_snn", False),
        getattr(cfg, "teacher_snn_timesteps", 4),
        getattr(cfg, "teacher_snn_gamma", 0.1),
        getattr(cfg, "teacher_snn_alpha", 0.1),
        getattr(cfg, "teacher_snn_beta", 1.0),
        getattr(cfg, "teacher_snn_hidden_dim", 512),
        getattr(cfg, "teacher_snn_threshold", 1.0),
        getattr(cfg, "teacher_snn_decay", 0.9),
        getattr(cfg, "teacher_snn_dropout", 0.1),
        getattr(cfg, "teacher_snn_fusion_mode", "add"),
        getattr(cfg, "teacher_ann_gate_snn", False),
        getattr(cfg, "teacher_gate_strength", 0.5),
        getattr(cfg, "teacher_leak_strength", 0.2),
        getattr(cfg, "teacher_spike_scale_strength", 0.2),
        getattr(cfg, "teacher_fire_rate_target", 0.1),
        getattr(cfg, "teacher_fire_rate_reg", 0.0),
    )


def evaluate_all_ckpts(
    stage_a_dir: Path,
    stage_b_dir: Path,
    cfg: Any,
    split: str,
    best_model_criterion: str = "score",
    device: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Returns one dict per checkpoint with keys checkpoint, epoch, split, Seen.. (percent floats as formatted strings)."""
    _ensure_root_dir_path(cfg)
    stage_a = stage_a_dir.resolve()
    stage_b = stage_b_dir.resolve()
    dev = device or getattr(cfg, "device", "cuda:0")
    eval_bs = int(cfg.eval_bs)
    eval_num_workers = int(cfg.eval_num_workers)
    bs_test = getattr(cfg, "batch_seqlen_test", "max")
    maxlen = getattr(cfg, "batch_seqlen_test_maxlen", 300)
    trim = getattr(cfg, "batch_seqlen_test_trim", "center")

    model_params = model_params_from_cfg(cfg)
    model_A = build_clipclap_model(model_params, input_size_audio=cfg.input_size_audio, input_size_video=cfg.input_size_video)
    model_B = copy.deepcopy(model_A)

    score_pt = list(stage_a.glob(f"*_{best_model_criterion}.pt"))
    if not score_pt:
        raise FileNotFoundError(f"No *_{best_model_criterion}.pt in {stage_a}")
    weights_a_path = sorted(score_pt)[0]
    load_model_weights(weights_a_path, model_A)

    val_ds, test_ds = _datasets(cfg, eval_bs, eval_num_workers, bs_test, maxlen, trim)
    ckpts = _load_ckpts_sorted(stage_b)
    if not ckpts:
        raise FileNotFoundError(f"No ckpt files in {stage_b}/checkpoints")

    model_A.to(dev)
    model_B.to(dev)

    rows: List[Dict[str, Any]] = []
    for ckpt_path in ckpts:
        m = re.search(r"_ckpt_(\d+)\.pt", ckpt_path.name)
        epoch_ix = int(m.group(1)) if m else -1
        load_dict = torch.load(ckpt_path, map_location=dev)
        load_model_parameters(model_B, load_dict["model"])
        model_B.eval()
        model_A.eval()

        both = _metrics_for_split(val_ds, test_ds, model_A, model_B, dev, cfg, split)
        rows.append(
            {
                "checkpoint": str(ckpt_path.resolve()),
                "epoch": epoch_ix,
                "split": split,
                "Seen": f"{100 * float(both['seen']):.4f}",
                "Unseen": f"{100 * float(both['unseen']):.4f}",
                "HM": f"{100 * float(both['hm']):.4f}",
                "ZSL": f"{100 * float(both['zsl']):.4f}",
                "_both_frac": both,
            }
        )
    return rows


def _both_fraction_metric(both: Dict[str, Any], selection_metric: str) -> float:
    key = selection_metric.upper()
    if key == "HM":
        return float(both["hm"])
    if key == "SEEN":
        return float(both["seen"])
    if key == "UNSEEN":
        return float(both["unseen"])
    if key == "ZSL":
        return float(both["zsl"])
    raise ValueError(f"Unknown selection_metric: {selection_metric}")


def evaluate_pick_best_and_test(
    stage_a_dir: Path,
    stage_b_dir: Path,
    cfg: Any,
    selection_metric: str = "HM",
    best_model_criterion: str = "score",
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Val sweep -> pick best by fraction metric HM/Seen/... ; return best row + test metrics for that ckpt."""
    val_rows = evaluate_all_ckpts(stage_a_dir, stage_b_dir, cfg, split="val", best_model_criterion=best_model_criterion, device=device)
    best_idx = max(
        range(len(val_rows)),
        key=lambda i: _both_fraction_metric(val_rows[i]["_both_frac"], selection_metric),
    )
    best = val_rows[best_idx]
    best_path = Path(best["checkpoint"])
    dev = device or getattr(cfg, "device", "cuda:0")

    model_params = model_params_from_cfg(cfg)
    model_A = build_clipclap_model(model_params, input_size_audio=cfg.input_size_audio, input_size_video=cfg.input_size_video)
    model_B = copy.deepcopy(model_A)
    stage_a = stage_a_dir.resolve()
    score_pt = list(stage_a.glob(f"*_{best_model_criterion}.pt"))
    weights_a_path = sorted(score_pt)[0]
    load_model_weights(weights_a_path, model_A)
    val_ds, test_ds = _datasets(
        cfg,
        int(cfg.eval_bs),
        int(cfg.eval_num_workers),
        getattr(cfg, "batch_seqlen_test", "max"),
        getattr(cfg, "batch_seqlen_test_maxlen", 300),
        getattr(cfg, "batch_seqlen_test_trim", "center"),
    )
    load_dict = torch.load(best_path, map_location=dev)
    load_model_parameters(model_B, load_dict["model"])
    model_A.to(dev)
    model_B.to(dev)
    model_A.eval()
    model_B.eval()
    test_both = _metrics_for_split(val_ds, test_ds, model_A, model_B, dev, cfg, "test")
    out = {
        "best_epoch": best["epoch"],
        "best_checkpoint": str(best_path.resolve()),
        "best_val_Seen": f"{100 * float(best['_both_frac']['seen']):.4f}",
        "best_val_Unseen": f"{100 * float(best['_both_frac']['unseen']):.4f}",
        "best_val_HM": f"{100 * float(best['_both_frac']['hm']):.4f}",
        "best_val_ZSL": f"{100 * float(best['_both_frac']['zsl']):.4f}",
        "test_Seen": f"{100 * float(test_both['seen']):.4f}",
        "test_Unseen": f"{100 * float(test_both['unseen']):.4f}",
        "test_HM": f"{100 * float(test_both['hm']):.4f}",
        "test_ZSL": f"{100 * float(test_both['zsl']):.4f}",
    }
    for r in val_rows:
        r.pop("_both_frac", None)
    return {"summary": out, "val_rows": val_rows}


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage_a_dir", type=Path, required=True)
    ap.add_argument("--stage_b_dir", type=Path, required=True)
    ap.add_argument("--output_csv", type=Path, required=True)
    ap.add_argument("--split", type=str, choices=("val", "test"), default="test", help="Evaluate model_B on val (for selection) or test.")
    ap.add_argument("--selection_metric", type=str, default="HM", choices=("HM", "Seen", "Unseen", "ZSL"), help="Informational only for this script.")
    ap.add_argument("--cfg", type=Path, default=None, help="Optional args.pkl; default stage_b_dir/args.pkl")
    ap.add_argument("--root_dir", type=str, default=None, help="Override cfg.root_dir before building datasets.")
    ap.add_argument("--dataset_name", type=str, default=None, help="Override cfg.dataset_name.")
    ap.add_argument("--device", type=str, default=None, help="Override cfg.device.")
    ap.add_argument("--best_model_criterion", type=str, default="score")
    args_ns = ap.parse_args()

    cfg_path = args_ns.cfg or (args_ns.stage_b_dir / "args.pkl")
    cfg = pickle.load(cfg_path.open("rb"))
    if args_ns.root_dir is not None:
        cfg.root_dir = Path(args_ns.root_dir)
    if args_ns.dataset_name is not None:
        cfg.dataset_name = args_ns.dataset_name
    if args_ns.device is not None:
        cfg.device = args_ns.device

    rows_out = evaluate_all_ckpts(
        args_ns.stage_a_dir,
        args_ns.stage_b_dir,
        cfg,
        split=args_ns.split,
        best_model_criterion=args_ns.best_model_criterion,
        device=args_ns.device,
    )
    for r in rows_out:
        r.pop("_both_frac", None)

    args_ns.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["checkpoint", "epoch", "split", "Seen", "Unseen", "HM", "ZSL"]
    with args_ns.output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            w.writerow({k: r[k] for k in fieldnames})

    print(f"Wrote {len(rows_out)} rows to {args_ns.output_csv.resolve()} (split={args_ns.split}, selection_metric={args_ns.selection_metric})")


if __name__ == "__main__":
    main()
