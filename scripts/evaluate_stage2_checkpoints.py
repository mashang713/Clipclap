#!/usr/bin/env python3
"""
Sweep every stage-2 checkpoint with the same evaluation pipeline as get_evaluation.py:

  model_A weights from stage_a_dir (*_{score|loss}.pt per criterion)
  For each checkpoints/ClipClap_model_score_ckpt_*.pt under stage_b_dir:
    load model_B from that file
    run src.test.test(...) -> Seen/Unseen/HM/ZSL on test (same beta tuning from val via model_A)

Outputs CSV columns:
  checkpoint, epoch, Seen, Unseen, HM, ZSL   (percentages 0-100)

Does NOT append to results_ablation.csv.

Requires stage_b_dir/args.pkl (same as evaluation config).

Example:
  python scripts/evaluate_stage2_checkpoints.py \\
    --stage_a_dir ~/ClipClap-GZSL/logs/.../stage1_folder \\
    --stage_b_dir ~/ClipClap-GZSL/logs/snn_full_T8_p099_seed43_apr28/snn_full_..._Apr28_... \\
    --output_csv ~/ClipClap-GZSL/reports/sweep_seed43.csv
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

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from torch.utils import data

from src.clipclap_model import build_clipclap_model
from src.dataset import ActivityNetDataset, AudioSetZSLDataset, ContrastiveDataset, DefaultCollator, UCFDataset, VGGSoundDataset
from src.logger import PD_Stats
from src.test import test
from src.utils import load_model_parameters, load_model_weights
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


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage_a_dir", type=Path, required=True)
    ap.add_argument("--stage_b_dir", type=Path, required=True)
    ap.add_argument("--output_csv", type=Path, required=True)
    ap.add_argument("--best_model_criterion", type=str, default="score")
    args_ns = ap.parse_args()

    stage_a = args_ns.stage_a_dir.resolve()
    stage_b = args_ns.stage_b_dir.resolve()
    cfg = pickle.load((stage_b / "args.pkl").open("rb"))

    device = getattr(cfg, "device", "cuda:0")
    eval_bs = int(cfg.eval_bs)
    eval_num_workers = int(cfg.eval_num_workers)
    bs_test = getattr(cfg, "batch_seqlen_test", "max")
    maxlen = getattr(cfg, "batch_seqlen_test_maxlen", 300)
    trim = getattr(cfg, "batch_seqlen_test_trim", "center")

    model_params = get_model_params(
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
    )

    model_A = build_clipclap_model(model_params, input_size_audio=cfg.input_size_audio, input_size_video=cfg.input_size_video)

    model_B = copy.deepcopy(model_A)

    score_pt = list(stage_a.glob(f"*_{args_ns.best_model_criterion}.pt"))
    if not score_pt:
        raise FileNotFoundError(f"No *_{args_ns.best_model_criterion}.pt in {stage_a}")
    weights_a_path = sorted(score_pt)[0]
    load_model_weights(weights_a_path, model_A)

    val_ds, test_ds = _datasets(cfg, eval_bs, eval_num_workers, bs_test, maxlen, trim)

    ckpts = _load_ckpts_sorted(stage_b)
    if not ckpts:
        raise FileNotFoundError(f"No ckpt files in {stage_b}/checkpoints")

    rows = []
    dummy_eval_dir = stage_b / "_eval_ckpt_sweep_tmp"
    dummy_eval_dir.mkdir(exist_ok=True)
    test_stats = PD_Stats(dummy_eval_dir / "dummy.pkl", ["seen", "unseen", "hm", "zsl"])

    model_A.to(device)
    model_B.to(device)

    for ckpt_path in ckpts:
        m = re.search(r"_ckpt_(\d+)\.pt", ckpt_path.name)
        epoch_ix = int(m.group(1)) if m else -1
        load_dict = torch.load(ckpt_path, map_location=device)
        load_model_parameters(model_B, load_dict["model"])
        model_B.eval()
        model_A.eval()

        results = test(
            eval_name=getattr(cfg, "eval_name", "Attention"),
            val_dataset=val_ds,
            test_dataset=test_ds,
            model_A=model_A,
            model_B=model_B,
            device=device,
            distance_fn=cfg.distance_fn,
            test_stats=test_stats,
            eval_dir=dummy_eval_dir,
            new_model_sequence=cfg.new_model_sequence,
            args=cfg,
            save_performances=False,
        )
        both = results["both"]
        rows.append(
            {
                "checkpoint": str(ckpt_path.resolve()),
                "epoch": epoch_ix,
                "Seen": f"{100 * float(both['seen']):.4f}",
                "Unseen": f"{100 * float(both['unseen']):.4f}",
                "HM": f"{100 * float(both['hm']):.4f}",
                "ZSL": f"{100 * float(both['zsl']):.4f}",
            }
        )

    args_ns.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args_ns.output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["checkpoint", "epoch", "Seen", "Unseen", "HM", "ZSL"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {len(rows)} rows to {args_ns.output_csv.resolve()}")


if __name__ == "__main__":
    main()
