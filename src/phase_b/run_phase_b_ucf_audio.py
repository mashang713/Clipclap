"""
UCF + audio-only Phase B: ContrastivePhaseBAudio + ClipClapPhaseB_AudioWrapper.
Not used by main.py; invoked from scripts/phase_b_train_audio.py.
"""

from __future__ import annotations

import importlib
import logging

import torch
from torch.utils import data

from src.args import args_main
from src.dataset import UCFDataset, DefaultCollator
from src.metrics import MeanClassAccuracy
from src.sampler import SamplerFactory
from src.train import train
from src.utils import fix_seeds, setup_experiment, get_git_revision_hash, print_model_size

from src.phase_b.model_factory import build_clipclap_phase_b_wrapped, resolve_phase_b_audio_source
from src.phase_b.ucf_phase_b_audio_dataset import ContrastivePhaseBAudio


def run_phase_b_ucf_audio():
    args, eval_args = args_main()
    if args.dataset_name != "UCF":
        raise SystemExit("Phase B skeleton currently supports UCF only.")
    if args.modality != "audio":
        logging.warning("Overriding modality to 'audio' for Phase B.")
        args.modality = "audio"

    if args.input_size is not None:
        args.input_size_audio = args.input_size
        args.input_size_video = args.input_size

    fix_seeds(args.seed)
    logger, log_dir, writer, train_stats, val_stats = setup_experiment(args, "epoch", "loss", "hm")
    logger.info("Git commit hash: %s", get_git_revision_hash())
    logger.info("Phase B (audio): SNN frontend + existing ClipClap audio head — not Phase A main.py")

    if args.retrain_all:
        raise SystemExit("Use retrain_all false for this skeleton (stage-1 style).")

    train_dataset = UCFDataset(
        args=args,
        dataset_split="train",
        zero_shot_mode="train",
    )
    val_all_dataset = UCFDataset(
        args=args,
        dataset_split="val",
        zero_shot_mode=None,
    )

    src = resolve_phase_b_audio_source(args)
    mel_flat_dim = int(getattr(args, "phase_b_mel_flat_dim", 4096))

    contrastive_train = ContrastivePhaseBAudio(train_dataset, source=src, mel_flat_dim=mel_flat_dim)
    contrastive_val = ContrastivePhaseBAudio(val_all_dataset, source=src, mel_flat_dim=mel_flat_dim)

    train_sampler = SamplerFactory(logger).get(
        class_idxs=list(contrastive_train.target_to_indices.values()),
        batch_size=args.bs,
        n_batches=args.n_batches,
        alpha=1,
        kind="random",
    )
    val_sampler = SamplerFactory(logger).get(
        class_idxs=list(contrastive_val.target_to_indices.values()),
        batch_size=args.bs,
        n_batches=args.n_batches,
        alpha=1,
        kind="random",
    )

    if args.selavi:
        collator_train = DefaultCollator(
            mode=args.batch_seqlen_train,
            max_len=args.batch_seqlen_train_maxlen,
            trim=args.batch_seqlen_train_trim,
            rate_video=1,
            rate_audio=1,
        )
        collator_test = DefaultCollator(
            mode=args.batch_seqlen_test,
            max_len=args.batch_seqlen_test_maxlen,
            trim=args.batch_seqlen_test_trim,
            rate_video=1,
            rate_audio=1,
        )
    else:
        collator_train = DefaultCollator(
            mode=args.batch_seqlen_train,
            max_len=args.batch_seqlen_train_maxlen,
            trim=args.batch_seqlen_train_trim,
        )
        collator_test = DefaultCollator(
            mode=args.batch_seqlen_test,
            max_len=args.batch_seqlen_test_maxlen,
            trim=args.batch_seqlen_test_trim,
        )

    train_loader = data.DataLoader(
        dataset=contrastive_train,
        batch_sampler=train_sampler,
        collate_fn=collator_train,
        num_workers=4,
    )
    val_all_loader = data.DataLoader(
        dataset=contrastive_val,
        batch_sampler=val_sampler,
        collate_fn=collator_test,
        num_workers=4,
    )
    final_test_loader = data.DataLoader(
        dataset=contrastive_val,
        collate_fn=collator_test,
        batch_size=args.bs,
        num_workers=4,
    )

    model = build_clipclap_phase_b_wrapped(args, device=args.device)
    if getattr(args, "model_backend", "ann") == "snn" and getattr(args, "snn_init_ann_path", None):
        logger.info("SNN head init from ANN checkpoint: %s", args.snn_init_ann_path)

    print_model_size(model, logger)
    logger.info(model)

    loss_mod = importlib.import_module("src.loss")
    distance_fn = getattr(loss_mod, args.distance_fn)()
    metrics = [
        MeanClassAccuracy(
            model=model,
            dataset=(val_all_dataset, final_test_loader),
            device=args.device,
            distance_fn=distance_fn,
            new_model_sequence=args.new_model_sequence,
            args=args,
        )
    ]

    train(
        train_loader=train_loader,
        val_loader=val_all_loader,
        model=model,
        criterion=None,
        optimizer=None,
        lr_scheduler=None,
        epochs=args.epochs,
        device=args.device,
        writer=writer,
        metrics=metrics,
        train_stats=train_stats,
        val_stats=val_stats,
        log_dir=log_dir,
        new_model_sequence=args.new_model_sequence,
        args=args,
    )
    logger.info("Phase B run finished. Log dir: %s", log_dir)

    if getattr(args, "save_checkpoints", False):
        try:
            from src.phase_b.eval_phase_b_ucf import run_phase_b_ucf_eval

            logger.info("Running Phase B val+test (same protocol as get_evaluation.test, one model for A/B).")
            run_phase_b_ucf_eval(log_dir, root_dir=args.root_dir, device=str(args.device))
        except Exception as exc:
            logger.warning("Phase B post-train eval failed: %s", exc)
    return log_dir


if __name__ == "__main__":
    run_phase_b_ucf_audio()
