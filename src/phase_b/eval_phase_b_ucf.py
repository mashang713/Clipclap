"""
Standalone GZSL evaluation for a Phase B audio run (wrapper + Phase B contrastive loaders).
Does not use get_evaluation.py (two-stage, no frontend).
"""

from __future__ import annotations

import logging
from pathlib import Path

from torch.utils import data

from src.dataset import UCFDataset, DefaultCollator
from src.test import test
from src.utils import load_args, load_model_weights, setup_evaluation

from src.phase_b.model_factory import build_clipclap_phase_b_wrapped
from src.phase_b.ucf_phase_b_audio_dataset import ContrastivePhaseBAudio, PhaseBAudioSource


def _pick_checkpoint(run_dir: Path) -> Path:
    """Best ``score`` ckpt lives next to train.log; per-epoch files under ``checkpoints/``."""
    run_dir = Path(run_dir)
    for name in (
        "ClipClapPhaseB_AudioWrapper_score.pt",
        "ClipClap_model_score.pt",
    ):
        p = run_dir / name
        if p.is_file():
            return p
    ckpt_dir = run_dir / "checkpoints"
    cpts = sorted(ckpt_dir.glob("*_score_ckpt_*.pt"))
    if not cpts:
        raise FileNotFoundError(
            f"No Phase B weights in {run_dir} (expected *score.pt or checkpoints/*_score_ckpt_*.pt)"
        )
    return cpts[-1]


def run_phase_b_ucf_eval(
    run_dir: Path,
    root_dir: Path | None = None,
    device: str | None = None,
    ckpt_path: Path | None = None,
):
    run_dir = Path(run_dir)
    config = load_args(run_dir)
    if root_dir is not None:
        config.root_dir = Path(root_dir)
    if device is not None:
        config.device = device

    if getattr(config, "modality", None) != "audio":
        logging.warning("Phase B eval expects modality=audio; got %s", getattr(config, "modality", None))

    if config.input_size is not None:
        config.input_size_audio = config.input_size
        config.input_size_video = config.input_size

    ckpt = Path(ckpt_path) if ckpt_path else _pick_checkpoint(run_dir)
    logging.info("Loading weights from %s", ckpt)

    class _EvalArgs:
        """Minimal object for setup_evaluation + test."""

        pass

    eval_stub = _EvalArgs()
    eval_stub.load_path_stage_B = run_dir
    eval_stub.root_dir = config.root_dir
    eval_stub.input_size = getattr(config, "input_size", None)
    eval_stub.batch_seqlen_test = config.batch_seqlen_test
    eval_stub.batch_seqlen_test_maxlen = config.batch_seqlen_test_maxlen
    eval_stub.batch_seqlen_test_trim = config.batch_seqlen_test_trim
    eval_stub.eval_bs = config.eval_bs
    eval_stub.eval_num_workers = config.eval_num_workers
    eval_stub.eval_name = config.eval_name
    eval_stub.eval_save_performances = getattr(config, "eval_save_performances", False)
    eval_stub.dataset_name = config.dataset_name

    logger, eval_dir, test_stats, tb_writer = setup_evaluation(eval_stub, config.__dict__.keys())

    val_all_dataset = UCFDataset(args=config, dataset_split="val", zero_shot_mode=None)
    test_dataset = UCFDataset(args=config, dataset_split="test", zero_shot_mode=None)

    src = PhaseBAudioSource(getattr(config, "phase_b_audio_source", "offline_as_mel"))
    contrastive_val = ContrastivePhaseBAudio(val_all_dataset, source=src, mel_flat_dim=4096)
    contrastive_test = ContrastivePhaseBAudio(test_dataset, source=src, mel_flat_dim=4096)

    if config.selavi:
        collator_test = DefaultCollator(
            mode=config.batch_seqlen_test,
            max_len=config.batch_seqlen_test_maxlen,
            trim=config.batch_seqlen_test_trim,
            rate_video=1,
            rate_audio=1,
        )
    else:
        collator_test = DefaultCollator(
            mode=config.batch_seqlen_test,
            max_len=config.batch_seqlen_test_maxlen,
            trim=config.batch_seqlen_test_trim,
        )

    final_val_loader = data.DataLoader(
        dataset=contrastive_val,
        collate_fn=collator_test,
        batch_size=config.eval_bs,
        num_workers=config.eval_num_workers,
    )
    final_test_loader = data.DataLoader(
        dataset=contrastive_test,
        collate_fn=collator_test,
        batch_size=config.eval_bs,
        num_workers=config.eval_num_workers,
    )

    model = build_clipclap_phase_b_wrapped(config, device=config.device)
    epoch = load_model_weights(ckpt, model)
    logger.info("Checkpoint epoch field: %s", epoch)

    distance_fn = config.distance_fn

    model.eval()
    results = test(
        eval_name=config.eval_name,
        val_dataset=(val_all_dataset, final_val_loader),
        test_dataset=(test_dataset, final_test_loader),
        model_A=model,
        model_B=model,
        device=config.device,
        distance_fn=distance_fn,
        test_stats=test_stats,
        eval_dir=eval_dir,
        new_model_sequence=config.new_model_sequence,
        args=config,
        save_performances=eval_stub.eval_save_performances,
    )
    logger.info("Phase B eval finished. Results keys: %s", results.keys())
    del tb_writer
    return results
