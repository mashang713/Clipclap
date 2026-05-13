#!/usr/bin/env python3
"""
Temporary diagnostic: print tensor/array shapes from UCF ContrastiveDataset batches
(use the same config path as training, e.g. config/clipclap.yaml).

Does not modify training code or run the full training loop.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils import data

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.args import args_main  # noqa: E402
from src.dataset import ContrastiveDataset, DefaultCollator, UCFDataset  # noqa: E402
from src.sampler import SamplerFactory  # noqa: E402


def _describe(name: str, obj, depth: int = 0) -> None:
    ind = "  " * depth
    if torch.is_tensor(obj):
        print(f"{ind}{name}: torch.Tensor  shape={tuple(obj.shape)}  dtype={obj.dtype}")
        return
    if isinstance(obj, np.ndarray):
        print(f"{ind}{name}: numpy.ndarray  shape={obj.shape}  dtype={obj.dtype}")
        return
    if isinstance(obj, dict):
        print(f"{ind}{name}: dict  keys={list(obj.keys())}")
        for k, v in obj.items():
            _describe(str(k), v, depth + 1)
        return
    if isinstance(obj, (list, tuple)):
        print(f"{ind}{name}: {type(obj).__name__}  len={len(obj)}")
        return
    print(f"{ind}{name}: {type(obj).__name__}  repr={repr(obj)[:200]}")


def _inspect_split(tag: str, batch) -> None:
    data_b, target_b = batch
    print(f"\n========== {tag} ==========")
    print("-- data['positive'] --")
    p = data_b["positive"]
    for key in ("audio", "video", "text", "url", "audio_mask", "video_mask", "timestep", "fps"):
        if key not in p:
            continue
        _describe(f"positive['{key}']", p[key])

    print("\n-- data['negative'] (subset) --")
    n = data_b["negative"]
    for key in ("audio", "video", "text"):
        _describe(f"negative['{key}']", n[key])

    print("\n-- target --")
    _describe("target['positive']", target_b["positive"])
    _describe("target['negative']", target_b["negative"])

    print("\n-- highlights (positive only) --")
    _describe("audio", p["audio"])
    _describe("video", p["video"])
    _describe("text", p["text"])
    _describe("target (positive class ids)", target_b["positive"])
    if "audio_mask" in p and "video_mask" in p:
        print("masks['positive']:")
        _describe("  audio_mask", p["audio_mask"])
        _describe("  video_mask", p["video_mask"])
    if "timestep" in p:
        print("timesteps['positive']:")
        _describe("  timestep", p["timestep"])


def main() -> None:
    args, _ = args_main()
    if args.dataset_name != "UCF":
        print(
            f"Note: dataset_name={args.dataset_name!r} (this script is tailored to UCF + clipclap.yaml).",
            flush=True,
        )

    if getattr(args, "root_dir", None) is None:
        print(
            "Error: root_dir is not set. Pass e.g. --root_dir /path/to/ucf_benchmark",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(2)

    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger("clipclap_inspect")

    args.retrain_all = False

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

    contrastive_train = ContrastiveDataset(train_dataset)
    contrastive_val = ContrastiveDataset(val_all_dataset)

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

    train_sampler = SamplerFactory(logger).get(
        class_idxs=list(contrastive_train.target_to_indices.values()),
        batch_size=args.bs,
        n_batches=max(1, int(args.n_batches)),
        alpha=1,
        kind="random",
    )
    val_sampler = SamplerFactory(logger).get(
        class_idxs=list(contrastive_val.target_to_indices.values()),
        batch_size=args.bs,
        n_batches=max(1, int(args.n_batches)),
        alpha=1,
        kind="random",
    )

    train_loader = data.DataLoader(
        dataset=contrastive_train,
        batch_sampler=train_sampler,
        collate_fn=collator_train,
        num_workers=0,
    )
    val_loader = data.DataLoader(
        dataset=contrastive_val,
        batch_sampler=val_sampler,
        collate_fn=collator_test,
        num_workers=0,
    )

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    _inspect_split("TRAIN (first batch)", train_batch)
    _inspect_split("VAL (first batch)", val_batch)


if __name__ == "__main__":
    main()
