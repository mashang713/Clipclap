"""
Merge UCF processed pkls: static video (non_averaged) + temporal video (temporal_16).

Reads:
  {root}/_features_processed/cls_features_non_averaged/{split}cls_split.pkl
  {root}/_features_processed/cls_features_temporal_16/{split}cls_split.pkl

Writes:
  {root}/_features_processed/cls_features_static_temporal_16/{split}cls_split.pkl

Each sample keeps audio/text from static; video = temporal [T,512]; video_static = static [512].
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.feature_constants import (
    BASELINE_AUDIO_FEATURE_METHOD,
    STATIC_TEMPORAL_CLIP_FEATURE_METHOD,
    TEMPORAL_CLIP_FEATURE_METHOD,
)


def _urls_equal(u1, u2):
    return str(u1) == str(u2)


def _assert_split_aligned(static_data, temporal_data, split_name: str):
    sa, sv, st = static_data["audio"], static_data["video"], static_data["text"]
    ta, tv, tt = temporal_data["audio"], temporal_data["video"], temporal_data["text"]

    n = len(sa["data"])
    if len(ta["data"]) != n:
        raise ValueError(f"{split_name}: audio length mismatch static={n} temporal={len(ta['data'])}")

    for i in range(n):
        if not _urls_equal(sa["url"][i], ta["url"][i]):
            raise ValueError(
                f"{split_name}: url mismatch at {i}: "
                f"static={sa['url'][i]!r} temporal={ta['url'][i]!r}"
            )
        if int(sa["target"][i]) != int(ta["target"][i]):
            raise ValueError(
                f"{split_name}: target mismatch at {i} for url={sa['url'][i]!r}"
            )
        if int(sv["target"][i]) != int(tv["target"][i]):
            raise ValueError(f"{split_name}: video target mismatch at {i}")

    if len(st["data"]) != len(tt["data"]):
        raise ValueError(f"{split_name}: text class count mismatch")


def merge_split(static_data, temporal_data, split_name: str):
    _assert_split_aligned(static_data, temporal_data, split_name)
    sa, sv = static_data["audio"], static_data["video"]
    tv = temporal_data["video"]

    static_video = []
    temporal_video = []
    for i in range(len(sa["data"])):
        vs = np.asarray(sv["data"][i], dtype=np.float32)
        vt = np.asarray(tv["data"][i], dtype=np.float32)
        if vs.ndim != 1 or vs.shape[0] != 512:
            raise ValueError(
                f"{split_name}: sample {i} video_static expected [512], got {vs.shape}"
            )
        if vt.ndim != 2 or vt.shape[1] != 512:
            raise ValueError(
                f"{split_name}: sample {i} temporal video expected [T,512], got {vt.shape}"
            )
        static_video.append(vs)
        temporal_video.append(vt)

    merged = {
        "audio": static_data["audio"],
        "video": {
            "data": temporal_video,
            "target": tv["target"],
            "url": tv["url"],
            "fps": tv.get("fps", sa.get("fps")),
        },
        "video_static": {
            "data": static_video,
            "target": sv["target"],
            "url": sv["url"],
            "fps": sv.get("fps", sa.get("fps")),
        },
        "text": static_data["text"],
    }
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/home/ubuntu/data/UCF",
        help="UCF dataset root containing _features_processed/",
    )
    parser.add_argument("--zero_shot_split", type=str, default="cls_split")
    parser.add_argument(
        "--static_method",
        type=str,
        default=BASELINE_AUDIO_FEATURE_METHOD,
    )
    parser.add_argument(
        "--temporal_method",
        type=str,
        default=TEMPORAL_CLIP_FEATURE_METHOD,
    )
    parser.add_argument(
        "--output_method",
        type=str,
        default=STATIC_TEMPORAL_CLIP_FEATURE_METHOD,
    )
    args = parser.parse_args()

    root = Path(args.root_dir)
    proc = root / "_features_processed"
    splits = ("training", "val", "test", "train_val")
    suffix = f"{args.zero_shot_split}.pkl"

    out_dir = proc / args.output_method
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        static_path = proc / args.static_method / f"{split}{suffix}"
        temporal_path = proc / args.temporal_method / f"{split}{suffix}"
        out_path = out_dir / f"{split}{suffix}"

        if not static_path.is_file():
            raise FileNotFoundError(static_path)
        if not temporal_path.is_file():
            raise FileNotFoundError(temporal_path)

        with static_path.open("rb") as f:
            static_data = pickle.load(f)
        with temporal_path.open("rb") as f:
            temporal_data = pickle.load(f)

        merged = merge_split(static_data, temporal_data, split)
        with out_path.open("wb") as f:
            pickle.dump(merged, f, pickle.HIGHEST_PROTOCOL)
        print(f"Wrote {out_path} ({len(merged['audio']['data'])} samples)")

    print(f"Done. Output under {out_dir.resolve()}")


if __name__ == "__main__":
    main()
