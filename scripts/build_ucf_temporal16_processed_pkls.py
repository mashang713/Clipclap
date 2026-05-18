"""
Build _features_processed/cls_features_temporal_16/{training,val,test,train_val}cls_split.pkl.

Requires:
  - features/cls_features_temporal_16/video/{split}/*.pkl
  - features/cls_features_non_averaged/audio/{split}/*.pkl
  - text embeddings under temporal or non_averaged text/ (see UCFDataset fallback)

Usage:
  python scripts/build_ucf_temporal16_processed_pkls.py --root_dir /path/to/ucf_root
"""
import argparse
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import UCFDataset
from src.feature_constants import TEMPORAL_CLIP_FEATURE_METHOD


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--zero_shot_split", type=str, default="cls_split")
    parser.add_argument("--device", type=str, default="cpu")
    args_ns = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    args = SimpleNamespace(
        root_dir=Path(args_ns.root_dir),
        dataset_name="UCF",
        feature_extraction_method=TEMPORAL_CLIP_FEATURE_METHOD,
        zero_shot_split=args_ns.zero_shot_split,
        device=args_ns.device,
        use_wavcaps_embeddings=True,
    )

    for split in ("train", "val", "train_val", "test"):
        UCFDataset(args=args, dataset_split=split, zero_shot_mode=None)
        logging.info("Built processed pkl for split=%s", split)

    out = Path(args_ns.root_dir) / "_features_processed" / TEMPORAL_CLIP_FEATURE_METHOD
    print(f"Done. Processed pkls under {out.resolve()}")


if __name__ == "__main__":
    main()
