"""
Extract T=16 uniform CLIP ViT-B/32 image features per UCF-101 video.

Output layout (per-class pickles, compatible with ``read_features`` in src.utils):
  {output_root}/video/{split}/{ClassName}.pkl
  each file: {'features': list of ndarray [T, 512], 'video_names': ..., 'fps': ...}

Audio is not written here; use cls_features_non_averaged audio pkls when building
processed splits (see splitting_scripts_cls/create_pkl_files_cls_temporal16.py).
"""
import argparse
import pickle
import sys
from pathlib import Path

import clip
import cv2
import numpy as np
import torch
import torchvision
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.args import str_to_bool
from src.feature_constants import TEMPORAL_CLIP_NUM_FRAMES


def uniform_frame_indices(num_frames: int, num_samples: int) -> np.ndarray:
    if num_frames <= 0:
        return np.zeros(num_samples, dtype=np.int64)
    if num_frames == 1:
        return np.zeros(num_samples, dtype=np.int64)
    return np.linspace(0, num_frames - 1, num_samples).round().astype(np.int64)


def load_video_frames(video_path: Path) -> torch.Tensor:
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video = torch.zeros((max(frame_count, 1), frame_h, frame_w, 3), dtype=torch.uint8)
    fc = 0
    while fc < frame_count:
        ret, image = cap.read()
        if not ret:
            break
        video[fc] = torch.from_numpy(image)
        fc += 1
    cap.release()
    return video[:fc]


def encode_video_clip(
    model,
    preprocess,
    device,
    video_tensor: torch.Tensor,
    num_frames: int,
    to_pil,
) -> np.ndarray:
    indices = uniform_frame_indices(int(video_tensor.shape[0]), num_frames)
    feats = []
    with torch.no_grad():
        for idx in indices:
            frame = video_tensor[int(idx)].permute(2, 0, 1)
            frame = to_pil(frame)
            batch = preprocess(frame).unsqueeze(0).to(device)
            emb = model.encode_image(batch)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            feats.append(emb.squeeze(0).cpu().numpy().astype(np.float32))
    return np.stack(feats, axis=0)


def main():
    parser = argparse.ArgumentParser(description="UCF-101 temporal CLIP feature extraction (T=16).")
    parser.add_argument(
        "--video_root",
        type=str,
        default="/home/ubuntu/data/UCF/raw/UCF-101",
        help="Root containing **/*.avi class folders.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="features/cls_features_temporal_16",
        help="Dataset root subfolder for temporal video features.",
    )
    parser.add_argument("--num_frames", type=int, default=TEMPORAL_CLIP_NUM_FRAMES)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--finetuned_clip_path",
        type=str,
        default="",
        help="Optional path to finetuned CLIP state_dict checkpoint.",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=500,
        help="Save intermediate master list every N videos.",
    )
    parser.add_argument(
        "--master_list_path",
        type=str,
        default="",
        help="Optional path to save master list [feat[T,512], class_id, name_file] for create_pkl.",
    )
    parser.add_argument("--finetuned_model", type=str_to_bool, default=False)
    args = parser.parse_args()

    video_root = Path(args.video_root)
    output_root = Path(args.output_root)
    device = args.device
    num_frames = int(args.num_frames)

    model, preprocess = clip.load("ViT-B/32", device=device)
    if args.finetuned_model and args.finetuned_clip_path:
        ckpt = torch.load(args.finetuned_clip_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
    model.eval()
    to_pil = torchvision.transforms.Compose([torchvision.transforms.ToPILImage()])

    list_classes = sorted({p.parent.name for p in video_root.glob("**/*.avi")})
    dict_classes_ids = {name: i for i, name in enumerate(list_classes)}

    avi_files = sorted(video_root.glob("**/*.avi"))
    master_list = []
    master_path = Path(args.master_list_path) if args.master_list_path else None

    for i, f in enumerate(tqdm(avi_files, desc="UCF temporal CLIP")):
        try:
            video = load_video_frames(f)
            feat = encode_video_clip(model, preprocess, device, video, num_frames, to_pil)
        except Exception as exc:
            print(f"skip {f}: {exc}")
            continue

        class_name = f.parent.name
        class_id = dict_classes_ids[class_name]
        name_file = f.name
        master_list.append([feat, class_id, name_file])

        if master_path and args.checkpoint_every > 0 and (i + 1) % args.checkpoint_every == 0:
            with master_path.open("wb") as handle:
                pickle.dump(master_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if master_path:
        with master_path.open("wb") as handle:
            pickle.dump(master_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Wrote master list ({len(master_list)} videos) to {master_path}")

    print(f"Done. Extracted {len(master_list)} videos. Run create_pkl_files_cls_temporal16.py next.")


if __name__ == "__main__":
    main()
