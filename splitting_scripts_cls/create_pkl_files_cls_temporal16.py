"""
Build per-class video pickles under features/cls_features_temporal_16/video/...

Video features come from a temporal master list ([T,512] per video).
Audio is not rebuilt here; training uses cls_features_non_averaged/audio/... via UCFDataset.

Example:
  python splitting_scripts_cls/create_pkl_files_cls_temporal16.py \\
    --path_temporal_master /path/to/ucf_temporal_clip16.pkl \\
    --path_splitted_dataset /path/to/dataset_root
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from splitting_scripts_cls.create_pkl_files_cls import create_dict_features

FEATURE_FOLDER = "cls_features_temporal_16"


def save_temporal_video_pkls(which_dataset, path_temporal_master, path_splitted_dataset_root):
    dictionary_pkl, list_h5 = create_dict_features(
        which_dataset=which_dataset,
        original_dataset_path=path_temporal_master,
        use_audio=False,
    )
    save_root_path = path_splitted_dataset_root
    number_of_videos = 0

    for class_path, videos in tqdm(list_h5.items(), desc="temporal16 video pkls"):
        list_embeddings_videos = []
        list_fps = []
        number_of_videos += len(videos)
        for video_item in videos:
            key = video_item.decode("utf-8")
            element = dictionary_pkl[key]
            embedding_video = element[0]
            if getattr(embedding_video, "size", 0) == 0:
                continue
            list_embeddings_videos.append(embedding_video)
            list_fps.append(25)

        concatenated_path = save_root_path + (
            class_path.decode("utf-8") if isinstance(class_path, bytes) else class_path
        )
        new_concatenated_list = {
            "features": list_embeddings_videos,
            "video_names": videos,
            "fps": list_fps,
        }

        split_path = concatenated_path.split("/")[1:]
        new_path = ""
        for element in split_path[:-1]:
            if element is split_path[-4]:
                element = FEATURE_FOLDER
            new_path += "/" + element
        try:
            os.makedirs(new_path)
        except OSError:
            pass
        concatenated_path = new_path + "/" + split_path[-1].rsplit(".", 1)[0]

        with open(concatenated_path + ".pkl", "wb") as f:
            pickle.dump(new_concatenated_list, f)

    print("total_number_of_videos", number_of_videos)


def main():
    parser = argparse.ArgumentParser(description="Create cls_features_temporal_16 video pkls for UCF.")
    parser.add_argument("--dataset_name", type=str, default="UCF", choices=["UCF"])
    parser.add_argument(
        "--path_temporal_master",
        type=str,
        required=True,
        help="Master pickle: list of [video_feat[T,512], class_id, name_file].",
    )
    parser.add_argument(
        "--path_splitted_dataset",
        type=str,
        required=True,
        help="Dataset root (same layout as create_pkl_files_cls path_splitted_dataset).",
    )
    args = parser.parse_args()
    save_temporal_video_pkls(args.dataset_name, args.path_temporal_master, args.path_splitted_dataset)


if __name__ == "__main__":
    main()
