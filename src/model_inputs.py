"""Helpers to pack dataloader batches for ClipClap_model."""
import torch


def get_positive_video_static(batch_positive, device):
    vs = batch_positive.get("video_static")
    if vs is None:
        return None
    return vs.to(device)


def maybe_zscore(tensor, enabled):
    if not enabled or tensor is None:
        return tensor
    return (tensor - torch.mean(tensor)) / torch.sqrt(torch.var(tensor))
