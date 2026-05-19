"""Minimal smoke tests for teacher SNN gate (Step 1 / Step 2)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.clipclap_model import ClipClap_model
from src.utils_improvements import get_model_params


def _gate_params(**teacher_kw):
    base = dict(
        lr=7e-5,
        reg_loss=False,
        dropout_encoder=0.1,
        dropout_decoder=0.1,
        additional_dropout=0.0,
        encoder_hidden_size=512,
        decoder_hidden_size=512,
        embeddings_batch_norm=True,
        rec_loss=False,
        cross_entropy_loss=True,
        transformer_use_embedding_net=False,
        transformer_dim=512,
        transformer_depth=1,
        transformer_heads=4,
        transformer_dim_head=32,
        transformer_mlp_dim=128,
        transformer_dropout=0.1,
        transformer_embedding_dim=64,
        transformer_embedding_time_len=60,
        transformer_embedding_dropout=0.1,
        transformer_embedding_time_embed_type="learnable",
        transformer_embedding_fourier_scale=10.0,
        transformer_embedding_embed_augment_position=False,
        lr_scheduler=False,
        optimizer="adam",
        use_self_attention=False,
        use_cross_attention=False,
        transformer_average_features=False,
        audio_only=False,
        video_only=False,
        transformer_use_class_token=False,
        transformer_embedding_modality="both",
        modality="both",
        word_embeddings="both",
        model_backend="ann",
        use_teacher_parallel_snn=False,
        teacher_use_snn_gate=False,
        teacher_gate_mode="none",
        teacher_sparsity_mode="none",
        teacher_sparse_lambda=0.0,
        feature_extraction_method="cls_features_static_temporal_16",
    )
    base.update(teacher_kw)
    return get_model_params(**base)


def _fake_batch(b=4, t=16):
    audio = torch.randn(b, 1024)
    video = torch.randn(b, t, 512)
    video_static = torch.randn(b, 512)
    text = torch.randn(b, 1536)
    masks = {"audio": torch.ones(b, 1), "video": torch.ones(b, t)}
    timesteps = {
        "audio": torch.zeros(b, 1),
        "video": torch.arange(t).float().unsqueeze(0).repeat(b, 1),
    }
    return audio, video, text, masks, timesteps, video_static


def test_ann_only_equivalence():
    model = ClipClap_model(_gate_params(), 1024, 512)
    model.eval()
    audio, video, text, masks, timesteps, video_static = _fake_batch()
    out = model.forward(
        audio, video, text, masks, timesteps, epoch=0, video_static=video_static
    )
    assert not out["teacher_gate_active"]
    assert out["teacher_z_fused"] is None
    emb_ann, _, _ = model.get_embeddings(
        audio, video, text, masks, timesteps, video_static=video_static
    )
    assert torch.allclose(emb_ann, out["theta_o"], atol=1e-5)
    print("OK ann_only_equivalence")


def test_residual_gate():
    model = ClipClap_model(
        _gate_params(
            use_teacher_parallel_snn=True,
            teacher_use_snn_gate=True,
            teacher_gate_mode="residual",
            teacher_snn_arch="temporal_video",
            teacher_snn_gamma=0.1,
        ),
        1024,
        512,
    )
    model.eval()
    audio, video, text, masks, timesteps, video_static = _fake_batch()
    out = model.forward(
        audio, video, text, masks, timesteps, epoch=0, video_static=video_static
    )
    assert out["teacher_gate_active"]
    assert out["teacher_z_fused"].shape == out["theta_o"].shape
    assert out["teacher_refine_l2_rel"] is not None
    _, details = model.compute_loss(
        out, None, torch.zeros(4, dtype=torch.long), epoch=0
    )
    assert "Diag/gate_mean" in details
    assert "Diag/refine_l2_rel" in details
    assert not torch.allclose(out["teacher_z_fused"], out["theta_o"], atol=1e-7)
    print("OK residual_gate refine_l2_rel=", float(out["teacher_refine_l2_rel"]))


def test_direct_gate():
    model = ClipClap_model(
        _gate_params(
            use_teacher_parallel_snn=True,
            teacher_use_snn_gate=True,
            teacher_gate_mode="direct",
            teacher_snn_arch="temporal_video",
        ),
        1024,
        512,
    )
    model.eval()
    audio, video, text, masks, timesteps, video_static = _fake_batch()
    out = model.forward(
        audio, video, text, masks, timesteps, epoch=0, video_static=video_static
    )
    assert out["teacher_gate_active"]
    print("OK direct_gate")


if __name__ == "__main__":
    test_ann_only_equivalence()
    test_residual_gate()
    test_direct_gate()
    print("All smoke tests passed.")
