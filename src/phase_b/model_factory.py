"""Shared construction of ClipClap inner + Phase B audio frontend wrapper."""

from __future__ import annotations

import torch.nn as nn
import torch.optim as optim

from src.clipclap_model import build_clipclap_model, init_snn_clipclap_from_ann_checkpoint
from src.utils_improvements import get_model_params

from src.phase_b.audio_frontend import PhaseBAudioEncoderSNN
from src.phase_b.phase_b_audio_model import ClipClapPhaseB_AudioWrapper
from src.phase_b.ucf_phase_b_audio_dataset import PhaseBAudioSource


def get_model_params_from_args(args):
    return get_model_params(
        args.lr,
        args.reg_loss,
        args.embedding_dropout,
        args.decoder_dropout,
        args.additional_dropout,
        args.embeddings_hidden_size,
        args.decoder_hidden_size,
        args.embeddings_batch_norm,
        args.rec_loss,
        args.cross_entropy_loss,
        args.transformer_use_embedding_net,
        args.transformer_dim,
        args.transformer_depth,
        args.transformer_heads,
        args.transformer_dim_head,
        args.transformer_mlp_dim,
        args.transformer_dropout,
        args.transformer_embedding_dim,
        args.transformer_embedding_time_len,
        args.transformer_embedding_dropout,
        args.transformer_embedding_time_embed_type,
        args.transformer_embedding_fourier_scale,
        args.transformer_embedding_embed_augment_position,
        args.lr_scheduler,
        args.optimizer,
        args.use_self_attention,
        args.use_cross_attention,
        args.transformer_average_features,
        args.audio_only,
        args.video_only,
        args.transformer_use_class_token,
        args.transformer_embedding_modality,
        args.modality,
        args.word_embeddings,
        getattr(args, "model_backend", "ann"),
        getattr(args, "snn_num_steps", 10),
        getattr(args, "snn_beta", 0.9),
        getattr(args, "snn_threshold", 1.0),
    )


def build_clipclap_phase_b_wrapped(args, device: str | None = None) -> nn.Module:
    """
    Build ``ClipClap_model`` (audio modality) + ``PhaseBAudioEncoderSNN`` wrapper.
    Optionally init SNN head from ``args.snn_init_ann_path``.
    """
    dev = device or getattr(args, "device", "cpu")
    model_params = get_model_params_from_args(args)
    inner = build_clipclap_model(
        model_params,
        input_size_audio=args.input_size_audio,
        input_size_video=args.input_size_video,
    ).to(dev)

    if getattr(args, "model_backend", "ann") == "snn" and getattr(args, "snn_init_ann_path", None):
        init_snn_clipclap_from_ann_checkpoint(
            inner,
            args.snn_init_ann_path,
            dev,
            model_params,
            args.input_size_audio,
            args.input_size_video,
        )

    fe = PhaseBAudioEncoderSNN(
        mel_flat_dim=4096,
        out_dim=1024,
        num_steps=min(32, getattr(args, "snn_num_steps", 10)),
        beta=args.snn_beta,
        threshold=args.snn_threshold,
        collect_frontend_diag=bool(getattr(args, "phase_b_frontend_diag", False)),
        diag_log_interval=int(getattr(args, "phase_b_diag_log_interval", 50)),
    ).to(dev)
    wrapped = ClipClapPhaseB_AudioWrapper(
        inner,
        fe,
        audio_input_scale=float(getattr(args, "phase_b_audio_input_scale", 1.0)),
    )
    # Inner ClipClap_model builds Adam on self.parameters() only; the wrapper's frontend
    # was never optimized, so dummy vs mel could yield similarly useless heads. Train both.
    if not getattr(inner, "is_sam_optim", False):
        inner.optimizer_gen = optim.Adam(
            list(inner.parameters()) + list(fe.parameters()),
            lr=inner.lr,
            weight_decay=1e-5,
        )
        if getattr(inner, "lr_scheduler", False):
            inner.scheduler_learning_rate = optim.lr_scheduler.ReduceLROnPlateau(
                inner.optimizer_gen, "max", patience=3, verbose=True
            )
    return wrapped


def resolve_phase_b_audio_source(args) -> PhaseBAudioSource:
    raw = getattr(args, "phase_b_audio_source", "offline_as_mel")
    try:
        return PhaseBAudioSource(raw)
    except ValueError:
        return PhaseBAudioSource.OFFLINE_AS_MEL
