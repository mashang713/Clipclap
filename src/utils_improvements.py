#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import glob
import shutil
import socket
import random
import itertools
import numpy as np
import multiprocessing
import configparser as cp
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score

import torch

np.random.seed(0)

def get_model_params(
    lr,
    reg_loss,
    dropout_encoder,
    dropout_decoder,
    additional_dropout,
    encoder_hidden_size,
    decoder_hidden_size,
    embeddings_batch_norm,
    rec_loss,
    cross_entropy_loss,
    transformer_use_embedding_net,
    transformer_dim,
    transformer_depth,
    transformer_heads,
    transformer_dim_head,
    transformer_mlp_dim,
    transformer_dropout,
    transformer_embedding_dim,
    transformer_embedding_time_len,
    transformer_embedding_dropout,
    transformer_embedding_time_embed_type,
    transformer_embedding_fourier_scale,
    transformer_embedding_embed_augment_position,
    lr_scheduler,
    optimizer,
    use_self_attention,
    use_cross_attention,
    transformer_average_features,
    audio_only,
    video_only,
    transformer_use_class_token,
    transformer_embedding_modality,
    modality,
    word_embeddings,
    model_backend="ann",
    snn_num_steps=10,
    snn_beta=0.9,
    snn_threshold=1.0,
    # Prototype-preserving fake-SNN conversion (ANN teacher detached)
    use_snn_conversion=False,
    snn_timesteps=4,
    lambda_proto=0.5,
    lambda_feat=0.1,
    proto_temperature=2.0,
    snn_conv_threshold_percentile=0.99,
    proto_kd_type="kl_all",
    proto_topk=10,
    proto_warmup_epochs=0,
    proto_conf_margin=0.0,
    debug_print_shapes=False,
    use_teacher_parallel_snn=False,
    teacher_use_snn_gate=False,
    teacher_gate_mode="none",
    teacher_sparsity_mode="none",
    teacher_sparse_lambda=0.0,
    teacher_snn_timesteps=4,
    teacher_snn_gamma=0.1,
    teacher_snn_alpha=0.1,
    teacher_snn_beta=1.0,
    teacher_snn_hidden_dim=512,
    teacher_snn_threshold=1.0,
    teacher_snn_decay=0.9,
    teacher_snn_dropout=0.1,
    teacher_snn_fusion_mode="add",
    teacher_sigmoid_centered=True,
    teacher_ann_gate_snn=False,
    teacher_gate_strength=0.5,
    teacher_leak_strength=0.2,
    teacher_spike_scale_strength=0.2,
    teacher_fire_rate_target=0.1,
    teacher_fire_rate_reg=0.0,
    teacher_snn_arch="backend_only",
    teacher_frontend_snn_timesteps=4,
    teacher_frontend_snn_hidden_dim=512,
    teacher_frontend_snn_decay=0.9,
    teacher_frontend_snn_threshold=1.0,
    teacher_frontend_snn_dropout=0.1,
    teacher_frontend_fire_rate_target=0.1,
    teacher_frontend_fire_rate_reg=0.0,
    teacher_eval_repr="fused",
    teacher_fusion_warmup_epochs=0,
    teacher_gamma_warmup=True,
    teacher_init_ann_path=None,
    teacher_freeze_ann=False,
    teacher_freeze_text_branch=False,
    use_snn_sigmoid_ann_gate=False,
    snn_sigmoid_ann_warmup_steps=500,
    snn_sigmoid_ann_ramp_steps=1500,
    snn_sigmoid_ann_lambda=0.3,
    snn_sigmoid_ann_detach_gate=True,
    teacher_snn_sigmoid_ann_snn_ce=False,
    teacher_snn_sigmoid_ann_snn_ce_weight=1.0,
    teacher_snn_only=False,
    teacher_snn_use_video_delta=True,
    teacher_snn_time_weight="linear",
    teacher_snn_proto_align_lambda=0.0,
    teacher_snn_proto_align_type="cosine",
    feature_extraction_method="",
    ):

    params_model = dict()
    # Dimensions
    params_model['dim_out'] = 64
    params_model['cross_entropy_loss']=cross_entropy_loss

    # Optimizers' parameters
    params_model['lr'] = lr
    params_model['optimizer'] = optimizer
    if encoder_hidden_size==0:
        encoder_hidden_size=None
    if decoder_hidden_size==0:
        decoder_hidden_size=None



    params_model['additional_dropout']=additional_dropout
    params_model['reg_loss']=reg_loss
    params_model['dropout_encoder']=dropout_encoder
    params_model['dropout_decoder']=dropout_decoder
    params_model['encoder_hidden_size']=encoder_hidden_size
    params_model['decoder_hidden_size']=decoder_hidden_size

    # Model Sequence
    params_model['embeddings_batch_norm'] = embeddings_batch_norm
    params_model['rec_loss'] = rec_loss
    params_model['transformer_average_features'] = transformer_average_features
    params_model['transformer_use_embedding_net'] = transformer_use_embedding_net
    params_model['transformer_dim'] = transformer_dim
    params_model['transformer_depth'] = transformer_depth
    params_model['transformer_heads'] = transformer_heads
    params_model['transformer_dim_head'] = transformer_dim_head
    params_model['transformer_mlp_dim'] = transformer_mlp_dim
    params_model['transformer_dropout'] = transformer_dropout
    params_model['transformer_embedding_dim'] = transformer_embedding_dim
    params_model['transformer_embedding_time_len'] = transformer_embedding_time_len
    params_model['transformer_embedding_dropout'] = transformer_embedding_dropout
    params_model['transformer_embedding_time_embed_type'] = transformer_embedding_time_embed_type
    params_model['transformer_embedding_fourier_scale'] = transformer_embedding_fourier_scale
    params_model['transformer_embedding_embed_augment_position'] = transformer_embedding_embed_augment_position
    params_model['transformer_embedding_modality'] = transformer_embedding_modality
    params_model['transformer_attention_use_self_attention']=use_self_attention
    params_model['transformer_attention_use_cross_attention']=use_cross_attention
    params_model['audio_only'] = audio_only
    params_model['video_only'] = video_only
    params_model['transformer_use_class_token'] = transformer_use_class_token

    params_model['lr_scheduler'] = lr_scheduler


    params_model['modality'] = modality
    params_model['word_embeddings'] = word_embeddings
    params_model['model_backend'] = model_backend
    if model_backend == "snn":
        params_model["snn_embedding_kwargs"] = {
            "num_steps": snn_num_steps,
            "beta": snn_beta,
            "threshold": snn_threshold,
        }
    else:
        params_model["snn_embedding_kwargs"] = {}

    # Fake-SNN conversion + prototype-preserving losses (kept independent from true SNN backend).
    params_model["use_snn_conversion"] = bool(use_snn_conversion)
    params_model["snn_timesteps"] = int(snn_timesteps)
    params_model["lambda_proto"] = float(lambda_proto)
    params_model["lambda_feat"] = float(lambda_feat)
    params_model["proto_temperature"] = float(proto_temperature)
    params_model["snn_conv_threshold_percentile"] = float(snn_conv_threshold_percentile)
    params_model["proto_kd_type"] = str(proto_kd_type)
    params_model["proto_topk"] = int(proto_topk)
    params_model["proto_warmup_epochs"] = int(proto_warmup_epochs)
    params_model["proto_conf_margin"] = float(proto_conf_margin)
    params_model["debug_print_shapes"] = bool(debug_print_shapes)
    params_model["use_teacher_parallel_snn"] = bool(use_teacher_parallel_snn)
    params_model["teacher_use_snn_gate"] = bool(teacher_use_snn_gate)
    params_model["teacher_gate_mode"] = str(teacher_gate_mode).lower()
    params_model["teacher_sparsity_mode"] = str(teacher_sparsity_mode).lower()
    params_model["teacher_sparse_lambda"] = float(teacher_sparse_lambda)
    params_model["teacher_snn_timesteps"] = int(teacher_snn_timesteps)
    params_model["teacher_snn_gamma"] = float(teacher_snn_gamma)
    params_model["teacher_snn_alpha"] = float(teacher_snn_alpha)
    params_model["teacher_snn_beta"] = float(teacher_snn_beta)
    params_model["teacher_snn_hidden_dim"] = int(teacher_snn_hidden_dim)
    params_model["teacher_snn_threshold"] = float(teacher_snn_threshold)
    params_model["teacher_snn_decay"] = float(teacher_snn_decay)
    params_model["teacher_snn_dropout"] = float(teacher_snn_dropout)
    params_model["teacher_snn_fusion_mode"] = str(teacher_snn_fusion_mode)
    params_model["teacher_sigmoid_centered"] = bool(teacher_sigmoid_centered)
    params_model["teacher_ann_gate_snn"] = bool(teacher_ann_gate_snn)
    params_model["teacher_gate_strength"] = float(teacher_gate_strength)
    params_model["teacher_leak_strength"] = float(teacher_leak_strength)
    params_model["teacher_spike_scale_strength"] = float(teacher_spike_scale_strength)
    params_model["teacher_fire_rate_target"] = float(teacher_fire_rate_target)
    params_model["teacher_fire_rate_reg"] = float(teacher_fire_rate_reg)
    params_model["teacher_snn_arch"] = str(teacher_snn_arch)
    params_model["teacher_frontend_snn_timesteps"] = int(teacher_frontend_snn_timesteps)
    params_model["teacher_frontend_snn_hidden_dim"] = int(teacher_frontend_snn_hidden_dim)
    params_model["teacher_frontend_snn_decay"] = float(teacher_frontend_snn_decay)
    params_model["teacher_frontend_snn_threshold"] = float(teacher_frontend_snn_threshold)
    params_model["teacher_frontend_snn_dropout"] = float(teacher_frontend_snn_dropout)
    params_model["teacher_frontend_fire_rate_target"] = float(teacher_frontend_fire_rate_target)
    params_model["teacher_frontend_fire_rate_reg"] = float(teacher_frontend_fire_rate_reg)
    params_model["teacher_eval_repr"] = str(teacher_eval_repr)
    params_model["teacher_fusion_warmup_epochs"] = int(teacher_fusion_warmup_epochs)
    params_model["teacher_gamma_warmup"] = bool(teacher_gamma_warmup)
    params_model["teacher_init_ann_path"] = teacher_init_ann_path
    params_model["teacher_freeze_ann"] = bool(teacher_freeze_ann)
    params_model["teacher_freeze_text_branch"] = bool(teacher_freeze_text_branch)
    params_model["use_snn_sigmoid_ann_gate"] = bool(use_snn_sigmoid_ann_gate)
    params_model["snn_sigmoid_ann_warmup_steps"] = int(snn_sigmoid_ann_warmup_steps)
    params_model["snn_sigmoid_ann_ramp_steps"] = int(snn_sigmoid_ann_ramp_steps)
    params_model["snn_sigmoid_ann_lambda"] = float(snn_sigmoid_ann_lambda)
    params_model["snn_sigmoid_ann_detach_gate"] = bool(snn_sigmoid_ann_detach_gate)
    params_model["teacher_snn_sigmoid_ann_snn_ce"] = bool(teacher_snn_sigmoid_ann_snn_ce)
    params_model["teacher_snn_sigmoid_ann_snn_ce_weight"] = float(
        teacher_snn_sigmoid_ann_snn_ce_weight
    )
    params_model["teacher_snn_only"] = bool(teacher_snn_only)
    params_model["teacher_snn_use_video_delta"] = bool(teacher_snn_use_video_delta)
    params_model["teacher_snn_time_weight"] = str(teacher_snn_time_weight)
    params_model["teacher_snn_proto_align_lambda"] = float(teacher_snn_proto_align_lambda)
    params_model["teacher_snn_proto_align_type"] = str(teacher_snn_proto_align_type)
    params_model["feature_extraction_method"] = str(feature_extraction_method or "")
    return params_model
