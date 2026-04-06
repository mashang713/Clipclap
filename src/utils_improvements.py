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
    snn_use_aif=False,
    snn_aif_mode="none",
    snn_aif_k=3.0,
    snn_aif_stats_path=None,
    snn_aif_collect_stats=False,
    snn_aif_num_batches=50,
    snn_aif_dataset_split="train",
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
            "use_aif": snn_use_aif,
        }
    else:
        params_model["snn_embedding_kwargs"] = {}

    # Keep AIF config at top-level for pipeline orchestration (stats collection / injection).
    params_model["snn_use_aif"] = snn_use_aif
    params_model["snn_aif_mode"] = snn_aif_mode
    params_model["snn_aif_k"] = snn_aif_k
    params_model["snn_aif_stats_path"] = snn_aif_stats_path
    params_model["snn_aif_collect_stats"] = snn_aif_collect_stats
    params_model["snn_aif_num_batches"] = snn_aif_num_batches
    params_model["snn_aif_dataset_split"] = snn_aif_dataset_split
    return params_model
