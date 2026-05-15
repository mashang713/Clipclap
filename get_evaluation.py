import logging

import copy
import torch

from pathlib import Path

from src.ablation_results import append_results_ablation_csv
from src.dataset import DefaultCollator
from src.args import args_main
from torch.utils import data
from src.dataset import ActivityNetDataset, AudioSetZSLDataset, VGGSoundDataset, UCFDataset,ContrastiveDataset
from src.clipclap_model import build_clipclap_model
from src.test import test
from src.utils import fix_seeds, load_args, load_model_parameters, setup_evaluation, load_model_weights, log_hparams, print_model_size
from src.utils_improvements import get_model_params


def get_evaluation(args):

    config = load_args(args.load_path_stage_B)
    config.root_dir = args.root_dir
    if config.input_size is not None:
        config.input_size_audio = config.input_size
        config.input_size_video = config.input_size
    if getattr(args, "eval_modality", None) is not None:
        config.eval_modality = args.eval_modality

    assert config.retrain_all, f"--retrain_all flag is not set in load_path_stage_B. Are you sure this is the correct path?. {args.load_path_stage_B}"
    fix_seeds(config.seed)

    logger, eval_dir, test_stats, tb_writer = setup_evaluation(args, config.__dict__.keys())


    if args.dataset_name == "AudioSetZSL":
        val_all_dataset = AudioSetZSLDataset(
            args=config,
            dataset_split="val",
            zero_shot_mode="all",
        )
        test_dataset = AudioSetZSLDataset(
            args=config,
            dataset_split="test",
            zero_shot_mode="all",
        )
    elif args.dataset_name == "VGGSound":
        val_all_dataset = VGGSoundDataset(
            args=config,
            dataset_split="val",
            zero_shot_mode=None,
        )
        test_dataset = VGGSoundDataset(
            args=config,
            dataset_split="test",
            zero_shot_mode=None,
        )
    elif args.dataset_name == "UCF":
        val_all_dataset = UCFDataset(
            args=config,
            dataset_split="val",
            zero_shot_mode=None,
        )
        test_dataset = UCFDataset(
            args=config,
            dataset_split="test",
            zero_shot_mode=None,
        )
    elif args.dataset_name == "ActivityNet":
        val_all_dataset = ActivityNetDataset(
            args=config,
            dataset_split="val",
            zero_shot_mode=None,
        )
        test_dataset = ActivityNetDataset(
            args=config,
            dataset_split="test",
            zero_shot_mode=None,
        )
    else:
        raise NotImplementedError()


    contrastive_val_dataset = ContrastiveDataset(val_all_dataset)
    contrastive_test_dataset = ContrastiveDataset(test_dataset)

    if config.selavi == False:
        collator_test = DefaultCollator(mode=args.batch_seqlen_test, max_len=args.batch_seqlen_test_maxlen, trim=args.batch_seqlen_test_trim)
    elif config.selavi==True:
        collator_test = DefaultCollator(mode=args.batch_seqlen_test, max_len=args.batch_seqlen_test_maxlen, trim=args.batch_seqlen_test_trim,rate_video=1, rate_audio=1)

    final_val_loader = data.DataLoader(
        dataset=contrastive_val_dataset,
        collate_fn=collator_test,
        batch_size=args.eval_bs,
        num_workers=args.eval_num_workers,
    )

    final_test_loader = data.DataLoader(
        dataset=contrastive_test_dataset,
        collate_fn=collator_test,
        batch_size=args.eval_bs,
        num_workers=args.eval_num_workers,
    )


    model_params = get_model_params(
        config.lr, config.reg_loss, config.embedding_dropout, config.decoder_dropout,
        config.additional_dropout,config.embeddings_hidden_size, config.decoder_hidden_size,
        config.embeddings_batch_norm, config.rec_loss,config.cross_entropy_loss,
        config.transformer_use_embedding_net, config.transformer_dim, config.transformer_depth, config.transformer_heads,
        config.transformer_dim_head, config.transformer_mlp_dim, config.transformer_dropout,
        config.transformer_embedding_dim, config.transformer_embedding_time_len, config.transformer_embedding_dropout,
        config.transformer_embedding_time_embed_type, config.transformer_embedding_fourier_scale, config.transformer_embedding_embed_augment_position,
        config.lr_scheduler, config.optimizer, config.use_self_attention, config.use_cross_attention, config.transformer_average_features,
        config.audio_only, config.video_only, config.transformer_use_class_token, config.transformer_embedding_modality,
        config.modality, config.word_embeddings,
        getattr(config, "model_backend", "ann"),
        getattr(config, "snn_num_steps", 10),
        getattr(config, "snn_beta", 0.9),
        getattr(config, "snn_threshold", 1.0),
        getattr(config, "use_snn_conversion", False),
        getattr(config, "snn_timesteps", 4),
        getattr(config, "lambda_proto", 1.0),
        getattr(config, "lambda_feat", 1.0),
        getattr(config, "proto_temperature", 1.0),
        getattr(config, "snn_conv_threshold_percentile", 0.99),
        getattr(config, "proto_kd_type", "kl_all"),
        getattr(config, "proto_topk", 10),
        getattr(config, "proto_warmup_epochs", 0),
        getattr(config, "proto_conf_margin", 0.0),
        getattr(config, "debug_print_shapes", False),
        getattr(config, "use_teacher_parallel_snn", False),
        getattr(config, "teacher_snn_timesteps", 4),
        getattr(config, "teacher_snn_gamma", 0.1),
        getattr(config, "teacher_snn_alpha", 0.1),
        getattr(config, "teacher_snn_beta", 1.0),
        getattr(config, "teacher_snn_hidden_dim", 512),
        getattr(config, "teacher_snn_threshold", 1.0),
        getattr(config, "teacher_snn_decay", 0.9),
        getattr(config, "teacher_snn_dropout", 0.1),
        getattr(config, "teacher_snn_fusion_mode", "add"),
        getattr(config, "teacher_ann_gate_snn", False),
        getattr(config, "teacher_gate_strength", 0.5),
        getattr(config, "teacher_leak_strength", 0.2),
        getattr(config, "teacher_spike_scale_strength", 0.2),
        getattr(config, "teacher_fire_rate_target", 0.1),
        getattr(config, "teacher_fire_rate_reg", 0.0),
        getattr(config, "teacher_snn_arch", "backend_only"),
        getattr(config, "teacher_frontend_snn_timesteps", 4),
        getattr(config, "teacher_frontend_snn_hidden_dim", 512),
        getattr(config, "teacher_frontend_snn_decay", 0.9),
        getattr(config, "teacher_frontend_snn_threshold", 1.0),
        getattr(config, "teacher_frontend_snn_dropout", 0.1),
        getattr(config, "teacher_frontend_fire_rate_target", 0.1),
        getattr(config, "teacher_frontend_fire_rate_reg", 0.0),
        getattr(config, "teacher_eval_repr", "fused"),
        getattr(config, "teacher_fusion_warmup_epochs", 0),
        getattr(config, "teacher_gamma_warmup", True),
    )

    if config.new_model_sequence==True:
        model_A = build_clipclap_model(model_params, input_size_audio=config.input_size_audio, input_size_video=config.input_size_video)
    else:
        raise AttributeError("No correct model_A name.")
    print_model_size(model_A, logger)
    logging.info(model_A)

    model_B = copy.deepcopy(model_A)

    # weights_path_stage_A = list(args.load_path_stage_A.glob("*_score.pt"))[0]
    weights_path_stage_A = list(args.load_path_stage_A.glob(f"*_{config.best_model_criterion}.pt"))[0]
    epoch_A = load_model_weights(weights_path_stage_A, model_A)
    weights_path_stage_B = list((args.load_path_stage_B / "checkpoints").glob(f"*_ckpt_{epoch_A - 1}.pt"))[0]


    _ = load_model_weights(weights_path_stage_B, model_B)

    model_A.to(config.device)
    model_B.to(config.device)

    results = test(
        eval_name=args.eval_name,
        val_dataset=(val_all_dataset, final_val_loader),
        test_dataset=(test_dataset, final_test_loader),
        model_A=model_A,
        model_B=model_B,
        device=args.device,
        distance_fn=config.distance_fn,
        test_stats=test_stats,
        eval_dir=eval_dir,
        new_model_sequence=config.new_model_sequence,
        args=config,
        save_performances=args.eval_save_performances
    )

    # Tensorboard HParam logging
    log_hparams(tb_writer, config, results['both'])

    csv_path = getattr(args, "ablation_csv", None)
    if csv_path is None:
        csv_path = Path(__file__).resolve().parent / "results_ablation.csv"
    else:
        csv_path = Path(csv_path)
    run_label = (getattr(config, "ablation_run_name", "") or "").strip()
    if not run_label:
        run_label = (getattr(config, "exp_name", "") or "unnamed").strip()
    append_results_ablation_csv(
        csv_path,
        run_name=run_label,
        config=config,
        results_both=results["both"],
        final_checkpoint_path=weights_path_stage_B,
        log_dir_for_tb=Path(args.load_path_stage_B),
    )
    logger.info("Appended ablation row to %s", csv_path.resolve())

    logger.info("FINISHED")
    # return results['both']


if __name__ == "__main__":
    args, eval_args = args_main()
    get_evaluation(eval_args)
