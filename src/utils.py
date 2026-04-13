from copy import deepcopy
import pathlib
import yaml
import logging
import pickle
import socket
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import subprocess
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.logger import PD_Stats, create_logger


def read_features(path):

    with open(path, 'rb') as f:
        x = pickle.load(f)

    data = x['features']
    fps=x['fps']
    url = [str(u) for u in list(x['video_names'])]

    return data, url, fps


def fix_seeds(seed=42):

    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def print_model_size(model, logger):
    num_params = 0
    for param in model.parameters():
        if param.requires_grad:
            num_params += param.numel()
    logger.info(
        "Created network [%s] with total number of parameters: %.1f million."
        % (type(model).__name__, num_params / 1000000)
    )


def get_git_revision_hash():
    try:
        hash_string = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except:
        hash_string = ""
    return hash_string

def dump_config_yaml(args, exp_dir):
    args_dict = deepcopy(vars(args))
    for k,v in args_dict.items():
        if isinstance(v, pathlib.PosixPath):
            args_dict[k] = v.as_posix()

    with open((exp_dir/"args.yaml"), "w") as f:
        yaml.safe_dump(args_dict, f)

def log_hparams(writer, args, metrics):
    args_dict = vars(args)
    for k,v in args_dict.items():
        if isinstance(v, pathlib.PosixPath):
            args_dict[k] = v.as_posix()
    del metrics["recall"]
    metrics = {"Eval/"+k: v for k,v in metrics.items()}
    args_dict.pop('audio_hip_blocks', None)
    args_dict.pop('video_hip_blocks', None)

    writer.add_hparams(args_dict, metrics)


def setup_experiment(args, *stats):
    if args.exp_name == "":
        exp_name = f"{datetime.now().strftime('%b%d_%H-%M-%S_%f')}_{socket.gethostname()}"
    else:
        exp_name = str(args.exp_name) + f"_{datetime.now().strftime('%b%d_%H-%M-%S_%f')}_{socket.gethostname()}"

    exp_dir = (args.log_dir / exp_name)
    exp_dir.mkdir(parents=True)

    (exp_dir / "checkpoints").mkdir()
    pickle.dump(args, (exp_dir / "args.pkl").open("wb"))

    dump_config_yaml(args, exp_dir)

    train_stats = PD_Stats(exp_dir / "train_stats.pkl", stats)
    val_stats = PD_Stats(exp_dir / "val_stats.pkl", stats)

    logger = create_logger(exp_dir / "train.log")

    logger.info(f"Start experiment {exp_name}")
    logger.info(
        "\n".join(f"{k}: {str(v)}" for k, v in sorted(dict(vars(args)).items()))
    )
    logger.info(f"The experiment will be stored in {exp_dir.resolve()}\n")
    logger.info("")

    writer = SummaryWriter(log_dir=exp_dir)
    return logger, exp_dir, writer, train_stats, val_stats


def setup_evaluation(args, *stats):

    eval_dir = args.load_path_stage_B
    assert eval_dir.exists()

    # test_stats = PD_Stats(eval_dir / "test_stats.pkl", list(sorted(stats)))
    test_stats = PD_Stats(eval_dir / "test_stats.pkl", ['seen', 'unseen', 'hm', 'zsl'])
    logger = create_logger(eval_dir / "eval.log")

    logger.info(f"Start evaluation {eval_dir}")
    logger.info(
        "\n".join(f"{k}: {str(v)}" for k, v in sorted(dict(vars(args)).items()))
    )
    logger.info(f"Loaded configuration {args.load_path_stage_B / 'args.pkl'}")
    logger.info(
        "\n".join(f"{k}: {str(v)}" for k, v in sorted(dict(vars(load_args(args.load_path_stage_B))).items()))
    )
    logger.info(f"The evaluation will be stored in {eval_dir.resolve()}\n")
    logger.info("")

    # for Tensorboard hparam logging
    writer = SummaryWriter(log_dir=eval_dir)

    return logger, eval_dir, test_stats, writer

def setup_visualizations(args, *stats):

    eval_dir = args.load_path_stage_B
    assert eval_dir.exists()

    # test_stats = PD_Stats(eval_dir / "test_stats.pkl", list(sorted(stats)))
    test_stats = PD_Stats(eval_dir / "test_stats.pkl", ['seen', 'unseen', 'hm', 'zsl'])
    logger = create_logger(eval_dir / "visuals.log")

    logger.info(f"Start visualizing {eval_dir}")
    logger.info(
        "\n".join(f"{k}: {str(v)}" for k, v in sorted(dict(vars(args)).items()))
    )
    logger.info(f"Loaded configuration {args.load_path_stage_B / 'args.pkl'}")
    logger.info(
        "\n".join(f"{k}: {str(v)}" for k, v in sorted(dict(vars(load_args(args.load_path_stage_B))).items()))
    )
    logger.info(f"The evaluation will be stored in {eval_dir.resolve()}\n")
    logger.info("")

    # for Tensorboard hparam logging
    writer = SummaryWriter(log_dir=eval_dir)

    return logger, eval_dir, test_stats, writer


def save_best_model(epoch, best_metric, model, optimizer, log_dir, args, metric="", checkpoint=False):
    logger = logging.getLogger()
    logger.info(f"Saving model to {log_dir} with {metric} = {best_metric:.4f}")
    if optimizer is None:
        optimizer = model.optimizer_gen
    save_dict = {
        "epoch": epoch + 1,
        "model": model.state_dict() if args.data_parallel == False else model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metric": metric
    }
    if checkpoint:
        torch.save(
            save_dict,
            log_dir / f"{model.__class__.__name__}_{metric}_ckpt_{epoch}.pt"
        )
    else:
        torch.save(
            save_dict,
            log_dir / f"{model.__class__.__name__}_{metric}.pt"
        )




def check_best_loss(epoch, best_loss, best_epoch, val_loss, model, optimizer, log_dir, args):
    if not best_loss:
        save_best_model(epoch, val_loss, model, optimizer, log_dir, args, metric="loss")
        return val_loss, epoch

    if val_loss < best_loss:
        best_loss = val_loss
        best_epoch = epoch
        save_best_model(epoch, best_loss, model, optimizer, log_dir, args, metric="loss")
    return best_loss, best_epoch



def check_best_score(epoch, best_score, best_epoch, hm_score, model, optimizer, log_dir, args):
    if not best_score:
        save_best_model(epoch, hm_score, model, optimizer, log_dir, args, metric="score")
        return hm_score, epoch
    if hm_score > best_score:
        best_score = hm_score
        best_epoch = epoch
        save_best_model(epoch, best_score, model, optimizer, log_dir, args, metric="score")
    return best_score, best_epoch


def load_model_parameters(model, model_weights):
    logger = logging.getLogger()
    loaded_state = model_weights
    self_state = model.state_dict()
    for name, param in loaded_state.items():
        param = param
        if 'module.' in name:
            name = name.replace('module.', '')
        if name in self_state.keys():
            self_state[name].copy_(param)
        else:
            logger.info("didnt load ", name)


def load_args(path):
    return pickle.load((path / "args.pkl").open("rb"))


def cos_dist(a, b):
    # https://stackoverflow.com/questions/50411191/how-to-compute-the-cosine-similarity-in-pytorch-for-all-rows-in-a-matrix-with-re
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    res = torch.mm(a_norm, b_norm.transpose(0, 1))
    return res


def _resolve_distance_fn_name(distance_fn, args):
    if isinstance(distance_fn, str):
        return distance_fn
    if args is not None and getattr(args, "distance_fn", None) is not None:
        return args.distance_fn
    return type(distance_fn).__name__


def collect_eval_embeddings(dataset_tuple, model, device, args):
    dataset = dataset_tuple[0]
    data_loader = dataset_tuple[1]
    data_t = torch.tensor(dataset.all_data["text"]).to(device)
    accumulated_audio_emb = []
    accumulated_video_emb = []
    accumulated_data_num = []

    for batch_idx, (data, target) in tqdm(enumerate(data_loader)):
        data_a = data["positive"]["audio"].to(device)
        data_v = data["positive"]["video"].to(device)
        data_num = target["positive"].to(device)
        masks = {}
        masks["positive"] = {"audio": data["positive"]["audio_mask"], "video": data["positive"]["video_mask"]}
        timesteps = {}
        timesteps["positive"] = {
            "audio": data["positive"]["timestep"]["audio"],
            "video": data["positive"]["timestep"]["video"],
        }

        all_data = (data_a, data_v, data_num, data_t, masks["positive"], timesteps["positive"])
        try:
            if args.z_score_inputs:
                all_data = tuple([(x - torch.mean(x)) / torch.sqrt(torch.var(x)) for x in all_data])
        except AttributeError:
            print("Namespace has no fitting attribute. Continuing")

        model.eval()
        with torch.no_grad():
            audio_emb, video_emb, emb_cls = model.get_embeddings(
                all_data[0], all_data[1], all_data[3], all_data[4], all_data[5]
            )
            accumulated_audio_emb.append(audio_emb)
            accumulated_video_emb.append(video_emb)
            outputs_all = (audio_emb, video_emb, emb_cls)

        accumulated_data_num.append(data_num)

    stacked_audio_emb = torch.cat(accumulated_audio_emb)
    stacked_video_emb = torch.cat(accumulated_video_emb)
    data_num = torch.cat(accumulated_data_num)
    emb_cls = outputs_all[2]
    return dataset, data_num, stacked_audio_emb, stacked_video_emb, emb_cls


def _compute_distance_matrices(
    dataset, a_p, v_p, t_p, mode, device, distance_fn_name, seen_label_array, unseen_label_array, seen_unseen_array
):
    huge = 99999999999999.0
    classes_embeddings = t_p
    if a_p is None:
        if mode != "video":
            raise ValueError("a_p is None is only supported for mode='video'")
        n = v_p.shape[0]
    else:
        n = a_p.shape[0]
    distance_mat = torch.full((n, len(dataset.all_class_ids)), huge, dtype=torch.float32, device=device)
    distance_mat_zsl = torch.full((n, len(dataset.all_class_ids)), huge, dtype=torch.float32, device=device)

    if mode == "audio":
        distance_mat[:, seen_unseen_array] = torch.cdist(a_p.float(), classes_embeddings.float(), p=2)
        mask_seen_inf = torch.zeros(len(dataset.all_class_ids), dtype=torch.float32, device=device)
        mask_seen_inf[seen_label_array] = huge
        distance_mat_zsl = distance_mat + mask_seen_inf
        if distance_fn_name == "SquaredL2Loss":
            distance_mat[:, seen_unseen_array] = distance_mat[:, seen_unseen_array].pow(2)
            distance_mat_zsl[:, unseen_label_array] = distance_mat_zsl[:, unseen_label_array].pow(2)
    elif mode == "video":
        v_pf = v_p.type(torch.float32)
        distance_mat[:, seen_unseen_array] = torch.cdist(v_pf, classes_embeddings.float(), p=2)
        mask_seen_inf = torch.zeros(len(dataset.all_class_ids), dtype=torch.float32, device=device)
        mask_seen_inf[seen_label_array] = huge
        distance_mat_zsl = distance_mat + mask_seen_inf
        if distance_fn_name == "SquaredL2Loss":
            distance_mat[:, seen_unseen_array] = distance_mat[:, seen_unseen_array].pow(2)
            distance_mat_zsl[:, unseen_label_array] = distance_mat_zsl[:, unseen_label_array].pow(2)
    elif mode == "both":
        audio_distance = torch.cdist(a_p.float(), classes_embeddings.float(), p=2)
        video_distance = torch.cdist(v_p.float(), classes_embeddings.float(), p=2)
        if distance_fn_name == "SquaredL2Loss":
            audio_distance = audio_distance.pow(2)
            video_distance = video_distance.pow(2)
        distance_mat[:, seen_unseen_array] = audio_distance + video_distance
        mask_seen_inf = torch.zeros(len(dataset.all_class_ids), dtype=torch.float32, device=device)
        mask_seen_inf[seen_label_array] = huge
        distance_mat_zsl = distance_mat + mask_seen_inf
    else:
        raise ValueError(f"Unknown mode {mode}")

    return distance_mat, distance_mat_zsl


def _gzsl_zsl_scores_from_neighbors(
    distance_mat,
    distance_mat_zsl,
    neighbor_gzsl,
    targets,
    dataset,
    seen_label_array,
    unseen_label_array,
    seen_unseen_array,
):
    match_idx = neighbor_gzsl.eq(targets.int()).nonzero().flatten()
    match_counts = torch.bincount(neighbor_gzsl[match_idx], minlength=len(dataset.all_class_ids))[seen_unseen_array]
    target_counts = torch.bincount(targets, minlength=len(dataset.all_class_ids))[seen_unseen_array]
    per_class_recall = torch.zeros(len(dataset.all_class_ids), dtype=torch.float, device=distance_mat.device)
    per_class_recall[seen_unseen_array] = match_counts / target_counts
    seen_recall_dict = per_class_recall[seen_label_array]
    unseen_recall_dict = per_class_recall[unseen_label_array]
    s = seen_recall_dict.mean()
    u = unseen_recall_dict.mean()
    hm = (2 * u * s) / ((u + s) + np.finfo(float).eps)

    neighbor_batch_zsl = torch.argmin(distance_mat_zsl, dim=1)
    match_idx = neighbor_batch_zsl.eq(targets.int()).nonzero().flatten()
    match_counts = torch.bincount(neighbor_batch_zsl[match_idx], minlength=len(dataset.all_class_ids))[seen_unseen_array]
    target_counts = torch.bincount(targets, minlength=len(dataset.all_class_ids))[seen_unseen_array]
    per_class_recall = torch.zeros(len(dataset.all_class_ids), dtype=torch.float, device=distance_mat.device)
    per_class_recall[seen_unseen_array] = match_counts / target_counts
    zsl = per_class_recall[unseen_label_array].mean()

    return s, u, hm, zsl, per_class_recall


def _eval_calibration_step(
    distance_mat,
    distance_mat_zsl,
    targets,
    dataset,
    seen_label_array,
    unseen_label_array,
    seen_unseen_array,
    beta,
    tau,
    use_calibrated_stacking,
    calibration_mode,
):
    n_classes = len(dataset.all_class_ids)
    bias = torch.zeros(n_classes, dtype=distance_mat.dtype, device=distance_mat.device)
    bias[seen_label_array] = beta

    if not use_calibrated_stacking:
        neighbor_gzsl = torch.argmin(distance_mat, dim=1)
    elif calibration_mode == "beta":
        neighbor_gzsl = torch.argmin(distance_mat + bias, dim=1)
    elif calibration_mode == "tau_beta":
        neighbor_gzsl = torch.argmin(tau * distance_mat + bias, dim=1)
    else:
        raise ValueError(f"Unknown calibration_mode {calibration_mode}")

    return _gzsl_zsl_scores_from_neighbors(
        distance_mat,
        distance_mat_zsl,
        neighbor_gzsl,
        targets,
        dataset,
        seen_label_array,
        unseen_label_array,
        seen_unseen_array,
    )


def evaluate_from_stacked_embeddings(
    dataset,
    data_num,
    a_p,
    v_p,
    t_p,
    device,
    distance_fn,
    best_beta=None,
    best_tau=None,
    save_performances=False,
    args=None,
    return_grid_results=False,
):
    use_cs = getattr(args, "use_calibrated_stacking", True) if args else True
    cal_mode = getattr(args, "calibration_mode", "beta") if args else "beta"
    return {
        "audio": get_best_evaluation(
            dataset,
            data_num,
            a_p,
            v_p,
            t_p,
            mode="audio",
            device=device,
            distance_fn=distance_fn,
            best_beta=best_beta,
            best_tau=best_tau,
            use_calibrated_stacking=use_cs,
            calibration_mode=cal_mode,
            save_performances=False,
            args=args,
            return_grid_results=False,
        ),
        "video": get_best_evaluation(
            dataset,
            data_num,
            a_p,
            v_p,
            t_p,
            mode="video",
            device=device,
            distance_fn=distance_fn,
            best_beta=best_beta,
            best_tau=best_tau,
            use_calibrated_stacking=use_cs,
            calibration_mode=cal_mode,
            save_performances=False,
            args=args,
            return_grid_results=False,
        ),
        "both": get_best_evaluation(
            dataset,
            data_num,
            a_p,
            v_p,
            t_p,
            mode="both",
            device=device,
            distance_fn=distance_fn,
            best_beta=best_beta,
            best_tau=best_tau,
            use_calibrated_stacking=use_cs,
            calibration_mode=cal_mode,
            save_performances=save_performances,
            args=args,
            return_grid_results=return_grid_results,
        ),
    }


def evaluate_dataset_baseline(
    dataset_tuple,
    model,
    device,
    distance_fn,
    best_beta=None,
    best_tau=None,
    new_model_sequence=False,
    args=None,
    save_performances=False,
):
    dataset, data_num, a_p, v_p, t_p = collect_eval_embeddings(dataset_tuple, model, device, args)
    return evaluate_from_stacked_embeddings(
        dataset,
        data_num,
        a_p,
        v_p,
        t_p,
        device,
        distance_fn,
        best_beta=best_beta,
        best_tau=best_tau,
        save_performances=save_performances,
        args=args,
        return_grid_results=False,
    )


def get_best_evaluation(
    dataset,
    targets,
    a_p,
    v_p,
    t_p,
    mode,
    device,
    distance_fn,
    best_beta=None,
    best_tau=None,
    use_calibrated_stacking=True,
    calibration_mode="beta",
    save_performances=False,
    args=None,
    attention_weights=None,
    return_grid_results=False,
):
    seen_label_array = torch.tensor(dataset.seen_class_ids, dtype=torch.long, device=device)
    unseen_label_array = torch.tensor(dataset.unseen_class_ids, dtype=torch.long, device=device)
    seen_unseen_array = torch.tensor(
        np.sort(np.concatenate((dataset.seen_class_ids, dataset.unseen_class_ids))), dtype=torch.long, device=device
    )
    fn_name = _resolve_distance_fn_name(distance_fn, args)

    beta_start = float(getattr(args, "calibration_beta_start", 0.0)) if args else 0.0
    beta_end = float(getattr(args, "calibration_beta_end", 5.0)) if args else 5.0
    beta_steps = int(getattr(args, "calibration_beta_steps", 76)) if args else 76
    tau_start = float(getattr(args, "calibration_tau_start", 0.5)) if args else 0.5
    tau_end = float(getattr(args, "calibration_tau_end", 2.0)) if args else 2.0
    tau_steps = int(getattr(args, "calibration_tau_steps", 16)) if args else 16

    distance_mat, distance_mat_zsl = _compute_distance_matrices(
        dataset, a_p, v_p, t_p, mode, device, fn_name, seen_label_array, unseen_label_array, seen_unseen_array
    )

    if not use_calibrated_stacking:
        s, u, hm, zsl, per_class_recall = _eval_calibration_step(
            distance_mat,
            distance_mat_zsl,
            targets,
            dataset,
            seen_label_array,
            unseen_label_array,
            seen_unseen_array,
            beta=0.0,
            tau=1.0,
            use_calibrated_stacking=False,
            calibration_mode="beta",
        )
        out = {
            "seen": s.item(),
            "unseen": u.item(),
            "hm": float(hm.item()) if hasattr(hm, "item") else float(hm),
            "recall": per_class_recall.tolist(),
            "zsl": zsl.item(),
            "beta": 0.0,
            "tau": 1.0,
        }
        if save_performances:
            seen_recall_dict = per_class_recall[seen_label_array]
            unseen_recall_dict = per_class_recall[unseen_label_array]
            seen_dict = {
                k: v
                for k, v in zip(
                    np.array(dataset.all_class_names)[seen_label_array.cpu().numpy()],
                    seen_recall_dict.cpu().numpy(),
                )
            }
            unseen_dict = {
                k: v
                for k, v in zip(
                    np.array(dataset.all_class_names)[unseen_label_array.cpu().numpy()],
                    unseen_recall_dict.cpu().numpy(),
                )
            }
            save_class_performances(seen_dict, unseen_dict, dataset.dataset_name, args=args)
        return out

    if best_beta is not None:
        betas = torch.tensor([float(best_beta)], dtype=torch.float, device=device)
        if calibration_mode == "tau_beta":
            t_val = float(best_tau) if best_tau is not None else 1.0
            taus = torch.tensor([t_val], dtype=torch.float, device=device)
        else:
            taus = torch.tensor([1.0], dtype=torch.float, device=device)
    else:
        betas = torch.linspace(beta_start, beta_end, beta_steps, device=device)
        if calibration_mode == "tau_beta":
            taus = torch.linspace(tau_start, tau_end, tau_steps, device=device)
        else:
            taus = torch.tensor([1.0], dtype=torch.float, device=device)

    seen_scores = []
    zsl_scores = []
    unseen_scores = []
    hm_scores = []
    per_class_recalls = []
    betas_list = []
    taus_list = []
    grid_results = []

    with torch.no_grad():
        for tau in taus:
            tau_f = float(tau.item())
            for beta in betas:
                beta_f = float(beta.item())
                s, u, hm, zsl, per_class_recall = _eval_calibration_step(
                    distance_mat,
                    distance_mat_zsl,
                    targets,
                    dataset,
                    seen_label_array,
                    unseen_label_array,
                    seen_unseen_array,
                    beta_f,
                    tau_f,
                    use_calibrated_stacking=True,
                    calibration_mode=calibration_mode,
                )
                seen_scores.append(s.item())
                unseen_scores.append(u.item())
                hm_scores.append(float(hm.item()) if hasattr(hm, "item") else float(hm))
                zsl_scores.append(zsl.item())
                per_class_recalls.append(per_class_recall.tolist())
                betas_list.append(beta_f)
                taus_list.append(tau_f)
                if return_grid_results:
                    grid_results.append(
                        {
                            "tau": tau_f,
                            "beta": beta_f,
                            "seen": s.item(),
                            "unseen": u.item(),
                            "hm": float(hm.item()) if hasattr(hm, "item") else float(hm),
                            "zsl": zsl.item(),
                        }
                    )

    argmax_hm = int(np.argmax(hm_scores))
    max_seen = seen_scores[argmax_hm]
    max_zsl = zsl_scores[argmax_hm]
    max_unseen = unseen_scores[argmax_hm]
    max_hm = hm_scores[argmax_hm]
    max_recall = per_class_recalls[argmax_hm]
    out_beta = betas_list[argmax_hm]
    out_tau = taus_list[argmax_hm]

    if save_performances:
        max_recall_t = torch.tensor(max_recall, dtype=torch.float, device=device)
        seen_recall_dict = max_recall_t[seen_label_array]
        unseen_recall_dict = max_recall_t[unseen_label_array]
        seen_dict = {
            k: v
            for k, v in zip(
                np.array(dataset.all_class_names)[seen_label_array.cpu().numpy()],
                seen_recall_dict.cpu().numpy(),
            )
        }
        unseen_dict = {
            k: v
            for k, v in zip(
                np.array(dataset.all_class_names)[unseen_label_array.cpu().numpy()],
                unseen_recall_dict.cpu().numpy(),
            )
        }
        save_class_performances(seen_dict, unseen_dict, dataset.dataset_name, args=args)

    out = {
        "seen": max_seen,
        "unseen": max_unseen,
        "hm": max_hm,
        "recall": max_recall,
        "zsl": max_zsl,
        "beta": out_beta,
        "tau": out_tau,
    }
    if return_grid_results:
        out["grid_results"] = grid_results
    return out

def get_class_names(path):
    if isinstance(path, str):
        path = Path(path)
    with path.open("r") as f:
        classes = sorted([line.strip() for line in f])
    return classes


def load_model_weights(weights_path, model):
    logging.info(f"Loading model weights from {weights_path}")
    load_dict = torch.load(weights_path)
    model_weights = load_dict["model"]
    epoch = load_dict["epoch"]
    logging.info(f"Load from epoch: {epoch}")
    load_model_parameters(model, model_weights)
    return epoch

def plot_hist_from_dict(dict):
    plt.bar(range(len(dict)), list(dict.values()), align="center")
    plt.xticks(range(len(dict)), list(dict.keys()), rotation='vertical')
    plt.tight_layout()
    plt.show()

def save_class_performances(seen_dict, unseen_dict, dataset_name, args=None):
    roor_path = args.load_path_stage_B
    seen_dir = os.path.join(roor_path, f'class_performance_{dataset_name}_seen.pkl')
    unseen_dir = os.path.join(roor_path, f'class_performance_{dataset_name}_unseen.pkl')

    seen_path = Path(seen_dir)
    unseen_path = Path(unseen_dir)
    with seen_path.open("wb") as f:
        pickle.dump(seen_dict, f)
        logging.info(f"Saving seen class performances to {seen_path}")
    with unseen_path.open("wb") as f:
        pickle.dump(unseen_dict, f)
        logging.info(f"Saving unseen class performances to {unseen_path}")
