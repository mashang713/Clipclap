import json
import logging
from pathlib import Path

from src.utils import collect_eval_embeddings, evaluate_from_stacked_embeddings, get_best_evaluation


def test(
    eval_name,
    val_dataset,
    test_dataset,
    model_A,
    model_B,
    device,
    distance_fn,
    test_stats,
    eval_dir,
    args,
    new_model_sequence=False,
    save_performances=False,
):
    logger = logging.getLogger()
    model_A.eval()
    model_B.eval()

    test_evaluation = _get_test_performance(
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        model_A=model_A,
        model_B=model_B,
        device=device,
        distance_fn=distance_fn,
        args=args,
        new_model_sequence=new_model_sequence,
        save_performances=save_performances,
        eval_dir=eval_dir,
    )

    if args.dataset_name == "AudioSetZSL":
        output_string = fr"""
                   Seen performance={100 * test_evaluation["both"]["seen"]:.2f}, Unseen performance={100 * test_evaluation["both"]["unseen"]:.2f}, GZSL performance={100 * test_evaluation["both"]["hm"]:.2f}, ZSL performance={100 * test_evaluation["both"]["zsl"]:.2f}
                   """
    elif args.dataset_name == "VGGSound" or args.dataset_name == "UCF" or args.dataset_name == "ActivityNet":
        output_string = fr"""
                    Seen performance={100 * test_evaluation["both"]["seen"]:.2f}, Unseen performance={100 * test_evaluation["both"]["unseen"]:.2f}, GZSL performance={100 * test_evaluation["both"]["hm"]:.2f}, ZSL performance={100 * test_evaluation["both"]["zsl"]:.2f}
                    """
    else:
        raise NotImplementedError()

    logger.info(output_string)
    test_stats.update(
        (
            100 * test_evaluation["both"]["seen"],
            100 * test_evaluation["both"]["unseen"],
            100 * test_evaluation["both"]["hm"],
            100 * test_evaluation["both"]["zsl"],
        )
    )
    return test_evaluation


def _get_test_performance(
    val_dataset,
    test_dataset,
    model_A,
    model_B,
    device,
    distance_fn,
    args,
    new_model_sequence,
    save_performances=False,
    eval_dir=None,
):
    logger = logging.getLogger()
    strategy = getattr(args, "calibration_combined_beta_strategy", "legacy_average")
    use_cs = getattr(args, "use_calibrated_stacking", True)
    cal_mode = getattr(args, "calibration_mode", "beta")
    save_json = getattr(args, "calibration_eval_save_json", True)

    calibration_log = {
        "use_calibrated_stacking": use_cs,
        "calibration_mode": cal_mode,
        "calibration_combined_beta_strategy": strategy,
        "val": {},
        "test": {},
    }

    if strategy == "search_on_both":
        dataset_v, data_num_v, a_v, v_v, t_v = collect_eval_embeddings(val_dataset, model_A, device, args)
        dataset_te, data_num_te, a_te, v_te, t_te = collect_eval_embeddings(test_dataset, model_B, device, args)

        val_baseline_both = get_best_evaluation(
            dataset_v,
            data_num_v,
            a_v,
            v_v,
            t_v,
            mode="both",
            device=device,
            distance_fn=distance_fn,
            best_beta=None,
            best_tau=None,
            use_calibrated_stacking=False,
            calibration_mode="beta",
            save_performances=False,
            args=args,
            return_grid_results=False,
        )
        if not use_cs:
            logger.warning(
                "calibration_combined_beta_strategy=search_on_both but use_calibrated_stacking=False; "
                "skipping (tau,beta) grid search. Combined metrics use uncalibrated argmin only."
            )
            val_search_both = val_baseline_both
            best_beta_combined = 0.0
            best_tau_combined = 1.0
            grid_results = []
        else:
            val_search_both = get_best_evaluation(
                dataset_v,
                data_num_v,
                a_v,
                v_v,
                t_v,
                mode="both",
                device=device,
                distance_fn=distance_fn,
                best_beta=None,
                best_tau=None,
                use_calibrated_stacking=True,
                calibration_mode=cal_mode,
                save_performances=False,
                args=args,
                return_grid_results=True,
            )
            best_beta_combined = val_search_both["beta"]
            best_tau_combined = val_search_both["tau"]
            grid_results = val_search_both.get("grid_results") or []

        test_baseline_both = get_best_evaluation(
            dataset_te,
            data_num_te,
            a_te,
            v_te,
            t_te,
            mode="both",
            device=device,
            distance_fn=distance_fn,
            best_beta=None,
            best_tau=None,
            use_calibrated_stacking=False,
            calibration_mode="beta",
            save_performances=False,
            args=args,
            return_grid_results=False,
        )
        test_both_calibrated = get_best_evaluation(
            dataset_te,
            data_num_te,
            a_te,
            v_te,
            t_te,
            mode="both",
            device=device,
            distance_fn=distance_fn,
            best_beta=best_beta_combined,
            best_tau=best_tau_combined,
            use_calibrated_stacking=use_cs,
            calibration_mode=cal_mode,
            save_performances=save_performances,
            args=args,
            return_grid_results=False,
        )
        test_evaluation = {
            "audio": get_best_evaluation(
                dataset_te,
                data_num_te,
                a_te,
                v_te,
                t_te,
                mode="audio",
                device=device,
                distance_fn=distance_fn,
                best_beta=None,
                best_tau=None,
                use_calibrated_stacking=False,
                calibration_mode="beta",
                save_performances=False,
                args=args,
                return_grid_results=False,
            ),
            "video": get_best_evaluation(
                dataset_te,
                data_num_te,
                a_te,
                v_te,
                t_te,
                mode="video",
                device=device,
                distance_fn=distance_fn,
                best_beta=None,
                best_tau=None,
                use_calibrated_stacking=False,
                calibration_mode="beta",
                save_performances=False,
                args=args,
                return_grid_results=False,
            ),
            "both": test_both_calibrated,
            "both_baseline_uncalibrated": test_baseline_both,
        }

        calibration_log["val"]["both_baseline_uncalibrated"] = {
            "seen": val_baseline_both["seen"],
            "unseen": val_baseline_both["unseen"],
            "hm": val_baseline_both["hm"],
            "zsl": val_baseline_both["zsl"],
        }
        calibration_log["val"]["grid_results_combined"] = grid_results
        calibration_log["val"]["best_on_val"] = {
            "tau": best_tau_combined,
            "beta": best_beta_combined,
            "seen": val_search_both["seen"],
            "unseen": val_search_both["unseen"],
            "hm": val_search_both["hm"],
            "zsl": val_search_both["zsl"],
        }
        calibration_log["test"]["fixed_tau_beta_from_val"] = {"tau": best_tau_combined, "beta": best_beta_combined}
        calibration_log["test"]["both_baseline_uncalibrated"] = {
            "seen": test_baseline_both["seen"],
            "unseen": test_baseline_both["unseen"],
            "hm": test_baseline_both["hm"],
            "zsl": test_baseline_both["zsl"],
        }
        calibration_log["test"]["both_calibrated"] = {
            "seen": test_both_calibrated["seen"],
            "unseen": test_both_calibrated["unseen"],
            "hm": test_both_calibrated["hm"],
            "zsl": test_both_calibrated["zsl"],
        }

        logger.info(
            "Validation combined (uncalibrated baseline): seen=%.4f unseen=%.4f HM=%.4f ZSL=%.4f",
            val_baseline_both["seen"],
            val_baseline_both["unseen"],
            val_baseline_both["hm"],
            val_baseline_both["zsl"],
        )
        logger.info(
            "Validation combined (best calibrated on val): tau=%s beta=%s seen=%.4f unseen=%.4f HM=%.4f ZSL=%.4f",
            best_tau_combined,
            best_beta_combined,
            val_search_both["seen"],
            val_search_both["unseen"],
            val_search_both["hm"],
            val_search_both["zsl"],
        )
        logger.info(
            "Test fixed params from val: tau=%s beta=%s",
            best_tau_combined,
            best_beta_combined,
        )
        logger.info(
            "Test combined (uncalibrated baseline): seen=%.4f unseen=%.4f HM=%.4f ZSL=%.4f",
            test_baseline_both["seen"],
            test_baseline_both["unseen"],
            test_baseline_both["hm"],
            test_baseline_both["zsl"],
        )
    else:
        dataset_v, data_num_v, a_v, v_v, t_v = collect_eval_embeddings(val_dataset, model_A, device, args)
        val_baseline_both = get_best_evaluation(
            dataset_v,
            data_num_v,
            a_v,
            v_v,
            t_v,
            mode="both",
            device=device,
            distance_fn=distance_fn,
            best_beta=None,
            best_tau=None,
            use_calibrated_stacking=False,
            calibration_mode="beta",
            save_performances=False,
            args=args,
            return_grid_results=False,
        )

        val_evaluation = evaluate_from_stacked_embeddings(
            dataset_v,
            data_num_v,
            a_v,
            v_v,
            t_v,
            device,
            distance_fn,
            best_beta=None,
            best_tau=None,
            save_performances=False,
            args=args,
            return_grid_results=False,
        )

        best_beta_combined = (1.0 / 3.0) * (
            val_evaluation["audio"]["beta"] + val_evaluation["video"]["beta"] + val_evaluation["both"]["beta"] + 1e-10
        )
        if cal_mode == "tau_beta":
            best_tau_combined = (1.0 / 3.0) * (
                val_evaluation["audio"]["tau"] + val_evaluation["video"]["tau"] + val_evaluation["both"]["tau"]
            )
        else:
            best_tau_combined = 1.0

        logger.info(
            "Validation betas:\tAudio=%s\tVideo=%s\tBoth=%s",
            val_evaluation["audio"]["beta"],
            val_evaluation["video"]["beta"],
            val_evaluation["both"]["beta"],
        )
        if cal_mode == "tau_beta":
            logger.info(
                "Validation taus:\tAudio=%s\tVideo=%s\tBoth=%s",
                val_evaluation["audio"]["tau"],
                val_evaluation["video"]["tau"],
                val_evaluation["both"]["tau"],
            )
        logger.info("Best beta combined (legacy average): %s", best_beta_combined)
        logger.info("Best tau combined (legacy): %s", best_tau_combined)

        dataset_te, data_num_te, a_te, v_te, t_te = collect_eval_embeddings(test_dataset, model_B, device, args)
        test_evaluation = evaluate_from_stacked_embeddings(
            dataset_te,
            data_num_te,
            a_te,
            v_te,
            t_te,
            device,
            distance_fn,
            best_beta=best_beta_combined,
            best_tau=best_tau_combined,
            save_performances=save_performances,
            args=args,
            return_grid_results=False,
        )
        test_baseline_both = get_best_evaluation(
            dataset_te,
            data_num_te,
            a_te,
            v_te,
            t_te,
            mode="both",
            device=device,
            distance_fn=distance_fn,
            best_beta=None,
            best_tau=None,
            use_calibrated_stacking=False,
            calibration_mode="beta",
            save_performances=False,
            args=args,
            return_grid_results=False,
        )
        test_evaluation["both_baseline_uncalibrated"] = test_baseline_both

        calibration_log["val"]["both_baseline_uncalibrated"] = {
            "seen": val_baseline_both["seen"],
            "unseen": val_baseline_both["unseen"],
            "hm": val_baseline_both["hm"],
            "zsl": val_baseline_both["zsl"],
        }
        calibration_log["val"]["per_modality_best_on_val"] = {
            "audio": {k: val_evaluation["audio"][k] for k in ("seen", "unseen", "hm", "zsl", "beta", "tau")},
            "video": {k: val_evaluation["video"][k] for k in ("seen", "unseen", "hm", "zsl", "beta", "tau")},
            "both": {k: val_evaluation["both"][k] for k in ("seen", "unseen", "hm", "zsl", "beta", "tau")},
        }
        calibration_log["val"]["legacy_averaged"] = {"tau": best_tau_combined, "beta": best_beta_combined}
        calibration_log["test"]["fixed_tau_beta_from_val"] = {"tau": best_tau_combined, "beta": best_beta_combined}
        calibration_log["test"]["both_baseline_uncalibrated"] = {
            "seen": test_baseline_both["seen"],
            "unseen": test_baseline_both["unseen"],
            "hm": test_baseline_both["hm"],
            "zsl": test_baseline_both["zsl"],
        }
        calibration_log["test"]["both_calibrated"] = {
            "seen": test_evaluation["both"]["seen"],
            "unseen": test_evaluation["both"]["unseen"],
            "hm": test_evaluation["both"]["hm"],
            "zsl": test_evaluation["both"]["zsl"],
        }

        logger.info(
            "Validation combined (uncalibrated baseline): seen=%.4f unseen=%.4f HM=%.4f ZSL=%.4f",
            val_baseline_both["seen"],
            val_baseline_both["unseen"],
            val_baseline_both["hm"],
            val_baseline_both["zsl"],
        )
        logger.info(
            "Test combined (uncalibrated baseline): seen=%.4f unseen=%.4f HM=%.4f ZSL=%.4f",
            test_baseline_both["seen"],
            test_baseline_both["unseen"],
            test_baseline_both["hm"],
            test_baseline_both["zsl"],
        )

    calibration_log["val"]["audio_video_uncalibrated_note"] = (
        "With search_on_both, audio/video entries are uncalibrated (argmin distance) on val for reference."
        if strategy == "search_on_both"
        else None
    )

    test_evaluation["calibration_log"] = calibration_log

    if save_json and eval_dir is not None:
        out_path = Path(eval_dir) / "calibration_stacking_log.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(calibration_log, f, indent=2)
        logger.info("Wrote calibration log to %s", out_path)

    return test_evaluation
