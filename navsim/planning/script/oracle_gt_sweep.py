import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from nuplan.planning.utils.multithreading.worker_pool import Task
from omegaconf import OmegaConf

from navsim.common.dataclasses import Trajectory
from navsim.common.dataloader import SceneLoader
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.script.gpu_inference import predict_trajectories, predict_trajectories_multi_gpu
from navsim.planning.script.oracle_gt import calibrate_lambda, proposal_distance, select_proposal


logger = logging.getLogger(__name__)


def evaluation_mappings(cfg, tokens, expected_tokens):
    available = set(tokens)
    if available != set(expected_tokens):
        raise ValueError("Oracle sweep requires metric caches for every selected scene")
    mappings = {
        (original, previous): [tuple(pair) for pair in pairs]
        for original, previous, pairs in cfg.train_test_split.reactive_all_mapping
        if original in available or previous in available
    }
    if not mappings:
        raise ValueError("Oracle sweep requires complete two-stage mappings")
    mapped_tokens = {token for pair in mappings for token in pair}
    mapped_tokens.update(token for pairs in mappings.values() for pair in pairs for token in pair)
    if mapped_tokens != available:
        raise ValueError("Two-stage mappings must cover exactly the selected evaluation tokens")
    return mappings


def load_ground_truth(cfg, scene_loader, tokens, sampling):
    ground_truth = {}
    sources = {}
    if cfg.oracle_gt.gt_path:
        with np.load(cfg.oracle_gt.gt_path, allow_pickle=False) as archive:
            if not np.isclose(float(archive["interval_length"].item()), sampling.interval_length):
                raise ValueError("External GT interval differs from proposal interval")
            external_tokens = archive["tokens"].tolist()
            poses = archive["poses"]
            if len(set(external_tokens)) != len(external_tokens) or poses.shape != (len(external_tokens), sampling.num_poses, 3):
                raise ValueError("External GT requires unique tokens and poses [scenes, num_poses, 3]")
            if not np.isfinite(poses).all():
                raise ValueError("External GT must be finite")
            for token, trajectory in zip(external_tokens, poses):
                if token in tokens:
                    ground_truth[token] = trajectory.copy()
                    sources[token] = "external_gt"
    for token in tokens:
        if token not in ground_truth and token in scene_loader.scene_frames_dicts:
            trajectory = scene_loader.get_scene_from_token(token).get_future_trajectory(sampling.num_poses)
            if not np.isclose(trajectory.trajectory_sampling.interval_length, sampling.interval_length):
                raise ValueError(f"GT interval mismatch for {token}")
            ground_truth[token] = trajectory.poses
            sources[token] = "original_human_future"
    missing = sorted(set(tokens) - set(ground_truth))
    if cfg.oracle_gt.missing_gt not in ("error", "baseline"):
        raise ValueError("oracle_gt.missing_gt must be error or baseline")
    if missing and cfg.oracle_gt.missing_gt == "error":
        raise ValueError(
            f"{len(missing)}/{len(tokens)} scenes have no human GT (synthetic scenes do not provide it). "
            "Supply oracle_gt.gt_path, or explicitly set oracle_gt.missing_gt=baseline for a partial oracle. "
            f"First missing token: {missing[0]}"
        )
    if not ground_truth:
        raise ValueError("No GT available for calibration or reranking")
    logger.warning("Oracle GT coverage: %d/%d; %d scenes retain baseline", len(ground_truth), len(tokens), len(missing))
    return ground_truth, sources


def cache_identity(cfg, tokens):
    checkpoint = Path(cfg.agent.checkpoint_path).resolve()
    stat = checkpoint.stat()
    return {
        "version": 1,
        "agent": OmegaConf.to_container(cfg.agent, resolve=True),
        "checkpoint": [str(checkpoint), stat.st_size, stat.st_mtime_ns],
        "tokens": tokens,
        "data_paths": {key: str(cfg[key]) for key in (
            "navsim_log_path", "original_sensor_path", "synthetic_sensor_path", "synthetic_scenes_path"
        )},
    }


def load_or_predict_candidates(cfg, scene_loader, tokens, sampling):
    cache_path = Path(cfg.oracle_gt.cache_path) if cfg.oracle_gt.cache_path else Path(cfg.output_dir) / "oracle_candidates.npz"
    identity = cache_identity(cfg, tokens)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as archive:
            if json.loads(archive["identity"].item()) != identity:
                raise ValueError("Candidate cache configuration/checkpoint/tokens mismatch; use a new cache_path")
            proposals, scores = archive["proposals"], archive["scores"]
        logger.info("Reusing candidates from %s", cache_path)
    else:
        device = torch.device(cfg.gpu_device)
        if cfg.gpu_num_devices > 1:
            predictions = predict_trajectories_multi_gpu(cfg, scene_loader, tokens, device, cfg.gpu_num_devices)
        else:
            agent = instantiate(cfg.agent)
            agent.initialize()
            inference_loader = SceneLoader(
                synthetic_sensor_path=Path(cfg.synthetic_sensor_path),
                original_sensor_path=Path(cfg.original_sensor_path),
                data_path=Path(cfg.navsim_log_path),
                synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
                scene_filter=instantiate(cfg.train_test_split.scene_filter),
                sensor_config=agent.get_sensor_config(),
                synthetic_scene_index=scene_loader.synthetic_scenes,
                original_scene_index=scene_loader.scene_frames_dicts,
            )
            predictions = predict_trajectories(
                agent, inference_loader, tokens, device, cfg.gpu_batch_size, cfg.gpu_num_workers,
                return_proposals=True,
            )
            del agent
            torch.cuda.empty_cache()
        if set(predictions) != set(tokens):
            raise ValueError("Candidate predictions do not cover the evaluation tokens")
        proposals = np.stack([predictions[token]["proposals"] for token in tokens])
        scores = np.stack([predictions[token]["scores"] for token in tokens])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary_path, identity=json.dumps(identity), tokens=np.array(tokens), proposals=proposals, scores=scores)
        temporary_path.replace(cache_path)
    expected = (len(tokens), cfg.agent.config.proposal_num, sampling.num_poses, 3)
    if proposals.shape != expected or scores.shape != expected[:2]:
        raise ValueError(f"Unexpected proposal/score cache shapes: {proposals.shape}, {scores.shape}")
    if not np.isfinite(proposals).all() or not np.isfinite(scores).all():
        raise ValueError("Nonfinite proposal or log-score in cache")
    return proposals, scores


def evaluate_selection(cfg, scene_loader, tasks, worker, tokens, trajectories, output_path):
    from navsim.planning.script.run_pdm_score import (
        calculate_individual_mapping_scores, compute_final_scores, create_scene_aggregators, run_pdm_score,
    )

    jobs = [dict(task, model_trajectories={token: trajectories[token] for token in task["tokens"]}) for task in tasks]
    rows = [row for batch in worker.map(Task(fn=run_pdm_score), [[job] for job in jobs]) for row in batch]
    results = pd.concat(rows, ignore_index=True)
    results.to_pickle(output_path.with_suffix(".raw.pkl"))
    if results["token"].duplicated().any() or set(results["token"]) != set(tokens) or not results["valid"].all():
        raise RuntimeError("Incomplete/failed evaluation; refusing to compare different scene populations")
    mappings = evaluation_mappings(cfg, tokens, scene_loader.tokens)
    results = create_scene_aggregators(mappings, results, instantiate(cfg.simulator.proposal_sampling))
    results = compute_final_scores(results)
    combined, stage_one, stage_two = calculate_individual_mapping_scores(results[["score", "token", "weight"]], mappings)
    scores = [float(series["score"]) for series in (combined, stage_one, stage_two)]
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite two-stage aggregate scores")
    results.to_pickle(output_path.with_suffix(".pkl"))
    results.drop(columns=["ego_simulated_states"], errors="ignore").to_csv(output_path, index=False)
    return dict(combined_score=scores[0], stage_one_score=scores[1], stage_two_score=scores[2], valid_scenes=len(results))


def run_oracle_gt_sweep(cfg, scene_loader, tokens):
    from navsim.planning.script.run_pdm_score import build_pdm_score_tasks

    if not cfg.gpu_inference:
        raise ValueError("Oracle sweep requires gpu_inference=true")
    tokens = sorted(tokens)
    if not tokens:
        raise ValueError("No evaluation scenes")
    evaluation_mappings(cfg, tokens, scene_loader.tokens)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sampling = instantiate(cfg.agent.trajectory_sampling)
    ground_truth, sources = load_ground_truth(cfg, scene_loader, tokens, sampling)
    proposals, scores = load_or_predict_candidates(cfg, scene_loader, tokens, sampling)
    kind = cfg.oracle_gt.distance
    has_gt = np.array([token in ground_truth for token in tokens])
    distance = np.full(scores.shape, np.nan)
    gt_poses = np.full((len(tokens), sampling.num_poses, 3), np.nan)
    for index, token in enumerate(tokens):
        if has_gt[index]:
            gt_poses[index] = ground_truth[token]
            distance[index] = proposal_distance(proposals[index], gt_poses[index], kind)
    reference, thresholds = calibrate_lambda(scores[has_gt], distance[has_gt])
    np.savez_compressed(
        output_dir / f"oracle_{kind}.npz", tokens=np.array(tokens), distance=distance, ground_truth=gt_poses,
        has_gt=has_gt, gt_sources=np.array([sources.get(token, "missing_keep_baseline") for token in tokens]),
        interval_length=sampling.interval_length,
    )
    diagnostics = {
        "distance": kind, "lambda_reference": reference, "gt_scenes": int(has_gt.sum()), "total_scenes": len(tokens),
        "scope": "all_scenes" if has_gt.all() else "partial_oracle_missing_gt_keeps_baseline",
        "zero_switch_thresholds": int((thresholds == 0).sum()),
        "baseline_already_min_distance": int(np.isinf(thresholds).sum()),
        "score_quantiles": np.quantile(scores, [0, .1, .5, .9, 1]).tolist(),
        "scene_score_span_quantiles": np.quantile(np.ptp(scores, axis=1), [0, .1, .5, .9, 1]).tolist(),
        "distance_quantiles_m": np.quantile(distance[has_gt], [0, .1, .5, .9, 1]).tolist(),
        "positive_switch_quantiles": np.quantile(thresholds[np.isfinite(thresholds) & (thresholds > 0)], [0, .1, .5, .9, 1]).tolist() if reference is not None else [],
    }
    (output_dir / "oracle_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    logger.info("Oracle calibration: %s", diagnostics)
    if cfg.oracle_gt.prepare_only:
        return
    if cfg.oracle_gt.lambdas is not None:
        strengths = sorted(set([0.0] + [float(value) for value in cfg.oracle_gt.lambdas]))
    else:
        if reference is None:
            raise ValueError("No positive switch thresholds; set oracle_gt.lambdas explicitly")
        strengths = sorted(set([0.0] + [reference * float(value) for value in cfg.oracle_gt.lambda_multipliers]))
    if not all(np.isfinite(value) and value >= 0 for value in strengths):
        raise ValueError("Lambda grid must be finite and nonnegative")
    settings = [(f"lambda_{index:02d}", strength, False) for index, strength in enumerate(strengths)]
    if cfg.oracle_gt.include_min_distance:
        settings.append((f"min_{kind}", 0.0, True))
    tasks = build_pdm_score_tasks(cfg, scene_loader, tokens, cfg.max_scenarios_per_task)
    worker = build_worker(cfg)
    baseline = scores.argmax(axis=1)
    scene_indices = np.arange(len(tokens))
    previous_distance = distance[scene_indices[has_gt], baseline[has_gt]]
    summaries = []
    for label, strength, min_distance in settings:
        selected = baseline.copy()
        for index in np.flatnonzero(has_gt):
            selected[index] = select_proposal(scores[index], distance[index], strength, min_distance)
        selected_distance = distance[scene_indices[has_gt], selected[has_gt]]
        if (selected_distance > previous_distance + 1e-8).any():
            raise AssertionError(f"Selected {kind} must not increase with lambda")
        previous_distance = selected_distance
        pd.DataFrame({
            "token": tokens, "has_gt": has_gt, "baseline_index": baseline, "selected_index": selected,
            f"selected_{kind}_m": distance[scene_indices, selected], "selected_log_score": scores[scene_indices, selected],
        }).to_csv(output_dir / f"{label}_selection.csv", index=False)
        trajectories = {token: Trajectory(proposals[index, selected[index]], sampling) for index, token in enumerate(tokens)}
        logger.info("Evaluating %s: %s lambda=%s, changed=%d/%d", label, kind, strength, int((selected != baseline).sum()), len(tokens))
        metrics = evaluate_selection(cfg, scene_loader, tasks, worker, tokens, trajectories, output_dir / f"{label}_scores.csv")
        summary = dict(
            setting=label, distance=kind, lambda_value=None if min_distance else strength, lambda_reference=reference,
            scope=diagnostics["scope"], gt_scenes=int(has_gt.sum()), mean_selected_distance_m=float(selected_distance.mean()),
            changed_fraction=float((selected != baseline).mean()),
            changed_fraction_gt=float((selected[has_gt] != baseline[has_gt]).mean()), **metrics,
        )
        summary["delta_baseline"] = metrics["combined_score"] - (summaries[0]["combined_score"] if summaries else metrics["combined_score"])
        summaries.append(summary)
        pd.DataFrame(summaries).to_csv(output_dir / "oracle_summary.csv", index=False)
        logger.info("Oracle result: %s", summary)