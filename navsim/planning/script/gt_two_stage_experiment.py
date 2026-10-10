import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from hydra.utils import instantiate
from nuplan.planning.utils.multithreading.worker_pool import Task
from omegaconf import OmegaConf

from navsim.common.dataclasses import SensorConfig, Trajectory
from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.script.gt_two_stage_diagnostics import summarize_branches


logger = logging.getLogger(__name__)


def checked_rows(rows, tokens):
    frame = pd.concat(rows, ignore_index=True)
    if frame.token.duplicated().any() or set(frame.token) != set(tokens) or not frame.valid.all():
        raise RuntimeError("Incomplete or failed PDM evaluation; refusing to compare different populations")
    if not np.isfinite(frame.pdm_score.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite candidate PDM score")
    return frame


def score_candidate_task(args):
    from navsim.planning.script.run_pdm_score import run_pdm_score

    task = args[0]
    proposals = task["candidate_proposals"]
    ground_truth = task["ground_truth"]
    sampling = instantiate(task["cfg"].agent.trajectory_sampling)
    model_indices = {token: int(np.argmax(scores)) for token, scores in task["candidate_scores"].items()}
    trajectories = {
        token: Trajectory(ground_truth[token] if token in ground_truth else proposals[token][model_indices[token]], sampling)
        for token in task["tokens"]
    }
    model = checked_rows(run_pdm_score([dict(task, model_trajectories=trajectories)]), task["tokens"])
    actual_rows = {row.token.iloc[0]: row.copy(deep=True) for _, row in model.groupby("token")}
    selected = {token: -1 if token in ground_truth else model_indices[token] for token in task["tokens"]}
    best_scores = model.set_index("token").pdm_score.to_dict()
    candidate_results = [
        {"token": token, "proposal_index": model_indices[token], "pdm_score": float(best_scores[token])}
        for token in task["tokens"] if token not in ground_truth
    ]
    candidate_count = next(iter(proposals.values())).shape[0]
    for proposal_index in range(candidate_count):
        pending = [token for token in task["tokens"]
                   if token not in ground_truth and model_indices[token] != proposal_index]
        if not pending:
            continue
        job = dict(task, tokens=pending, model_trajectories={
            token: Trajectory(proposals[token][proposal_index], sampling) for token in pending
        })
        job["original_scenes"] = {token: value for token, value in task["original_scenes"].items() if token in pending}
        job["synthetic_scenes"] = {token: value for token, value in task["synthetic_scenes"].items() if token in pending}
        evaluated = checked_rows(run_pdm_score([job]), pending)
        for token, row in evaluated.groupby("token"):
            score = float(row.pdm_score.iloc[0])
            candidate_results.append({"token": token, "proposal_index": proposal_index, "pdm_score": score})
            if score > best_scores[token] or (score == best_scores[token] and proposal_index < selected[token]):
                best_scores[token] = score
                selected[token] = proposal_index
                actual_rows[token] = row.copy(deep=True)
    return {
        "model": model,
        "actual": pd.concat(list(actual_rows.values()), ignore_index=True),
        "candidate_scores": candidate_results,
        "selection": [
            {"token": token, "has_gt": token in ground_truth,
             "model_index": -1 if token in ground_truth else model_indices[token],
             "actual_pdm_index": selected[token]}
            for token in task["tokens"]
        ],
    }


def load_candidates(cfg, tokens, sampling):
    path = Path(cfg.gt_two_stage.candidate_cache_path)
    with np.load(path, allow_pickle=False) as archive:
        cached_tokens = archive["tokens"].tolist()
        proposals = archive["proposals"]
        scores = archive["scores"]
        identity = json.loads(str(archive["identity"].item()))
    if len(set(cached_tokens)) != len(cached_tokens) or set(cached_tokens) != set(tokens):
        raise ValueError("Candidate cache must cover exactly the evaluation tokens")
    expected = (len(tokens), cfg.agent.config.proposal_num, sampling.num_poses, 3)
    if proposals.shape != expected or scores.shape != expected[:2]:
        raise ValueError("Candidate cache shape or sampling mismatch")
    if not np.isfinite(proposals).all() or not np.isfinite(scores).all():
        raise ValueError("Nonfinite candidates or model scores")
    if identity["agent"] != OmegaConf.to_container(cfg.agent, resolve=True):
        raise ValueError("Candidate cache agent configuration differs from this experiment")
    checkpoint = Path(cfg.agent.checkpoint_path).resolve()
    stat = checkpoint.stat()
    if identity["checkpoint"] != [str(checkpoint), stat.st_size, stat.st_mtime_ns]:
        raise ValueError("Candidate cache checkpoint identity mismatch")
    if identity["data_paths"] != {key: str(cfg[key]) for key in (
        "navsim_log_path", "original_sensor_path", "synthetic_sensor_path", "synthetic_scenes_path"
    )}:
        raise ValueError("Candidate cache dataset paths differ from this experiment")
    return dict(zip(cached_tokens, proposals)), dict(zip(cached_tokens, scores))


def load_ground_truth(cfg, loader, tokens, sampling):
    ground_truth = {}
    if cfg.gt_two_stage.gt_cache_path:
        with np.load(cfg.gt_two_stage.gt_cache_path, allow_pickle=False) as archive:
            if not np.isclose(float(archive["interval_length"].item()), sampling.interval_length):
                raise ValueError("GT sampling interval mismatch")
            cached_tokens = archive["tokens"].tolist()
            poses = archive["ground_truth"] if "ground_truth" in archive else archive["poses"]
            has_gt = archive["has_gt"] if "has_gt" in archive else np.ones(len(cached_tokens), dtype=bool)
            if len(set(cached_tokens)) != len(cached_tokens) or poses.shape != (len(cached_tokens), sampling.num_poses, 3):
                raise ValueError("GT cache requires unique tokens and poses [scenes, num_poses, 3]")
            if has_gt.shape != (len(cached_tokens),):
                raise ValueError("Invalid GT availability mask")
            ground_truth.update({token: pose.copy() for token, pose, valid in zip(cached_tokens, poses, has_gt)
                                 if valid and token in tokens})
    for token in tokens:
        if token not in ground_truth and token in loader.scene_frames_dicts:
            trajectory = loader.get_scene_from_token(token).get_future_trajectory(sampling.num_poses)
            if not np.isclose(trajectory.trajectory_sampling.interval_length, sampling.interval_length):
                raise ValueError("Scene GT sampling interval mismatch")
            ground_truth[token] = trajectory.poses
    if any(not np.isfinite(poses).all() for poses in ground_truth.values()):
        raise ValueError("Nonfinite available GT")
    return ground_truth


def run_gt_two_stage_experiment(cfg):
    from navsim.planning.script.run_pdm_score import (
        build_pdm_score_tasks, compute_final_scores, create_scene_aggregators,
    )

    if not cfg.gt_two_stage.candidate_cache_path:
        raise ValueError("gt_two_stage.candidate_cache_path is required")
    loader = SceneLoader(
        synthetic_sensor_path=None, original_sensor_path=None,
        data_path=Path(cfg.navsim_log_path), synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter), sensor_config=SensorConfig.build_no_sensors(),
    )
    tokens = sorted(loader.tokens)
    if not tokens or not set(tokens).issubset(set(MetricCacheLoader(Path(cfg.metric_cache_path)).tokens)):
        raise ValueError("Every selected scene must have a metric cache")
    available = set(tokens)
    mappings = {(original, previous): [tuple(pair) for pair in pairs]
                for original, previous, pairs in cfg.train_test_split.reactive_all_mapping
                if original in available or previous in available}
    mapped = {token for pair in mappings for token in pair}
    mapped.update(token for pairs in mappings.values() for pair in pairs for token in pair)
    if not mappings or mapped != available or any(not pairs for pairs in mappings.values()):
        raise ValueError("Complete two-stage mappings must cover exactly the selected scenes")
    sampling = instantiate(cfg.agent.trajectory_sampling)
    ground_truth = load_ground_truth(cfg, loader, tokens, sampling)
    if not {token for pair in mappings for token in pair}.issubset(ground_truth):
        raise ValueError("Every stage-one scene requires GT for this experiment")
    proposals, model_scores = load_candidates(cfg, tokens, sampling)
    jobs = build_pdm_score_tasks(cfg, loader, tokens, cfg.max_scenarios_per_task)
    for job in jobs:
        job["candidate_proposals"] = {token: proposals[token] for token in job["tokens"]}
        job["candidate_scores"] = {token: model_scores[token] for token in job["tokens"]}
        job["ground_truth"] = {token: ground_truth[token] for token in job["tokens"] if token in ground_truth}
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    logger.info("GT experiment: %d scenes, %d available GT, scoring every candidate for %d scenes",
                len(tokens), len(ground_truth), len(tokens) - len(ground_truth))
    results = build_worker(cfg).map(Task(fn=score_candidate_task), [[job] for job in jobs])
    pd.DataFrame([row for result in results for row in result["candidate_scores"]]).to_csv(
        output / "actual_candidate_scores.csv", index=False
    )
    pd.DataFrame([row for result in results for row in result["selection"]]).to_csv(output / "selection.csv", index=False)
    summary = {
        "gt_scenes": len(ground_truth), "total_scenes": len(tokens), "mappings": len(mappings),
        "actual_top1_objective": "per-scene pdm_score before two-frame extended comfort; lowest index breaks ties",
        "aggregation": "mean of stage-one times weighted stage-two per branch; two branches per mapping",
        "safety_counts": "causes may overlap; counts include zero-weight scenes; weighted mass also reported",
        "settings": {},
    }
    for setting in ("model", "actual"):
        raw = checked_rows([result[setting] for result in results], tokens)
        raw.to_pickle(output / f"gt_{setting}_top1.raw.pkl")
        final = compute_final_scores(create_scene_aggregators(
            mappings, raw.copy(deep=True), instantiate(cfg.simulator.proposal_sampling)
        ))
        if not np.isfinite(final.score.to_numpy(dtype=float)).all():
            raise ValueError("Nonfinite final scores")
        final.to_pickle(output / f"gt_{setting}_top1.pkl")
        final.drop(columns=["ego_simulated_states"], errors="ignore").to_csv(output / f"gt_{setting}_top1_scores.csv", index=False)
        statistics, branches, causes = summarize_branches(final, mappings)
        branches.to_csv(output / f"gt_{setting}_top1_branches.csv", index=False)
        causes.to_csv(output / f"gt_{setting}_top1_safety_zero_causes.csv", index=False)
        summary["settings"][f"gt_{setting}_top1"] = {
            "stage_two_without_gt_selection": "model_score_argmax" if setting == "model" else "actual_pdm_score_argmax",
            "stage_two_gt_scenes": sum(token in ground_truth for token in available - {token for pair in mappings for token in pair}),
            "groups": statistics,
        }
    model_groups = summary["settings"]["gt_model_top1"]["groups"]
    actual_groups = summary["settings"]["gt_actual_top1"]["groups"]
    if any(model_groups[group]["branches"] != actual_groups[group]["branches"] for group in model_groups):
        raise AssertionError("GT coverage groups must not change with stage-two selection")
    (output / "gt_two_stage_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    logger.info("GT two-stage experiment complete: %s", output)