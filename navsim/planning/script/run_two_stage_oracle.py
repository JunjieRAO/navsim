import hashlib
import json
import logging
import lzma
import multiprocessing
import os
import pickle
import random
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from omegaconf import DictConfig, OmegaConf

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SensorConfig, Trajectory
from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.evaluate.two_stage_oracle import (
    METRIC_COLUMNS,
    compare_oracles,
    evaluate_scene_proposals,
    evaluate_scene_proposals_with_history,
    gaussian_weights,
    proposal_endpoints,
    select_history_pairs,
    select_now_mappings,
    simulate_fixed_history,
    stage_two_oracles,
)
from navsim.planning.script.gpu_inference import (
    predict_proposals, predict_proposals_multi_gpu, predict_trajectories, predict_trajectories_multi_gpu,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex


CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_two_stage_oracle"
logger = logging.getLogger(__name__)


def score_scene_job(
    cache_path: Path,
    proposals: np.ndarray,
    cfg: DictConfig,
    history: Optional[Dict[str, np.ndarray]] = None,
    history_cache_path: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    before = cache_stamp(cache_path)
    if history is not None:
        if history_cache_path is None or not np.array_equal(history["source_stamp"], cache_stamp(history_cache_path)):
            raise ValueError("Fixed-history metric cache changed after simulation")
    with lzma.open(cache_path, "rb") as cache_file:
        metric_cache = pickle.load(cache_file)

    simulator = instantiate(cfg.simulator)
    scorer = instantiate(cfg.scorer)
    if simulator.proposal_sampling != scorer.proposal_sampling:
        raise ValueError("Simulator and scorer sampling must match")
    traffic_policy = instantiate(cfg.traffic_agents_policy.reactive, simulator.proposal_sampling)
    model_sampling = instantiate(cfg.agent.trajectory_sampling)
    if history is None:
        scores, metrics = evaluate_scene_proposals(
            metric_cache, proposals, model_sampling, simulator, scorer, traffic_policy,
        )
    else:
        scores, metrics, fixed_scores, comfort_scores = evaluate_scene_proposals_with_history(
            metric_cache, proposals, history["simulated_states"], float(history["start_time"]),
            model_sampling, simulator, scorer, traffic_policy,
        )
    if not np.array_equal(before, cache_stamp(cache_path)):
        raise RuntimeError(f"Metric cache changed while evaluating {cache_path}")
    if history is not None and not np.array_equal(history["source_stamp"], cache_stamp(history_cache_path)):
        raise RuntimeError("Fixed-history metric cache changed while evaluating proposals")

    rear_axle = metric_cache.ego_state.rear_axle
    result = {
        "proposals": proposals,
        "scores": scores,
        "metrics": np.array([[row[column] for column in METRIC_COLUMNS] for row in metrics], dtype=np.float64),
        "endpoints": proposal_endpoints(proposals, rear_axle),
        "start": np.array([rear_axle.x, rear_axle.y], dtype=np.float64),
    }
    if history is not None:
        result.update(fixed_ec_scores=fixed_scores, extended_comfort=comfort_scores)
    return result


def score_history_job(cache_path: Path, selected_trajectory: Trajectory, cfg: DictConfig) -> Dict[str, np.ndarray]:
    before = cache_stamp(cache_path)
    with lzma.open(cache_path, "rb") as cache_file:
        metric_cache = pickle.load(cache_file)

    simulator = instantiate(cfg.simulator)
    sampling = instantiate(cfg.agent.trajectory_sampling)
    poses = np.asarray(selected_trajectory.poses)
    if selected_trajectory.trajectory_sampling != sampling or poses.shape != (sampling.num_poses, 3) or not np.isfinite(poses).all():
        raise ValueError("Invalid DrivoR top-1 trajectory for fixed history")
    simulated = simulate_fixed_history(metric_cache, selected_trajectory, simulator)
    if not np.array_equal(before, cache_stamp(cache_path)):
        raise RuntimeError(f"Metric cache changed while simulating fixed history {cache_path}")
    return {
        "selected_trajectory": poses,
        "simulated_states": simulated,
        "start_time": np.array(metric_cache.timepoint.time_s, dtype=np.float64),
    }


def cache_stamp(cache_path: Path) -> np.ndarray:
    stat = cache_path.stat()
    return np.array([stat.st_size, stat.st_mtime_ns], dtype=np.int64)


def token_artifact_path(output_dir: Path, token: str) -> Path:
    if not token or not all(character.isalnum() or character in "-_" for character in token):
        raise ValueError(f"Invalid scene token: {token}")
    return output_dir / "tokens" / f"{token}.npz"


def save_token_artifact(
    output_dir: Path,
    token: str,
    result: Dict[str, np.ndarray],
    fingerprint: str,
    cache_path: Path,
    history_token: Optional[str] = None,
    history_cache_path: Optional[Path] = None,
) -> None:
    path = token_artifact_path(output_dir, token)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{token}-", suffix=".npz", delete=False) as temp:
        temp_path = Path(temp.name)
        try:
            metadata = {"fingerprint": fingerprint, "cache_stamp": cache_stamp(cache_path)}
            if history_token is not None and history_cache_path is not None:
                metadata.update(history_token=history_token, history_cache_stamp=cache_stamp(history_cache_path))
            np.savez_compressed(temp, **result, **metadata)
            temp.flush()
            os.fsync(temp.fileno())
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
    os.replace(temp_path, path)


def load_token_artifact(
    output_dir: Path,
    token: str,
    fingerprint: str,
    cache_path: Path,
    model_sampling: TrajectorySampling,
    history_token: Optional[str] = None,
    history_cache_path: Optional[Path] = None,
) -> Optional[Dict[str, np.ndarray]]:
    path = token_artifact_path(output_dir, token)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            if archive["fingerprint"].item() != fingerprint or not np.array_equal(
                archive["cache_stamp"], cache_stamp(cache_path)
            ):
                return None
            keys = ("proposals", "scores", "metrics", "endpoints", "start")
            if history_token is not None and history_cache_path is not None:
                if archive["history_token"].item() != history_token or not np.array_equal(
                    archive["history_cache_stamp"], cache_stamp(history_cache_path)
                ):
                    return None
                keys += ("fixed_ec_scores", "extended_comfort")
            result = {key: archive[key] for key in keys}
        expected_shapes = {
            "proposals": (64, model_sampling.num_poses, 3),
            "scores": (64,),
            "metrics": (64, len(METRIC_COLUMNS)),
            "endpoints": (64, 2),
            "start": (2,),
        }
        if history_token is not None and history_cache_path is not None:
            expected_shapes.update(fixed_ec_scores=(64,), extended_comfort=(64,))
        if any(result[key].shape != shape or not np.isfinite(result[key]).all() for key, shape in expected_shapes.items()):
            return None
        return result
    except (OSError, ValueError, KeyError):
        return None


def history_artifact_path(output_dir: Path, token: str) -> Path:
    return output_dir / "histories" / token_artifact_path(output_dir, token).name


def save_history_artifact(
    output_dir: Path,
    token: str,
    result: Dict[str, np.ndarray],
    fingerprint: str,
    cache_path: Path,
) -> None:
    write_npz_atomic(history_artifact_path(output_dir, token), {
        **result, "fingerprint": np.asarray(fingerprint), "cache_stamp": cache_stamp(cache_path),
    })


def load_history_artifact(
    output_dir: Path,
    token: str,
    fingerprint: str,
    cache_path: Path,
    model_sampling: TrajectorySampling,
    evaluator_sampling: TrajectorySampling,
) -> Optional[Dict[str, np.ndarray]]:
    path = history_artifact_path(output_dir, token)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            if archive["fingerprint"].item() != fingerprint or not np.array_equal(
                archive["cache_stamp"], cache_stamp(cache_path)
            ):
                return None
            result = {key: archive[key] for key in ("selected_trajectory", "simulated_states", "start_time")}
        expected_shapes = {
            "selected_trajectory": (model_sampling.num_poses, 3),
            "simulated_states": (evaluator_sampling.num_poses + 1, StateIndex.size()),
            "start_time": (),
        }
        if any(result[key].shape != shape or not np.isfinite(result[key]).all() for key, shape in expected_shapes.items()):
            return None
        return result
    except (OSError, ValueError, KeyError):
        return None


def configuration_fingerprint(cfg: DictConfig) -> str:
    source_paths = [
        str(Path(cfg.navsim_log_path).resolve()),
        str(Path(cfg.original_sensor_path).resolve()),
        str(Path(cfg.synthetic_sensor_path).resolve()),
        str(Path(cfg.synthetic_scenes_path).resolve()),
        str(Path(cfg.metric_cache_path).resolve()),
    ]
    settings = {
        "mode": f"stage-two-max-now-{cfg.oracle_ec_mode}-v1",
        "agent": OmegaConf.to_container(cfg.agent, resolve=True),
        "scorer": OmegaConf.to_container(cfg.scorer, resolve=True),
        "simulator": OmegaConf.to_container(cfg.simulator, resolve=True),
        "traffic_policy": OmegaConf.to_container(cfg.traffic_agents_policy.reactive, resolve=True),
        "split": OmegaConf.to_container(cfg.train_test_split, resolve=True),
        "data_paths": source_paths,
        "seed": cfg.oracle_seed,
        "sigma_squared": cfg.oracle_sigma_squared,
        "gpu_device": cfg.gpu_device,
        "gpu_batch_size": cfg.gpu_batch_size,
        "gpu_num_devices": cfg.gpu_num_devices,
    }
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode("utf-8"))

    workspace_root = Path(__file__).resolve().parents[3]
    source_files = [
        "navsim/planning/script/run_two_stage_oracle.py",
        "navsim/planning/script/gpu_inference.py",
        "navsim/evaluate/two_stage_oracle.py",
        "navsim/evaluate/pdm_score.py",
        "navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py",
        "navsim/planning/simulation/planner/pdm_planner/scoring/scene_aggregator.py",
        "navsim/planning/simulation/planner/pdm_planner/simulation/pdm_simulator.py",
        "navsim/agents/drivoR/drivor_model.py",
        "navsim/traffic_agents_policies/navsim_IDM_traffic_agents.py",
    ]
    external_files = [
        Path(cfg.agent.checkpoint_path),
        Path(cfg.agent.config.image_backbone.model_weights),
        Path(cfg.agent.config.lidar_backbone.model_weights),
        *sorted((Path(cfg.metric_cache_path) / "metadata").glob("*.csv")),
    ]
    if not list((Path(cfg.metric_cache_path) / "metadata").glob("*.csv")):
        raise FileNotFoundError("Metric cache metadata CSV is missing")
    for path in dict.fromkeys([workspace_root / name for name in source_files] + external_files):
        digest.update(str(path.resolve()).encode("utf-8"))
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as temp:
        temp_path = Path(temp.name)
        try:
            json.dump(data, temp, indent=2, ensure_ascii=True)
            temp.flush()
            os.fsync(temp.fileno())
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
    os.replace(temp_path, path)


def verify_manifest(output_dir: Path, fingerprint: str, ec_mode: str = "none") -> None:
    path = output_dir / "manifest.json"
    if path.exists():
        with path.open(encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        if manifest.get("fingerprint") != fingerprint:
            raise ValueError("Oracle output_dir already contains a different configuration; select a new output_dir")
    else:
        write_json_atomic(path, {
            "fingerprint": fingerprint,
            "metric": "EPDMS-no-EC" if ec_mode == "none" else "EPDMS-no-EC and EPDMS-fixed-history-EC",
            "ec_mode": ec_mode,
            "branch": "now",
            "num_proposals": 64,
        })


def aggregate_mapping_results(
    mappings: List[Tuple[str, List[str]]],
    token_results: Dict[str, Dict[str, np.ndarray]],
    token_errors: Dict[str, str],
    proposal_sampling: TrajectorySampling,
    sigma_squared: float,
    score_key: str = "scores",
    shared_weights: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Dict[str, np.ndarray]]]:
    unique_followups = list(dict.fromkeys(token for _, followups in mappings for token in followups))
    stage_two_best = {}
    for token in unique_followups:
        if token in token_results:
            maxima, indices = stage_two_oracles(token_results[token][score_key][None])
            stage_two_best[token] = (float(maxima[0]), int(indices[0]))

    rows: List[Dict[str, Any]] = []
    invalid: List[Dict[str, Any]] = []
    details: Dict[str, Dict[str, np.ndarray]] = {}
    for original_token, followup_tokens in mappings:
        required_tokens = [original_token] + followup_tokens
        missing = [token for token in required_tokens if token not in token_results]
        if missing:
            invalid.append({
                "original_token": original_token,
                "missing_tokens": missing,
                "errors": {token: token_errors.get(token, "No evaluation result") for token in missing},
            })
            continue

        try:
            stage_one = token_results[original_token]
            stage_two_scores = np.array([stage_two_best[token][0] for token in followup_tokens])
            stage_two_indices = np.array([stage_two_best[token][1] for token in followup_tokens])
            starts = np.stack([token_results[token]["start"] for token in followup_tokens])
            weights = (
                shared_weights[original_token] if shared_weights is not None
                else gaussian_weights(stage_one["endpoints"], followup_tokens, starts, proposal_sampling, sigma_squared)
            )
            stage_one_scores = stage_one[score_key]
            comparison = compare_oracles(stage_one_scores, stage_two_scores, weights)
            distances_squared = ((stage_one["endpoints"][:, None] - starts[None]) ** 2).sum(axis=2)
            gaussian_sums = np.exp(-distances_squared / (2 * sigma_squared)).sum(axis=1)
            fallback_rows = int(np.count_nonzero(np.isclose(gaussian_sums, 0.0) | np.isnan(gaussian_sums)))
            short_index = comparison["short_index"]
            long_index = comparison["long_index"]
            ranked_stage_one = np.sort(stage_one_scores)
            rows.append({
                "original_token": original_token,
                "followup_count": len(followup_tokens),
                "short_index": short_index,
                "long_index": long_index,
                "E1_short": float(stage_one_scores[short_index]),
                "E1_long": float(stage_one_scores[long_index]),
                "V2_short": float(comparison["downstream_values"][short_index]),
                "V2_long": float(comparison["downstream_values"][long_index]),
                "T_short": comparison["short_score"],
                "T_long": comparison["long_score"],
                "gain": comparison["gain"],
                "disagreement": comparison["disagreement"],
                "E1_top_gap": float(ranked_stage_one[-1] - ranked_stage_one[-2]),
                "uniform_fallback_rows": fallback_rows,
                "mean_nearest_followup_distance_m": float(np.sqrt(distances_squared.min(axis=1)).mean()),
            })
            details[original_token] = {
                "followup_tokens": np.asarray(followup_tokens),
                "stage_two_oracle_scores": stage_two_scores,
                "stage_two_oracle_indices": stage_two_indices,
                "stage_one_scores": stage_one_scores,
                "stage_one_endpoints": stage_one["endpoints"],
                "stage_two_starts": starts,
                "weights": weights,
                "downstream_values": comparison["downstream_values"],
                "combined_scores": stage_one_scores * comparison["downstream_values"],
            }
        except (KeyError, ValueError, IndexError) as error:
            logger.exception("Unable to aggregate original scene %s", original_token)
            invalid.append({"original_token": original_token, "errors": {original_token: str(error)}})

    return rows, invalid, details


def inference_scene_loader(cfg: DictConfig, scene_loader: SceneLoader, agent: AbstractAgent) -> SceneLoader:
    return SceneLoader(
        synthetic_sensor_path=Path(cfg.synthetic_sensor_path),
        original_sensor_path=Path(cfg.original_sensor_path),
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=agent.get_sensor_config(),
        synthetic_scene_index=scene_loader.synthetic_scenes,
        original_scene_index=scene_loader.scene_frames_dicts,
    )


def infer_proposals(cfg: DictConfig, scene_loader: SceneLoader, tokens: List[str]) -> Dict[str, np.ndarray]:
    device = torch.device(cfg.gpu_device)
    if cfg.gpu_num_devices > 1:
        return predict_proposals_multi_gpu(cfg, scene_loader, tokens, device, cfg.gpu_num_devices)

    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    try:
        return predict_proposals(
            agent, inference_scene_loader(cfg, scene_loader, agent), tokens, device,
            cfg.gpu_batch_size, cfg.gpu_num_workers,
        )
    finally:
        del agent
        torch.cuda.empty_cache()


def infer_previous_trajectories(cfg: DictConfig, scene_loader: SceneLoader, tokens: List[str]) -> Dict[str, Trajectory]:
    device = torch.device(cfg.gpu_device)
    if cfg.gpu_num_devices > 1:
        return predict_trajectories_multi_gpu(cfg, scene_loader, tokens, device, cfg.gpu_num_devices)

    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    try:
        return predict_trajectories(
            agent, inference_scene_loader(cfg, scene_loader, agent), tokens, device,
            cfg.gpu_batch_size, cfg.gpu_num_workers,
        )
    finally:
        del agent
        torch.cuda.empty_cache()


def evaluate_history_group(
    tokens: List[str],
    cfg: DictConfig,
    scene_loader: SceneLoader,
    cache_paths: Dict[str, Path],
    output_dir: Path,
    fingerprint: str,
    model_sampling: TrajectorySampling,
    evaluator_sampling: TrajectorySampling,
    history_results: Dict[str, Dict[str, np.ndarray]],
    token_errors: Dict[str, str],
) -> None:
    pending = []
    available = set(scene_loader.tokens)
    for token in tokens:
        if token in history_results:
            continue
        if token not in available or token not in cache_paths:
            token_errors[token] = "Missing fixed-history scene or metric cache"
            continue
        cache_path = Path(cache_paths[token])
        try:
            cached = load_history_artifact(
                output_dir, token, fingerprint, cache_path, model_sampling, evaluator_sampling,
            )
        except OSError as error:
            token_errors[token] = str(error)
            continue
        if cached is not None:
            cached["source_stamp"] = cache_stamp(cache_path)
            history_results[token] = cached
        else:
            pending.append(token)

    if not pending:
        return
    logger.info("Inferring %s uncached fixed-history scenes", len(pending))
    try:
        trajectories = infer_previous_trajectories(cfg, scene_loader, pending)
        if set(trajectories) != set(pending):
            raise ValueError("GPU inference did not return all fixed-history tokens")
    except Exception as error:
        logger.exception("Unable to infer fixed-history trajectories")
        token_errors.update({token: f"Fixed-history inference failed: {error}" for token in pending})
        return

    worker_cfg = OmegaConf.create({
        "simulator": OmegaConf.to_container(cfg.simulator, resolve=True),
        "agent": {"trajectory_sampling": OmegaConf.to_container(cfg.agent.trajectory_sampling, resolve=True)},
    })

    def record_result(token: str, result: Dict[str, np.ndarray]) -> None:
        expected_shapes = {
            "selected_trajectory": (model_sampling.num_poses, 3),
            "simulated_states": (evaluator_sampling.num_poses + 1, StateIndex.size()),
            "start_time": (),
        }
        if any(result[key].shape != shape or not np.isfinite(result[key]).all() for key, shape in expected_shapes.items()):
            raise ValueError(f"Incomplete fixed-history simulation for {token}")
        cache_path = Path(cache_paths[token])
        save_history_artifact(output_dir, token, result, fingerprint, cache_path)
        result["source_stamp"] = cache_stamp(cache_path)
        history_results[token] = result

    if cfg.oracle_workers == 1:
        for token in pending:
            try:
                record_result(token, score_history_job(Path(cache_paths[token]), trajectories[token], worker_cfg))
            except Exception as error:
                logger.exception("Unable to simulate fixed-history %s", token)
                token_errors[token] = f"Fixed-history simulation failed: {error}"
    else:
        with ProcessPoolExecutor(max_workers=cfg.oracle_workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {
                pool.submit(score_history_job, Path(cache_paths[token]), trajectories[token], worker_cfg): token
                for token in pending
            }
            for future in as_completed(futures):
                token = futures[future]
                try:
                    record_result(token, future.result())
                except Exception as error:
                    logger.exception("Unable to simulate fixed-history %s", token)
                    token_errors[token] = f"Fixed-history simulation failed: {error}"


def evaluate_token_group(
    tokens: List[str],
    cfg: DictConfig,
    scene_loader: SceneLoader,
    cache_paths: Dict[str, Path],
    output_dir: Path,
    fingerprint: str,
    model_sampling: TrajectorySampling,
    token_results: Dict[str, Dict[str, np.ndarray]],
    token_errors: Dict[str, str],
    history_pairs: Optional[Dict[str, str]] = None,
    history_results: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
) -> None:
    pending = []
    available = set(scene_loader.tokens)
    for token in tokens:
        if token in token_results:
            continue
        if token not in available or token not in cache_paths:
            token_errors[token] = "Missing scene or metric cache"
            continue
        history_token = history_pairs[token] if history_pairs is not None else None
        if history_token is not None and (
            history_results is None or history_token not in history_results or history_token not in cache_paths
        ):
            token_errors[token] = f"Missing fixed-history simulation for {history_token}"
            continue
        cache_path = Path(cache_paths[token])
        history_cache_path = Path(cache_paths[history_token]) if history_token is not None else None
        try:
            cached = load_token_artifact(
                output_dir, token, fingerprint, cache_path, model_sampling, history_token, history_cache_path,
            )
        except OSError as error:
            token_errors[token] = str(error)
            continue
        if cached is not None:
            token_results[token] = cached
        else:
            pending.append(token)

    if not pending:
        return
    logger.info("Inferring %s uncached scenes", len(pending))
    try:
        proposals = infer_proposals(cfg, scene_loader, pending)
        if set(proposals) != set(pending):
            raise ValueError("GPU inference did not return all requested tokens")
    except Exception as error:
        logger.exception("Unable to infer proposal group")
        token_errors.update({token: f"Inference failed: {error}" for token in pending})
        return

    worker_cfg = OmegaConf.create({
        "simulator": OmegaConf.to_container(cfg.simulator, resolve=True),
        "scorer": OmegaConf.to_container(cfg.scorer, resolve=True),
        "traffic_agents_policy": {
            "reactive": OmegaConf.to_container(cfg.traffic_agents_policy.reactive, resolve=True),
        },
        "agent": {"trajectory_sampling": OmegaConf.to_container(cfg.agent.trajectory_sampling, resolve=True)},
    })

    def record_result(token: str, result: Dict[str, np.ndarray]) -> None:
        cache_path = Path(cache_paths[token])
        history_token = history_pairs[token] if history_pairs is not None else None
        history_cache_path = Path(cache_paths[history_token]) if history_token is not None else None
        if (
            result["scores"].shape != (64,)
            or result["metrics"].shape != (64, len(METRIC_COLUMNS))
            or result["endpoints"].shape != (64, 2)
            or result["start"].shape != (2,)
            or result["proposals"].shape != (64, model_sampling.num_poses, 3)
            or any(not np.isfinite(values).all() for values in result.values())
            or (history_token is not None and (
                result["fixed_ec_scores"].shape != (64,) or result["extended_comfort"].shape != (64,)
            ))
        ):
            raise ValueError(f"Incomplete evaluator result for {token}")
        if history_token is not None and not np.array_equal(
            history_results[history_token]["source_stamp"], cache_stamp(history_cache_path),
        ):
            raise RuntimeError(f"Fixed-history cache changed while evaluating {token}")
        save_token_artifact(output_dir, token, result, fingerprint, cache_path, history_token, history_cache_path)
        token_results[token] = result

    def submit_args(token: str) -> Tuple[Any, ...]:
        history_token = history_pairs[token] if history_pairs is not None else None
        if history_token is None:
            return (Path(cache_paths[token]), proposals[token], worker_cfg)
        return (
            Path(cache_paths[token]), proposals[token], worker_cfg, history_results[history_token],
            Path(cache_paths[history_token]),
        )

    if cfg.oracle_workers == 1:
        for token in pending:
            try:
                record_result(token, score_scene_job(*submit_args(token)))
            except Exception as error:
                logger.exception("Unable to evaluate %s", token)
                token_errors[token] = f"Evaluation failed: {error}"
    else:
        with ProcessPoolExecutor(
            max_workers=cfg.oracle_workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = {
                pool.submit(score_scene_job, *submit_args(token)): token
                for token in pending
            }
            for future in as_completed(futures):
                token = futures[future]
                try:
                    record_result(token, future.result())
                except Exception as error:
                    logger.exception("Unable to evaluate %s", token)
                    token_errors[token] = f"Evaluation failed: {error}"


MAPPING_COLUMNS = (
    "original_token", "followup_count", "short_index", "long_index", "E1_short", "E1_long",
    "V2_short", "V2_long", "T_short", "T_long", "gain", "disagreement", "E1_top_gap",
    "uniform_fallback_rows", "mean_nearest_followup_distance_m",
)


def summarize_results(
    rows: List[Dict[str, Any]],
    invalid: List[Dict[str, Any]],
    required_tokens: List[str],
    token_results: Dict[str, Dict[str, np.ndarray]],
    token_errors: Dict[str, str],
    history_results: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
) -> Dict[str, Any]:
    complete_tokens = set(token_results) | set(history_results or {})
    summary: Dict[str, Any] = {
        "metric": "EPDMS-no-EC",
        "branch": "now",
        "attempted_mappings": len(rows) + len(invalid),
        "valid_mappings": len(rows),
        "invalid_mappings": len(invalid),
        "required_tokens": len(required_tokens),
        "complete_tokens": len(complete_tokens),
        "failed_tokens": len(set(required_tokens) - complete_tokens),
        "failed_token_details": token_errors,
        "complete_proposal_scores": 64 * len(token_results),
    }
    if history_results is not None:
        summary["complete_fixed_histories"] = len(history_results)
    if rows:
        gains = np.array([row["gain"] for row in rows])
        summary.update({
            "disagreement_rate": float(np.mean([row["disagreement"] for row in rows])),
            "mean_T_short": float(np.mean([row["T_short"] for row in rows])),
            "mean_T_long": float(np.mean([row["T_long"] for row in rows])),
            "mean_gain": float(gains.mean()),
            "median_gain": float(np.median(gains)),
            "p90_gain": float(np.percentile(gains, 90)),
            "positive_gain_rate": float(np.mean(gains > 1e-8)),
            "near_tie_E1_rate": float(np.mean([row["E1_top_gap"] <= 1e-8 for row in rows])),
            "uniform_fallback_rows": int(sum(row["uniform_fallback_rows"] for row in rows)),
            "mean_nearest_followup_distance_m": float(np.mean([
                row["mean_nearest_followup_distance_m"] for row in rows
            ])),
        })
    return summary


def write_npz_atomic(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".oracle-", suffix=".npz", delete=False) as temp:
        temp_path = Path(temp.name)
        try:
            np.savez_compressed(temp, **arrays)
            temp.flush()
            os.fsync(temp.fileno())
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
    os.replace(temp_path, path)


def write_report(
    output_dir: Path,
    rows: List[Dict[str, Any]],
    invalid: List[Dict[str, Any]],
    details: Dict[str, Dict[str, np.ndarray]],
    summary: Dict[str, Any],
    token_errors: Dict[str, str],
    fixed_rows: Optional[List[Dict[str, Any]]] = None,
    fixed_details: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    fixed_summary: Optional[Dict[str, Any]] = None,
) -> None:
    if any(value is None for value in (fixed_rows, fixed_details, fixed_summary)) != all(
        value is None for value in (fixed_rows, fixed_details, fixed_summary)
    ):
        raise ValueError("Fixed-history report requires both rows, details and summary")
    if fixed_rows is not None and (
        set(details) != set(fixed_details)
        or [row["original_token"] for row in rows] != [row["original_token"] for row in fixed_rows]
    ):
        raise ValueError("No-EC and fixed-history results must cover the same mappings")

    for token, arrays in details.items():
        artifact = dict(arrays)
        if fixed_details is not None:
            for key in (
                "stage_two_oracle_scores", "stage_two_oracle_indices", "stage_one_scores",
                "downstream_values", "combined_scores",
            ):
                artifact[f"fixed_{key}"] = fixed_details[token][key]
        write_npz_atomic(output_dir / "mappings" / f"{token}.npz", artifact)

    write_csv_atomic(output_dir / "mappings.csv", rows)
    write_json_atomic(output_dir / "summary.json", summary)
    if fixed_rows is not None:
        write_csv_atomic(output_dir / "mappings_fixed_history.csv", fixed_rows)
        write_json_atomic(output_dir / "summary_fixed_history.json", fixed_summary)
        fixed_by_token = {row["original_token"]: row for row in fixed_rows}
        comparisons = [
            {
                "original_token": token,
                "short_index_no_ec": row["short_index"],
                "short_index_fixed_history": fixed_by_token[token]["short_index"],
                "long_index_no_ec": row["long_index"],
                "long_index_fixed_history": fixed_by_token[token]["long_index"],
                "gain_no_ec": row["gain"],
                "gain_fixed_history": fixed_by_token[token]["gain"],
                "stage_two_oracle_changes": int(np.count_nonzero(
                    details[token]["stage_two_oracle_indices"] != fixed_details[token]["stage_two_oracle_indices"]
                )),
            }
            for row in rows
            for token in (row["original_token"],)
        ]
        write_csv_atomic(output_dir / "comparisons.csv", comparisons)
    write_json_atomic(output_dir / "failures.json", {"invalid_mappings": invalid, "token_errors": token_errors})


def write_csv_atomic(path: Path, rows: List[Dict[str, Any]]) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as temp:
        temp_path = Path(temp.name)
        try:
            pd.DataFrame(rows, columns=MAPPING_COLUMNS if path.name.startswith("mappings") else None).to_csv(
                temp, index=False,
            )
            temp.flush()
            os.fsync(temp.fileno())
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
    os.replace(temp_path, path)


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    if cfg.oracle_workers < 1 or cfg.oracle_sigma_squared <= 0:
        raise ValueError("oracle_workers and oracle_sigma_squared must be positive")
    if cfg.oracle_ec_mode not in ("none", "fixed_history"):
        raise ValueError("oracle_ec_mode must be none or fixed_history")
    device = torch.device(cfg.gpu_device)
    if (
        device.type != "cuda"
        or not torch.cuda.is_available()
        or cfg.gpu_num_devices < 1
        or (device.index or 0) + cfg.gpu_num_devices > torch.cuda.device_count()
    ):
        raise RuntimeError("Oracle inference requires the requested number of CUDA devices")
    for device_index in range(device.index or 0, (device.index or 0) + cfg.gpu_num_devices):
        major, minor = torch.cuda.get_device_capability(device_index)
        if f"sm_{major}{minor}" not in torch.cuda.get_arch_list():
            raise RuntimeError(
                f"PyTorch does not support CUDA device {device_index} (sm_{major}{minor}); "
                "use a compatible PYTHON_BIN"
            )

    random.seed(cfg.oracle_seed)
    np.random.seed(cfg.oracle_seed)
    torch.manual_seed(cfg.oracle_seed)
    mappings = select_now_mappings(cfg.train_test_split.reactive_all_mapping, cfg.oracle_max_mappings)
    history_pairs = (
        select_history_pairs(cfg.train_test_split.reactive_all_mapping, cfg.oracle_max_mappings)
        if cfg.oracle_ec_mode == "fixed_history" else None
    )
    selected_mapping = cfg.train_test_split.reactive_all_mapping[: cfg.oracle_max_mappings or None]
    stage_one_tokens = [token for token, _ in mappings]
    stage_two_tokens = list(dict.fromkeys(token for _, followups in mappings for token in followups))
    stage_one_previous = list(dict.fromkeys(entry[1] for entry in selected_mapping)) if history_pairs else []
    stage_two_previous = list(dict.fromkeys(pair[1] for entry in selected_mapping for pair in entry[2])) if history_pairs else []
    if set(stage_one_tokens) & set(stage_two_tokens):
        raise ValueError("Original now scenes and follow-up now scenes must be distinct")
    required_tokens = list(dict.fromkeys(stage_two_tokens + stage_one_tokens + stage_two_previous + stage_one_previous))
    output_dir = Path(cfg.output_dir)
    fingerprint = configuration_fingerprint(cfg)
    verify_manifest(output_dir, fingerprint, cfg.oracle_ec_mode)

    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.tokens = list(dict.fromkeys(stage_one_tokens + stage_one_previous))
    scene_filter.synthetic_scene_tokens = list(dict.fromkeys(stage_two_tokens + stage_two_previous))
    scene_filter.max_scenes = None
    scene_loader = SceneLoader(
        synthetic_sensor_path=Path(cfg.synthetic_sensor_path),
        original_sensor_path=Path(cfg.original_sensor_path),
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    model_sampling = instantiate(cfg.agent.trajectory_sampling)
    token_results: Dict[str, Dict[str, np.ndarray]] = {}
    history_results: Dict[str, Dict[str, np.ndarray]] = {}
    token_errors: Dict[str, str] = {}
    logger.info("Oracle evaluation (%s): %d mappings, %d follow-up scenes, %d previous scenes",
                cfg.oracle_ec_mode, len(mappings), len(stage_two_tokens), len(stage_one_previous) + len(stage_two_previous))

    for stage_name, tokens, previous_tokens in (
        ("stage-two", stage_two_tokens, stage_two_previous),
        ("stage-one", stage_one_tokens, stage_one_previous),
    ):
        if history_pairs is not None:
            logger.info("Simulating %s fixed histories: %d scene tokens", stage_name, len(previous_tokens))
            evaluate_history_group(
                previous_tokens, cfg, scene_loader, metric_cache_loader.metric_cache_paths, output_dir,
                fingerprint, model_sampling, instantiate(cfg.simulator.proposal_sampling), history_results,
                token_errors,
            )
        logger.info("Evaluating %s: %d scene tokens", stage_name, len(tokens))
        evaluate_token_group(
            tokens, cfg, scene_loader, metric_cache_loader.metric_cache_paths, output_dir,
            fingerprint, model_sampling, token_results, token_errors,
            history_pairs, history_results if history_pairs is not None else None,
        )

    rows, invalid, details = aggregate_mapping_results(
        mappings, token_results, token_errors, instantiate(cfg.simulator.proposal_sampling), cfg.oracle_sigma_squared
    )
    summary = summarize_results(
        rows, invalid, required_tokens, token_results, token_errors,
        history_results if history_pairs is not None else None,
    )
    fixed_rows = fixed_details = fixed_summary = None
    if history_pairs is not None:
        fixed_rows, fixed_invalid, fixed_details = aggregate_mapping_results(
            mappings, token_results, token_errors, instantiate(cfg.simulator.proposal_sampling),
            cfg.oracle_sigma_squared, "fixed_ec_scores",
            {token: detail["weights"] for token, detail in details.items()},
        )
        if [row["original_token"] for row in rows] != [row["original_token"] for row in fixed_rows] or (
            [row["original_token"] for row in invalid] != [row["original_token"] for row in fixed_invalid]
        ):
            raise RuntimeError("No-EC and fixed-history evaluation coverage differs")
        fixed_summary = summarize_results(
            fixed_rows, fixed_invalid, required_tokens, token_results, token_errors, history_results,
        )
        fixed_summary["metric"] = "EPDMS-fixed-history-EC"
    write_report(output_dir, rows, invalid, details, summary, token_errors, fixed_rows, fixed_details, fixed_summary)
    logger.info("Oracle summary: %s", json.dumps(summary))
    if fixed_summary is not None:
        logger.info("Fixed-history oracle summary: %s", json.dumps(fixed_summary))
    logger.info("Oracle results: %s", output_dir)
    if not rows:
        raise RuntimeError(f"No complete original now scenes. See {output_dir / 'failures.json'}")


if __name__ == "__main__":
    main()