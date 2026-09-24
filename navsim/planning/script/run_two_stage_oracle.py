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
from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.evaluate.two_stage_oracle import (
    METRIC_COLUMNS,
    compare_oracles,
    evaluate_scene_proposals,
    gaussian_weights,
    proposal_endpoints,
    select_now_mappings,
    stage_two_oracles,
)
from navsim.planning.script.gpu_inference import predict_proposals, predict_proposals_multi_gpu


CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_two_stage_oracle"
logger = logging.getLogger(__name__)


def score_scene_job(cache_path: Path, proposals: np.ndarray, cfg: DictConfig) -> Dict[str, np.ndarray]:
    before = cache_stamp(cache_path)
    with lzma.open(cache_path, "rb") as cache_file:
        metric_cache = pickle.load(cache_file)

    simulator = instantiate(cfg.simulator)
    scorer = instantiate(cfg.scorer)
    if simulator.proposal_sampling != scorer.proposal_sampling:
        raise ValueError("Simulator and scorer sampling must match")
    traffic_policy = instantiate(cfg.traffic_agents_policy.reactive, simulator.proposal_sampling)
    model_sampling = instantiate(cfg.agent.trajectory_sampling)
    scores, metrics = evaluate_scene_proposals(
        metric_cache, proposals, model_sampling, simulator, scorer, traffic_policy
    )
    if not np.array_equal(before, cache_stamp(cache_path)):
        raise RuntimeError(f"Metric cache changed while evaluating {cache_path}")

    rear_axle = metric_cache.ego_state.rear_axle
    return {
        "proposals": proposals,
        "scores": scores,
        "metrics": np.array([[row[column] for column in METRIC_COLUMNS] for row in metrics], dtype=np.float64),
        "endpoints": proposal_endpoints(proposals, rear_axle),
        "start": np.array([rear_axle.x, rear_axle.y], dtype=np.float64),
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
) -> None:
    path = token_artifact_path(output_dir, token)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{token}-", suffix=".npz", delete=False) as temp:
        temp_path = Path(temp.name)
        try:
            np.savez_compressed(temp, **result, fingerprint=fingerprint, cache_stamp=cache_stamp(cache_path))
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
            result = {key: archive[key] for key in ("proposals", "scores", "metrics", "endpoints", "start")}
        expected_shapes = {
            "proposals": (64, model_sampling.num_poses, 3),
            "scores": (64,),
            "metrics": (64, len(METRIC_COLUMNS)),
            "endpoints": (64, 2),
            "start": (2,),
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
        "mode": "stage-two-max-now-no-ec-v1",
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


def verify_manifest(output_dir: Path, fingerprint: str) -> None:
    path = output_dir / "manifest.json"
    if path.exists():
        with path.open(encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        if manifest.get("fingerprint") != fingerprint:
            raise ValueError("Oracle output_dir already contains a different configuration; select a new output_dir")
    else:
        write_json_atomic(path, {
            "fingerprint": fingerprint,
            "metric": "EPDMS-no-EC",
            "branch": "now",
            "num_proposals": 64,
        })


def aggregate_mapping_results(
    mappings: List[Tuple[str, List[str]]],
    token_results: Dict[str, Dict[str, np.ndarray]],
    token_errors: Dict[str, str],
    proposal_sampling: TrajectorySampling,
    sigma_squared: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Dict[str, np.ndarray]]]:
    unique_followups = list(dict.fromkeys(token for _, followups in mappings for token in followups))
    stage_two_best = {}
    for token in unique_followups:
        if token in token_results:
            maxima, indices = stage_two_oracles(token_results[token]["scores"][None])
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
            weights = gaussian_weights(
                stage_one["endpoints"], followup_tokens, starts, proposal_sampling, sigma_squared
            )
            comparison = compare_oracles(stage_one["scores"], stage_two_scores, weights)
            distances_squared = ((stage_one["endpoints"][:, None] - starts[None]) ** 2).sum(axis=2)
            gaussian_sums = np.exp(-distances_squared / (2 * sigma_squared)).sum(axis=1)
            fallback_rows = int(np.count_nonzero(np.isclose(gaussian_sums, 0.0) | np.isnan(gaussian_sums)))
            short_index = comparison["short_index"]
            long_index = comparison["long_index"]
            ranked_stage_one = np.sort(stage_one["scores"])
            rows.append({
                "original_token": original_token,
                "followup_count": len(followup_tokens),
                "short_index": short_index,
                "long_index": long_index,
                "E1_short": float(stage_one["scores"][short_index]),
                "E1_long": float(stage_one["scores"][long_index]),
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
                "stage_one_scores": stage_one["scores"],
                "stage_one_endpoints": stage_one["endpoints"],
                "stage_two_starts": starts,
                "weights": weights,
                "downstream_values": comparison["downstream_values"],
                "combined_scores": stage_one["scores"] * comparison["downstream_values"],
            }
        except (KeyError, ValueError, IndexError) as error:
            logger.exception("Unable to aggregate original scene %s", original_token)
            invalid.append({"original_token": original_token, "errors": {original_token: str(error)}})

    return rows, invalid, details


def infer_proposals(cfg: DictConfig, scene_loader: SceneLoader, tokens: List[str]) -> Dict[str, np.ndarray]:
    device = torch.device(cfg.gpu_device)
    if cfg.gpu_num_devices > 1:
        return predict_proposals_multi_gpu(cfg, scene_loader, tokens, device, cfg.gpu_num_devices)

    agent: AbstractAgent = instantiate(cfg.agent)
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
    try:
        return predict_proposals(
            agent, inference_loader, tokens, device, cfg.gpu_batch_size, cfg.gpu_num_workers
        )
    finally:
        del agent
        torch.cuda.empty_cache()


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
) -> None:
    pending = []
    available = set(scene_loader.tokens)
    for token in tokens:
        if token in token_results:
            continue
        if token not in available or token not in cache_paths:
            token_errors[token] = "Missing scene or metric cache"
            continue
        cache_path = Path(cache_paths[token])
        try:
            cached = load_token_artifact(output_dir, token, fingerprint, cache_path, model_sampling)
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
        if (
            result["scores"].shape != (64,)
            or result["metrics"].shape != (64, len(METRIC_COLUMNS))
            or result["endpoints"].shape != (64, 2)
            or result["start"].shape != (2,)
            or result["proposals"].shape != (64, model_sampling.num_poses, 3)
            or any(not np.isfinite(values).all() for values in result.values())
        ):
            raise ValueError(f"Incomplete evaluator result for {token}")
        save_token_artifact(output_dir, token, result, fingerprint, cache_path)
        token_results[token] = result

    if cfg.oracle_workers == 1:
        for token in pending:
            try:
                record_result(token, score_scene_job(Path(cache_paths[token]), proposals[token], worker_cfg))
            except Exception as error:
                logger.exception("Unable to evaluate %s", token)
                token_errors[token] = f"Evaluation failed: {error}"
    else:
        with ProcessPoolExecutor(
            max_workers=cfg.oracle_workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = {
                pool.submit(score_scene_job, Path(cache_paths[token]), proposals[token], worker_cfg): token
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
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "metric": "EPDMS-no-EC",
        "branch": "now",
        "attempted_mappings": len(rows) + len(invalid),
        "valid_mappings": len(rows),
        "invalid_mappings": len(invalid),
        "required_tokens": len(required_tokens),
        "complete_tokens": len(token_results),
        "failed_tokens": len(set(required_tokens) - set(token_results)),
        "failed_token_details": token_errors,
        "complete_proposal_scores": 64 * len(token_results),
    }
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
) -> None:
    for token, arrays in details.items():
        write_npz_atomic(output_dir / "mappings" / f"{token}.npz", arrays)

    path = output_dir / "mappings.csv"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output_dir, delete=False) as temp:
        temp_path = Path(temp.name)
        try:
            pd.DataFrame(rows, columns=MAPPING_COLUMNS).to_csv(temp, index=False)
            temp.flush()
            os.fsync(temp.fileno())
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
    os.replace(temp_path, path)
    write_json_atomic(output_dir / "summary.json", summary)
    write_json_atomic(output_dir / "failures.json", {"invalid_mappings": invalid, "token_errors": token_errors})


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    if cfg.oracle_workers < 1 or cfg.oracle_sigma_squared <= 0:
        raise ValueError("oracle_workers and oracle_sigma_squared must be positive")
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
    stage_one_tokens = [token for token, _ in mappings]
    stage_two_tokens = list(dict.fromkeys(token for _, followups in mappings for token in followups))
    if set(stage_one_tokens) & set(stage_two_tokens):
        raise ValueError("Original now scenes and follow-up now scenes must be distinct")
    required_tokens = stage_two_tokens + stage_one_tokens
    output_dir = Path(cfg.output_dir)
    fingerprint = configuration_fingerprint(cfg)
    verify_manifest(output_dir, fingerprint)

    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.tokens = stage_one_tokens
    scene_filter.synthetic_scene_tokens = stage_two_tokens
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
    token_errors: Dict[str, str] = {}
    logger.info("Oracle evaluation: %d mappings, %d follow-up scenes", len(mappings), len(stage_two_tokens))

    for stage_name, tokens in (("stage-two", stage_two_tokens), ("stage-one", stage_one_tokens)):
        logger.info("Evaluating %s: %d scene tokens", stage_name, len(tokens))
        evaluate_token_group(
            tokens, cfg, scene_loader, metric_cache_loader.metric_cache_paths, output_dir,
            fingerprint, model_sampling, token_results, token_errors,
        )

    rows, invalid, details = aggregate_mapping_results(
        mappings, token_results, token_errors, instantiate(cfg.simulator.proposal_sampling), cfg.oracle_sigma_squared
    )
    summary = summarize_results(rows, invalid, required_tokens, token_results, token_errors)
    write_report(output_dir, rows, invalid, details, summary, token_errors)
    logger.info("Oracle summary: %s", json.dumps(summary))
    logger.info("Oracle results: %s", output_dir)
    if not rows:
        raise RuntimeError(f"No complete original now scenes. See {output_dir / 'failures.json'}")


if __name__ == "__main__":
    main()