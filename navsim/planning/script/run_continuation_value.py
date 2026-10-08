import hashlib
import inspect
import json
import logging
import multiprocessing
import os
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import SceneLoader
from navsim.evaluate.continuation_cache import (
    cache_shared_scene,
    frames_for_entry,
    index_current_scenes,
    load_metric_cache,
    load_scene_index,
    metric_cache_path,
    source_stamp,
    write_json_atomic,
)
from navsim.evaluate.continuation_value import (
    CONTINUATION_MODES,
    _proposal_history,
    _terminal_agent_overlap,
    _trajectory_states,
    analyze_continuation_scores,
    generate_continuations,
    score_proposal_continuations,
    simulate_stage_one_proposals,
)
from navsim.evaluate.two_stage_oracle import METRIC_COLUMNS, evaluate_scene_proposals
from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
from navsim.planning.script.gpu_inference import predict_proposals
from navsim.planning.script.run_two_stage_oracle import token_artifact_path, write_csv_atomic, write_npz_atomic


CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_continuation_value"
logger = logging.getLogger(__name__)


def _root(cfg: DictConfig) -> Path:
    return Path(cfg.v2_root).resolve()


def _index(cfg: DictConfig) -> Dict[str, Any]:
    return load_scene_index(_root(cfg) / "index.json", Path(cfg.navsim_log_path))


def _groups(index: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    groups = defaultdict(list)
    for entry in index["entries"]:
        if entry["reason"] == "eligible":
            groups[entry["log_file"]].append(entry)
    return groups


def _fingerprint(payload: Dict[str, Any], paths=()) -> str:
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))
    for path in paths:
        path = Path(path).resolve()
        digest.update(str(path).encode("utf-8"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _manifest(path: Path, fingerprint: str, details: Dict[str, Any]) -> None:
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            current = json.load(stream)
        if current.get("fingerprint") != fingerprint:
            raise ValueError(f"V2 cache configuration changed; use a new v2_root: {path}")
    else:
        write_json_atomic(path, {"fingerprint": fingerprint, **details})


def _require_manifest(path: Path, fingerprint: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing V2 stage: {path}")
    _manifest(path, fingerprint, {})


def index_scenes(cfg: DictConfig) -> None:
    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    result = index_current_scenes(Path(cfg.navsim_log_path), scene_filter, int(cfg.max_scenes))
    write_json_atomic(_root(cfg) / "index.json", result)
    logger.info("Indexed %d eligible scenes; skipped: %s", result["eligible"], result["skipped"])


def _cache_fingerprint(cfg: DictConfig, index: Dict[str, Any]) -> str:
    v1_root = Path(cfg.v1_cache_root).resolve() if cfg.v1_cache_root else None
    return _fingerprint({
        "index": index["source_stamps"],
        "dataset": index["data_root"],
        "sampling": OmegaConf.to_container(cfg.proposal_sampling, resolve=True),
        "map_root": os.environ["NUPLAN_MAPS_ROOT"],
        "v1_root": str(v1_root) if v1_root else None,
    }, [Path(__file__).resolve().parents[2] / "evaluate" / "continuation_cache.py",
        Path(__file__).resolve().parents[2] / "planning" / "metric_caching" / "metric_cache_processor.py"])


def _cache_log_job(log_path, entries, log_stamp, window, sampling, root, v1_root, map_root):
    original_processor = MetricCacheProcessor(root / "original_metric_cache", False, sampling)
    shifted_processor = MetricCacheProcessor(root / "shifted_metric_cache", False, sampling)
    before = source_stamp(log_path)
    with log_path.open("rb") as stream:
        frames = pickle.load(stream)
    if before != log_stamp or source_stamp(log_path) != before:
        raise RuntimeError(f"Log changed during V2 caching: {log_path.name}")

    failures = {}
    total = 0
    for entry in entries:
        try:
            cache_shared_scene(
                frames, entry, window, original_processor, shifted_processor, map_root, v1_root,
            )
            total += 1
        except Exception as error:
            failures[entry["token"]] = str(error)
            logger.exception("Unable to cache V2 GT for %s", entry["token"])
    return total, failures


def cache_gt(cfg: DictConfig) -> None:
    index = _index(cfg)
    root = _root(cfg)
    if cfg.v2_workers < 1:
        raise ValueError("v2_workers must be positive")
    sampling = instantiate(cfg.proposal_sampling)
    v1_root = Path(cfg.v1_cache_root).resolve() if cfg.v1_cache_root else None
    fingerprint = _cache_fingerprint(cfg, index)
    _manifest(root / "cache_manifest.json", fingerprint, {"metric": "GT t+4..t+8", "source": index["data_root"]})
    window = {"history_frames": index["history_frames"], "original_future_frames": index["original_future_frames"]}
    groups = _groups(index)
    jobs = {
        log_name: (
            Path(index["data_root"]) / log_name, entries, index["source_stamps"][log_name],
            window, sampling, root, v1_root, os.environ["NUPLAN_MAPS_ROOT"],
        )
        for log_name, entries in groups.items()
    }
    failures = {}
    total = 0

    def record(log_name, result):
        nonlocal total
        cached_count, log_failures = result
        total += cached_count
        failures.update(log_failures)
        logger.info("Cached %d / %d V2 scenes in %s", cached_count, len(groups[log_name]), log_name)

    def record_failure(log_name, error):
        logger.exception("Unable to cache V2 log %s", log_name)
        failures.update({entry["token"]: str(error) for entry in groups[log_name]})

    if cfg.v2_workers == 1 or len(jobs) <= 1:
        for log_name, job in jobs.items():
            try:
                record(log_name, _cache_log_job(*job))
            except Exception as error:
                record_failure(log_name, error)
    else:
        with ProcessPoolExecutor(
            max_workers=min(cfg.v2_workers, len(jobs)), mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            futures = {pool.submit(_cache_log_job, *job): log_name for log_name, job in jobs.items()}
            for future in as_completed(futures):
                log_name = futures[future]
                try:
                    record(log_name, future.result())
                except Exception as error:
                    record_failure(log_name, error)
    write_json_atomic(root / "cache_status.json", {"complete": total, "failed": failures})
    logger.info("Cached %d / %d eligible scenes", total, index["eligible"])
    if failures:
        raise RuntimeError(f"GT cache failed for {len(failures)} scenes; see cache_status.json")


def _inference_fingerprint(cfg: DictConfig, index: Dict[str, Any]) -> str:
    workspace = Path(__file__).resolve().parents[3]
    weights = Path(cfg.agent.config.image_backbone.model_weights)
    if not weights.is_absolute():
        weights = workspace / weights
    return _fingerprint({
        "model": OmegaConf.to_container(cfg.agent, resolve=True),
        "source_stamps": index["source_stamps"],
        "data_root": index["data_root"],
        "sensor_root": str(Path(cfg.original_sensor_path).resolve()),
    }, [
        Path(cfg.agent.checkpoint_path), weights,
        workspace / "navsim/agents/drivoR/drivor_model.py",
        workspace / "navsim/agents/drivoR/drivor_features.py",
        workspace / "navsim/planning/script/gpu_inference.py",
    ])


def _cached_proposal(path: Path, fingerprint: str, log_stamp: List[int], num_poses: int) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            poses = data["proposals"]
            return (data["fingerprint"].item() == fingerprint
                    and np.array_equal(data["log_stamp"], log_stamp)
                    and poses.shape == (64, num_poses, 3) and np.isfinite(poses).all())
    except (OSError, KeyError, ValueError):
        return False


def _inference_device(cfg: DictConfig) -> torch.device:
    device = torch.device(cfg.gpu_device)
    if cfg.gpu_num_devices < 1 or cfg.gpu_batch_size < 1 or cfg.gpu_num_workers < 0:
        raise ValueError("GPU count, batch size and worker count must be valid")
    if device.type == "cuda":
        if not torch.cuda.is_available() or (device.index or 0) + cfg.gpu_num_devices > torch.cuda.device_count():
            raise RuntimeError("Requested CUDA devices are unavailable")
        for index in range(device.index or 0, (device.index or 0) + cfg.gpu_num_devices):
            major, minor = torch.cuda.get_device_capability(index)
            if f"sm_{major}{minor}" not in torch.cuda.get_arch_list():
                raise RuntimeError(f"PyTorch cannot run on CUDA device {index} (sm_{major}{minor})")
    elif cfg.gpu_num_devices != 1:
        raise ValueError("Multiple inference devices require CUDA")
    return device


def _inference_shards(groups, device_indices):
    shards = {device_index: [] for device_index in device_indices}
    loads = {device_index: 0 for device_index in device_indices}
    for log_name, entries in sorted(groups.items(), key=lambda item: len(item[1]), reverse=True):
        device_index = min(loads, key=loads.get)
        shards[device_index].append((log_name, entries))
        loads[device_index] += len(entries)
    return {device_index: assignments for device_index, assignments in shards.items() if assignments}


def _infer_gpu_shard(cfg, assignments, fingerprint, device_index, data_root, source_stamps, window):
    device = torch.device("cuda", device_index) if device_index is not None else torch.device("cpu")
    if device_index is not None:
        torch.cuda.set_device(device)
    agent = instantiate(cfg.agent)
    agent.initialize()
    sensor_config = agent.get_sensor_config()
    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    pending_scenes = {}
    pending_entries = []
    failures = {}
    complete = 0

    def infer_chunk():
        nonlocal complete
        if not pending_entries:
            return
        try:
            scene_loader = SceneLoader(
                data_path=Path(data_root),
                original_sensor_path=Path(cfg.original_sensor_path),
                synthetic_sensor_path=None,
                scene_filter=scene_filter,
                sensor_config=sensor_config,
                original_scene_index=pending_scenes,
                synthetic_scene_index={},
            )
            tokens = [entry["token"] for entry, _ in pending_entries]
            predictions = predict_proposals(
                agent, scene_loader, tokens, device, cfg.gpu_batch_size, cfg.gpu_num_workers,
            )
            if set(predictions) != set(tokens):
                raise ValueError("DrivoR did not return proposals for every inference chunk")
            for entry, log_stamp in pending_entries:
                write_npz_atomic(token_artifact_path(_root(cfg) / "proposals", entry["token"]), {
                    "proposals": predictions[entry["token"]],
                    "fingerprint": np.asarray(fingerprint),
                    "log_stamp": np.asarray(log_stamp, dtype=np.int64),
                })
                complete += 1
        except Exception as error:
            logger.exception("DrivoR inference failed for a %d-scene chunk", len(pending_entries))
            failures.update({entry["token"]: str(error) for entry, _ in pending_entries})
        finally:
            pending_scenes.clear()
            pending_entries.clear()

    for log_name, entries in assignments:
        log_path = Path(data_root) / log_name
        try:
            before = source_stamp(log_path)
            with log_path.open("rb") as stream:
                frames = pickle.load(stream)
            if before != source_stamps[log_name] or source_stamp(log_path) != before:
                raise RuntimeError(f"Log changed during inference: {log_name}")
            for entry in entries:
                pending_scenes[entry["token"]] = frames_for_entry(frames, entry, window)[0]
                pending_entries.append((entry, before))
                if len(pending_entries) >= cfg.gpu_inference_chunk_scenes:
                    infer_chunk()
        except Exception as error:
            logger.exception("Cannot read inference scenes in %s", log_name)
            failures.update({entry["token"]: str(error) for entry in entries})
    infer_chunk()
    return complete, failures


def infer_proposals(cfg: DictConfig) -> None:
    index = _index(cfg)
    root = _root(cfg)
    sampling = instantiate(cfg.agent.trajectory_sampling)
    if sampling.num_poses != 8 or not np.isclose(sampling.interval_length, 0.5) or cfg.agent.config.proposal_num != 64:
        raise ValueError("V2 requires the 64 x 8 pose, four-second DrivoR checkpoint")
    if cfg.gpu_inference_chunk_scenes < 1:
        raise ValueError("gpu_inference_chunk_scenes must be positive")
    fingerprint = _inference_fingerprint(cfg, index)
    _manifest(root / "proposals" / "manifest.json", fingerprint, {"checkpoint": str(cfg.agent.checkpoint_path)})
    device = _inference_device(cfg)

    complete = 0
    groups = {}
    for log_name, entries in _groups(index).items():
        pending = [entry for entry in entries if not _cached_proposal(
            token_artifact_path(root / "proposals", entry["token"]),
            fingerprint, index["source_stamps"][log_name], sampling.num_poses,
        )]
        complete += len(entries) - len(pending)
        if pending:
            groups[log_name] = pending

    failures = {}
    if groups:
        device_indices = [
            (device.index or 0) + offset if device.type == "cuda" else None
            for offset in range(cfg.gpu_num_devices)
        ]
        shards = _inference_shards(groups, device_indices)
        window = {
            "history_frames": index["history_frames"],
            "original_future_frames": index["original_future_frames"],
        }
        if len(shards) == 1:
            device_index, assignments = next(iter(shards.items()))
            new_count, shard_failures = _infer_gpu_shard(
                cfg, assignments, fingerprint, device_index, index["data_root"], index["source_stamps"], window,
            )
            complete += new_count
            failures.update(shard_failures)
        else:
            with ProcessPoolExecutor(max_workers=len(shards), mp_context=multiprocessing.get_context("spawn")) as pool:
                futures = {
                    pool.submit(
                        _infer_gpu_shard, cfg, assignments, fingerprint, device_index,
                        index["data_root"], index["source_stamps"], window,
                    ): assignments
                    for device_index, assignments in shards.items()
                }
                for future in as_completed(futures):
                    assignments = futures[future]
                    try:
                        new_count, shard_failures = future.result()
                        complete += new_count
                        failures.update(shard_failures)
                    except Exception as error:
                        logger.exception("GPU inference worker failed")
                        failures.update({entry["token"]: str(error) for _, entries in assignments for entry in entries})

    write_json_atomic(root / "inference_status.json", {"complete": complete, "failed": failures})
    if failures:
        raise RuntimeError(f"Inference failed for {len(failures)} scenes; see inference_status.json")


def _metric_paths(cfg: DictConfig, entry: Dict[str, Any]):
    root = _root(cfg)
    log_name = entry["log_name"]
    original = metric_cache_path(root / "original_metric_cache", log_name, entry["token"])
    shifted = metric_cache_path(root / "shifted_metric_cache", log_name, entry["shift_token"])
    if cfg.v1_cache_root:
        previous = Path(cfg.v1_cache_root)
        previous_original = metric_cache_path(previous, log_name, entry["token"])
        previous_shifted = metric_cache_path(previous, log_name, entry["shift_token"])
        if previous_original.is_file():
            original = previous_original
        if previous_shifted.is_file():
            shifted = previous_shifted
    return original, shifted


def _score_fingerprint(cfg: DictConfig, index: Dict[str, Any], inference_fingerprint: str) -> str:
    workspace = Path(__file__).resolve().parents[3]
    score_functions = (
        generate_continuations, _trajectory_states, simulate_stage_one_proposals,
        _proposal_history, _terminal_agent_overlap, score_proposal_continuations,
    )
    return _fingerprint({
        "mode": "GT-continuation-no-EC-v1",
        "inference": inference_fingerprint,
        "cache": _cache_fingerprint(cfg, index),
        "settings": _score_settings(cfg),
        "modes": repr(CONTINUATION_MODES),
        "score_functions": [inspect.getsource(function) for function in score_functions],
    }, [
        workspace / "navsim/evaluate/two_stage_oracle.py",
        workspace / "navsim/evaluate/pdm_score.py",
        workspace / "navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py",
        workspace / "navsim/planning/simulation/planner/pdm_planner/simulation/pdm_simulator.py",
        workspace / "navsim/traffic_agents_policies/log_replay_traffic_agents.py",
    ])


def _score_settings(cfg: DictConfig) -> Dict[str, Any]:
    return {
        "score_sampling": OmegaConf.to_container(cfg.proposal_sampling, resolve=True),
        "model_sampling": OmegaConf.to_container(cfg.agent.trajectory_sampling, resolve=True),
        "simulator": OmegaConf.to_container(cfg.simulator, resolve=True),
        "scorer": OmegaConf.to_container(cfg.scorer, resolve=True),
        "gt_traffic": OmegaConf.to_container(cfg.traffic_agents_policy.non_reactive, resolve=True),
        "max_k": int(cfg.max_k),
    }


def _stored_score_fingerprint(cfg: DictConfig, current_fingerprint: str) -> str:
    root = _root(cfg)
    manifest_path = root / "scores" / "manifest.json"
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    fingerprint = manifest.get("fingerprint")
    if (not isinstance(fingerprint, str) or not fingerprint
            or manifest.get("metric") != "EPDMS-no-EC" or manifest.get("max_k") != cfg.max_k):
        raise ValueError("Missing or incompatible stored V2 continuation scores")
    if fingerprint == current_fingerprint:
        return fingerprint

    settings = manifest.get("score_settings")
    if settings is None:
        snapshot = root / "hydra" / "score" / "code" / "hydra" / "config.yaml"
        if not snapshot.is_file():
            raise ValueError("Stored scores have no scoring configuration snapshot")
        settings = _score_settings(OmegaConf.load(snapshot))
    if settings != _score_settings(cfg):
        raise ValueError("Stored scoring settings differ; use matching V2 score artifacts")
    return fingerprint


def _score_input_stamps(cfg: DictConfig, entry: Dict[str, Any]):
    original, shifted = _metric_paths(cfg, entry)
    proposal = token_artifact_path(_root(cfg) / "proposals", entry["token"])
    return {
        "original_stamp": source_stamp(original),
        "shifted_stamp": source_stamp(shifted),
        "proposal_stamp": source_stamp(proposal),
    }


def _cached_score(path: Path, fingerprint: str, stamps: Dict[str, List[int]], max_k: int) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            expected_shapes = {
                "continuation_scores": (64, max_k),
                "metric_values": (64, max_k, len(METRIC_COLUMNS)),
                "terminal_overlap": (64,),
                "e1_scores": (64,),
                "e1_metrics": (64, len(METRIC_COLUMNS)),
                "endpoint_states": (64, 4),
            }
            return (data["fingerprint"].item() == fingerprint
                    and all(np.array_equal(data[key], stamp) for key, stamp in stamps.items())
                    and all(data[key].shape == shape and np.isfinite(data[key]).all()
                            for key, shape in expected_shapes.items()))
    except (OSError, KeyError, ValueError):
        return False


def score_scene_job(entry: Dict[str, Any], cfg: DictConfig, log_stamp: List[int], inference_fingerprint: str):
    original_path, shifted_path = _metric_paths(cfg, entry)
    proposal_path = token_artifact_path(_root(cfg) / "proposals", entry["token"])
    if not _cached_proposal(proposal_path, inference_fingerprint, log_stamp, 8):
        raise ValueError(f"Missing or stale DrivoR proposals: {entry['token']}")
    stamps = _score_input_stamps(cfg, entry)
    original_cache = load_metric_cache(original_path, entry["timestamp"])
    shifted_cache = load_metric_cache(shifted_path, entry["shift_timestamp"])
    with np.load(proposal_path, allow_pickle=False) as data:
        proposals = data["proposals"]

    simulator = instantiate(cfg.simulator)
    scorer = instantiate(cfg.scorer)
    if scorer._config.human_penalty_filter:
        raise ValueError("V2 requires human_penalty_filter=false for hypothetical proposal endpoints")
    sampling = instantiate(cfg.agent.trajectory_sampling)
    stage_one_states = simulate_stage_one_proposals(proposals, original_cache, sampling, simulator)
    e1_scores, e1_metrics = evaluate_scene_proposals(
        original_cache, proposals, sampling, simulator, scorer,
        instantiate(cfg.traffic_agents_policy.non_reactive, simulator.proposal_sampling),
    )
    scores, metrics, terminal_overlap = score_proposal_continuations(
        stage_one_states, original_cache, shifted_cache, sampling, simulator, scorer, int(cfg.max_k),
    )
    if stamps != _score_input_stamps(cfg, entry):
        raise RuntimeError(f"V2 scoring inputs changed while computing {entry['token']}")
    endpoints = stage_one_states[:, -1]
    return {
        "continuation_scores": scores,
        "metric_values": metrics,
        "terminal_overlap": terminal_overlap,
        "e1_scores": e1_scores,
        "e1_metrics": np.array([[row[column] for column in METRIC_COLUMNS] for row in e1_metrics]),
        "endpoint_states": np.column_stack((
            endpoints[:, :3], np.hypot(endpoints[:, 3], endpoints[:, 4]),
        )),
        **{key: np.array(stamp, dtype=np.int64) for key, stamp in stamps.items()},
    }


def _scoring_worker_config(cfg: DictConfig) -> DictConfig:
    return OmegaConf.create({
        "v2_root": str(_root(cfg)),
        "v1_cache_root": cfg.v1_cache_root,
        "max_k": cfg.max_k,
        "simulator": OmegaConf.to_container(cfg.simulator, resolve=True),
        "scorer": OmegaConf.to_container(cfg.scorer, resolve=True),
        "agent": {"trajectory_sampling": OmegaConf.to_container(cfg.agent.trajectory_sampling, resolve=True)},
        "traffic_agents_policy": {
            "non_reactive": OmegaConf.to_container(cfg.traffic_agents_policy.non_reactive, resolve=True),
        },
    })


def score_scenes(cfg: DictConfig) -> None:
    index = _index(cfg)
    root = _root(cfg)
    if not 1 <= cfg.max_k <= 64 or cfg.v2_workers < 1:
        raise ValueError("max_k must be 1..64 and v2_workers must be positive")
    inference_fingerprint = _inference_fingerprint(cfg, index)
    _require_manifest(root / "proposals" / "manifest.json", inference_fingerprint)
    _require_manifest(root / "cache_manifest.json", _cache_fingerprint(cfg, index))
    fingerprint = _score_fingerprint(cfg, index, inference_fingerprint)
    _manifest(root / "scores" / "manifest.json", fingerprint, {
        "max_k": cfg.max_k, "metric": "EPDMS-no-EC", "score_settings": _score_settings(cfg),
    })

    failures = {}
    pending = []
    complete = 0
    for log_name, entries in _groups(index).items():
        for entry in entries:
            path = token_artifact_path(root / "scores", entry["token"])
            try:
                stamps = _score_input_stamps(cfg, entry)
                if _cached_score(path, fingerprint, stamps, cfg.max_k):
                    complete += 1
                    continue
            except (OSError, ValueError):
                pass
            pending.append((entry, index["source_stamps"][log_name]))

    worker_cfg = _scoring_worker_config(cfg)

    def record(entry, result):
        nonlocal complete
        write_npz_atomic(token_artifact_path(root / "scores", entry["token"]), {
            **result, "fingerprint": np.asarray(fingerprint),
        })
        complete += 1

    if cfg.v2_workers == 1 or len(pending) <= 1:
        for entry, log_stamp in pending:
            try:
                record(entry, score_scene_job(entry, worker_cfg, log_stamp, inference_fingerprint))
            except Exception as error:
                failures[entry["token"]] = str(error)
                logger.exception("V2 scoring failed for %s", entry["token"])
    else:
        with ProcessPoolExecutor(
            max_workers=min(cfg.v2_workers, len(pending)), mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            futures = {
                pool.submit(score_scene_job, entry, worker_cfg, log_stamp, inference_fingerprint): entry
                for entry, log_stamp in pending
            }
            for future in as_completed(futures):
                entry = futures[future]
                try:
                    record(entry, future.result())
                except Exception as error:
                    failures[entry["token"]] = str(error)
                    logger.exception("V2 scoring failed for %s", entry["token"])

    write_json_atomic(root / "score_status.json", {"complete": complete, "failed": failures})
    logger.info("Scored %d / %d eligible scenes", complete, index["eligible"])
    if failures:
        raise RuntimeError(f"V2 scoring failed for {len(failures)} scenes; see score_status.json")


def analyze_scores(cfg: DictConfig) -> None:
    index = _index(cfg)
    root = _root(cfg)
    k_values = tuple(cfg.k_values)
    if (not k_values or tuple(sorted(k_values)) != k_values
            or len(set(k_values)) != len(k_values)
            or any(isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= cfg.max_k for k in k_values)):
        raise ValueError("k_values must be strictly increasing integers within scored max_k")

    inference_fingerprint = _inference_fingerprint(cfg, index)
    _require_manifest(root / "proposals" / "manifest.json", inference_fingerprint)
    _require_manifest(root / "cache_manifest.json", _cache_fingerprint(cfg, index))
    score_fingerprint = _stored_score_fingerprint(
        cfg, _score_fingerprint(cfg, index, inference_fingerprint),
    )

    analysis_root = root / "analysis" / "top25" / ("k_" + "_".join(str(k) for k in k_values))
    analysis_root.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = {}
    for entries in _groups(index).values():
        for entry in entries:
            token = entry["token"]
            path = token_artifact_path(root / "scores", token)
            try:
                stamps = _score_input_stamps(cfg, entry)
                if not _cached_score(path, score_fingerprint, stamps, cfg.max_k):
                    raise ValueError("Missing, stale or incomplete continuation scores")
                with np.load(path, allow_pickle=False) as data:
                    targets, scene_rows = analyze_continuation_scores(
                        data["continuation_scores"], data["e1_scores"], k_values,
                    )
                    targets["terminal_overlap"] = data["terminal_overlap"]
                    targets["endpoint_states"] = data["endpoint_states"]
                write_npz_atomic(token_artifact_path(analysis_root, token), {
                    **targets,
                    "score_fingerprint": np.asarray(score_fingerprint),
                    "source_stamp": np.asarray(source_stamp(path), dtype=np.int64),
                })
                rows.extend({"token": token, **row} for row in scene_rows)
            except Exception as error:
                failures[token] = str(error)
                logger.exception("Unable to analyze V2 target %s", token)

    by_k = {}
    for k in k_values:
        subset = [row for row in rows if row["k"] == k]
        defined = [row["spearman_vs_max_k"] for row in subset if row["spearman_vs_max_k"] is not None]
        by_k[str(k)] = {
            "v2_top1_agreement": float(np.mean([row["v2_top_agrees_with_max_k"] for row in subset])) if subset else None,
            "e1_v2_top1_agreement": float(np.mean([
                row["e1_v2_top_agrees_with_max_k"] for row in subset
            ])) if subset else None,
            "all_proposals_tied_rate": float(np.mean([
                row["v2_top_ties"] == 64 for row in subset
            ])) if subset else None,
            "mean_v2_top_ties": float(np.mean([row["v2_top_ties"] for row in subset])) if subset else None,
            "mean_v2_value_span": float(np.mean([row["v2_value_span"] for row in subset])) if subset else None,
            "mean_v2_saturated_fraction": float(np.mean([
                row["v2_saturated_fraction"] for row in subset
            ])) if subset else None,
            "mean_spearman_vs_max_k": float(np.mean(defined)) if defined else None,
            "defined_spearman_scenes": len(defined),
            "mean_abs_v2_change": float(np.mean([row["mean_abs_v2_change"] for row in subset])) if subset else None,
            "max_abs_v2_change": max((row["max_abs_v2_change"] for row in subset), default=None),
        }

    write_csv_atomic(analysis_root / "sensitivity.csv", rows)
    write_json_atomic(analysis_root / "summary.json", {
        "metric": "EPDMS-no-EC",
        "value_target": "top25_mean",
        "top_fraction": 0.25,
        "top_counts": [(k + 3) // 4 for k in k_values],
        "scored_max_k": cfg.max_k,
        "k_values": list(k_values),
        "eligible_scenes": index["eligible"],
        "analyzed_scenes": len(rows) // len(k_values),
        "failed": failures,
        "by_k": by_k,
    })
    logger.info("Analyzed %d / %d scenes at K=%s", len(rows) // len(k_values), index["eligible"], k_values)
    if failures:
        raise RuntimeError(f"V2 analysis missing {len(failures)} scenes; see analysis summary.json")


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    if cfg.stage == "index":
        index_scenes(cfg)
    elif cfg.stage == "cache":
        cache_gt(cfg)
    elif cfg.stage == "infer":
        infer_proposals(cfg)
    elif cfg.stage == "score":
        score_scenes(cfg)
    elif cfg.stage == "analyze":
        analyze_scores(cfg)
    else:
        raise ValueError(f"Unsupported V2 stage: {cfg.stage}")


if __name__ == "__main__":
    main()