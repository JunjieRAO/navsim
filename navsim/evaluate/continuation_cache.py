import json
import lzma
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from navsim.common.dataclasses import Scene, SceneFilter, SensorConfig
from navsim.common.enums import SceneFrameType
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario


INDEX_VERSION = 3
SHIFT_FRAMES = 8
FUTURE_FRAMES = 8


def source_stamp(path: Path) -> List[int]:
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns]


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temp_path = Path(stream.name)
        try:
            json.dump(payload, stream, indent=2, ensure_ascii=True)
            stream.flush()
            os.fsync(stream.fileno())
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
    os.replace(temp_path, path)


def index_current_scenes(data_root: Path, scene_filter: SceneFilter, max_scenes: int = 0) -> Dict[str, Any]:
    """Index current scenes and require t+8 annotations in the same original log."""
    if max_scenes < 0 or scene_filter.num_history_frames != 4 or scene_filter.num_future_frames < 10:
        raise ValueError("V2 requires four history frames and at least ten original future frames")
    if scene_filter.include_synthetic_scenes:
        raise ValueError("V2 indexes only original trainval logs")
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)

    log_files = sorted(path for path in data_root.iterdir() if path.suffix == ".pkl")
    if scene_filter.log_names is not None:
        names = set(scene_filter.log_names)
        log_files = [path for path in log_files if path.stem in names]
        missing = names - {path.stem for path in log_files}
        if missing:
            raise FileNotFoundError(f"Missing trainval logs: {sorted(missing)[:5]}")
    if not log_files:
        raise ValueError("No original trainval logs to index")

    entries: List[Dict[str, Any]] = []
    stamps: Dict[str, List[int]] = {}
    seen_tokens = set()
    selected_tokens = set(scene_filter.tokens) if scene_filter.tokens is not None else None
    eligible = 0
    for log_path in log_files:
        before = source_stamp(log_path)
        with log_path.open("rb") as stream:
            frames = pickle.load(stream)
        if before != source_stamp(log_path):
            raise RuntimeError(f"Log changed while indexing: {log_path}")
        stamps[log_path.name] = before
        for start in range(0, len(frames) - scene_filter.num_frames + 1, scene_filter.frame_interval):
            anchor = start + scene_filter.num_history_frames - 1
            token = frames[anchor]["token"]
            if selected_tokens is not None and token not in selected_tokens:
                continue
            if token in seen_tokens:
                raise ValueError(f"Repeated current scene token: {token}")
            seen_tokens.add(token)
            shift = anchor + SHIFT_FRAMES
            last = shift + FUTURE_FRAMES
            reason = "eligible"
            if not frames[anchor]["roadblock_ids"]:
                reason = "missing_current_route"
            elif last >= len(frames):
                reason = "missing_eight_second_future"
            elif not frames[shift]["roadblock_ids"]:
                reason = "missing_shifted_route"
            else:
                timestamps = np.array([frame["timestamp"] for frame in frames[start : last + 1]], dtype=np.int64)
                expected = frames[anchor]["timestamp"] + (np.arange(start, last + 1) - anchor) * 500_000
                if not np.all(np.abs(timestamps - expected) <= 20_000):
                    reason = "discontinuous_timestamps"

            entries.append({
                "token": token,
                "log_file": log_path.name,
                "log_name": frames[anchor]["log_name"],
                "start_index": start,
                "timestamp": frames[anchor]["timestamp"],
                "shift_token": frames[shift]["token"] if shift < len(frames) else None,
                "shift_timestamp": frames[shift]["timestamp"] if shift < len(frames) else None,
                "reason": reason,
            })
            eligible += reason == "eligible"
            if max_scenes and eligible >= max_scenes:
                break
        if max_scenes and eligible >= max_scenes:
            break

    if not eligible:
        raise ValueError("No navtrain scenes have a complete eight-second GT window")

    return {
        "version": INDEX_VERSION,
        "data_root": str(data_root.resolve()),
        "history_frames": scene_filter.num_history_frames,
        "original_future_frames": scene_filter.num_future_frames,
        "frame_interval": scene_filter.frame_interval,
        "source_stamps": stamps,
        "entries": entries,
        "eligible": eligible,
        "skipped": {reason: sum(entry["reason"] == reason for entry in entries) for reason in sorted({
            entry["reason"] for entry in entries if entry["reason"] != "eligible"
        })},
    }


def load_scene_index(path: Path, data_root: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        index = json.load(stream)
    if index.get("version") != INDEX_VERSION or index.get("data_root") != str(data_root.resolve()):
        raise ValueError("V2 scene index version or dataset path changed; rerun the index step")
    for log_name, stamp in index["source_stamps"].items():
        if source_stamp(data_root / log_name) != stamp:
            raise ValueError(f"Original log changed: {log_name}; rerun the index step in a new V2 directory")
    return index


def frames_for_entry(frames: List[Dict[str, Any]], entry: Dict[str, Any], index: Dict[str, Any]) -> Tuple[List, List]:
    if entry["reason"] != "eligible":
        raise ValueError("Only eligible scenes have a complete shifted GT window")
    start = entry["start_index"]
    anchor = start + index["history_frames"] - 1
    shifted_start = anchor + SHIFT_FRAMES - index["history_frames"] + 1
    original = frames[start : start + index["history_frames"] + index["original_future_frames"]]
    shifted = frames[shifted_start : shifted_start + index["history_frames"] + FUTURE_FRAMES]
    if (len(original) != index["history_frames"] + index["original_future_frames"]
            or len(shifted) != index["history_frames"] + FUTURE_FRAMES
            or original[index["history_frames"] - 1]["token"] != entry["token"]
            or original[index["history_frames"] - 1]["log_name"] != entry["log_name"]
            or shifted[index["history_frames"] - 1]["token"] != entry["shift_token"]):
        raise ValueError("Original log does not match the cached V2 scene index")
    return original, shifted


def metric_cache_path(root: Path, log_name: str, token: str) -> Path:
    if not log_name or Path(log_name).name != log_name:
        raise ValueError(f"Invalid metric-cache log name: {log_name}")
    return root / log_name / "unknown" / token / "metric_cache.pkl"


def load_metric_cache(path: Path, expected_timestamp: int) -> MetricCache:
    with lzma.open(path, "rb") as stream:
        cache = pickle.load(stream)
    observation = getattr(cache, "observation", None)
    traffic_lights = getattr(observation, "_occupancy_maps_tl", None)
    if (not isinstance(cache, MetricCache) or getattr(cache, "scene_type", None) != SceneFrameType.ORIGINAL
            or cache.ego_state.time_point.time_us != expected_timestamp
            or len(observation.detections_tracks) < 41
            or traffic_lights is None or len(traffic_lights) < 41):
        raise ValueError(f"Incompatible NAVSIM-v2 metric cache or wrong anchor time: {path}")
    return cache


def cache_shared_scene(
    frames: List[Dict[str, Any]],
    entry: Dict[str, Any],
    index: Dict[str, Any],
    original_processor,
    shifted_processor,
    map_root: str,
    existing_v1_root: Path = None,
) -> Tuple[Path, Path]:
    """Reuse V1 when possible and cache original/shifted GT independently of proposals."""
    original_frames, shifted_frames = frames_for_entry(frames, entry, index)
    log_name = entry["log_name"]
    original_root = Path(original_processor._cache_path)
    shifted_root = Path(shifted_processor._cache_path)
    reusable_v1 = metric_cache_path(existing_v1_root, log_name, entry["token"]) if existing_v1_root else None
    reusable_shift = metric_cache_path(existing_v1_root, log_name, entry["shift_token"]) if existing_v1_root else None
    original_path = reusable_v1 if reusable_v1 and reusable_v1.exists() else metric_cache_path(
        original_root, log_name, entry["token"],
    )
    shifted_path = reusable_shift if reusable_shift and reusable_shift.exists() else metric_cache_path(
        shifted_root, log_name, entry["shift_token"],
    )

    for path, scene_frames, processor, future_frames in (
        (original_path, original_frames, original_processor, index["original_future_frames"]),
        (shifted_path, shifted_frames, shifted_processor, FUTURE_FRAMES),
    ):
        expected_timestamp = scene_frames[index["history_frames"] - 1]["timestamp"]
        if not path.exists():
            scene = Scene.from_scene_dict_list(
                scene_frames, None,
                num_history_frames=index["history_frames"],
                num_future_frames=future_frames,
                sensor_config=SensorConfig.build_no_sensors(),
            )
            scenario = NavSimScenario(scene, map_root=map_root, map_version="nuplan-maps-v1.0")
            result = processor.compute_and_save_metric_cache(scenario)
            if result is None or not path.is_file():
                raise RuntimeError(f"Metric cache generation failed: {path}")
        load_metric_cache(path, expected_timestamp)

    return original_path, shifted_path