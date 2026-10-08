import lzma
import os
import pickle
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from navsim.common.dataclasses import SceneFilter
from navsim.evaluate.continuation_cache import (
    cache_shared_scene, frames_for_entry, index_current_scenes, load_metric_cache, load_scene_index,
    metric_cache_path, write_json_atomic,
)


def make_log(tmp_path, length=21):
    frames = [
        {"token": f"scene-{index}", "timestamp": index * 500_000,
         "roadblock_ids": ["road"], "log_name": "train_log.db"}
        for index in range(length)
    ]
    path = tmp_path / "train_log.pkl"
    with path.open("wb") as stream:
        pickle.dump(frames, stream)
    return frames, path


def test_index_records_window_coverage_and_shifted_alignment(tmp_path):
    frames, _ = make_log(tmp_path)
    scene_filter = SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1)

    index = index_current_scenes(tmp_path, scene_filter)
    original, shifted = frames_for_entry(frames, index["entries"][0], index)

    assert index["eligible"] == 2
    assert index["skipped"] == {"missing_eight_second_future": 6}
    assert [original[3]["token"], shifted[3]["token"], shifted[-1]["token"]] == [
        "scene-3", "scene-11", "scene-19",
    ]
    assert (index["entries"][0]["timestamp"], index["entries"][0]["shift_timestamp"]) == (
        1_500_000, 5_500_000,
    )
    assert index["entries"][0]["log_file"] == "train_log.pkl"
    assert index["entries"][0]["log_name"] == "train_log.db"
    assert index["entries"][-1]["reason"] == "missing_eight_second_future"

    destination = tmp_path / "v2" / "index.json"
    write_json_atomic(destination, index)
    assert load_scene_index(destination, tmp_path)["entries"] == index["entries"]


def test_navtrain_token_filter_avoids_linear_list_membership(tmp_path):
    make_log(tmp_path)

    class NoLinearLookup(list):
        def __contains__(self, token):
            raise AssertionError("Index must use constant-time token membership")

    scene_filter = SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1)
    scene_filter.tokens = NoLinearLookup(["scene-3"])
    index = index_current_scenes(tmp_path, scene_filter)

    assert [entry["token"] for entry in index["entries"]] == ["scene-3"]


def test_index_rejects_missing_route_gaps_and_modified_logs(tmp_path):
    frames, path = make_log(tmp_path, length=40)
    frames[3]["roadblock_ids"] = []
    frames[12]["timestamp"] += 200_000
    with path.open("wb") as stream:
        pickle.dump(frames, stream)

    index = index_current_scenes(tmp_path, SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1))
    assert index["entries"][0]["reason"] == "missing_current_route"
    assert index["entries"][1]["reason"] == "discontinuous_timestamps"
    with pytest.raises(ValueError, match="Only eligible"):
        frames_for_entry(frames, index["entries"][0], index)
    destination = tmp_path / "index.json"
    write_json_atomic(destination, index)
    current_stamp = path.stat().st_mtime_ns
    os.utime(path, ns=(current_stamp, current_stamp + 1_000_000_000))
    with pytest.raises(ValueError, match="Original log changed"):
        load_scene_index(destination, tmp_path)


def test_index_excludes_synthetic_and_can_limit_eligible_scenes(tmp_path):
    make_log(tmp_path)
    scene_filter = SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1)
    assert index_current_scenes(tmp_path, scene_filter, max_scenes=1)["eligible"] == 1
    scene_filter.include_synthetic_scenes = True
    with pytest.raises(ValueError, match="only original"):
        index_current_scenes(tmp_path, scene_filter)


def test_index_rejects_small_shifted_timestamp_drift(tmp_path):
    frames, path = make_log(tmp_path, length=40)
    frames[11]["timestamp"] += 30_000
    with path.open("wb") as stream:
        pickle.dump(frames, stream)

    index = index_current_scenes(tmp_path, SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1))

    assert index["entries"][0]["reason"] == "discontinuous_timestamps"


def test_index_fails_when_no_scene_has_eight_seconds(tmp_path):
    make_log(tmp_path, length=14)
    with pytest.raises(ValueError, match="No navtrain scenes"):
        index_current_scenes(tmp_path, SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1))


def test_cache_reuses_compatible_v1_and_builds_only_shifted_gt(tmp_path):
    frames, _ = make_log(tmp_path)
    index = index_current_scenes(tmp_path, SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1))
    entry = index["entries"][0]
    previous = metric_cache_path(tmp_path / "previous", entry["log_name"], entry["token"])
    previous.parent.mkdir(parents=True)
    previous.touch()
    original_processor = SimpleNamespace(_cache_path=tmp_path / "original")
    shifted_processor = SimpleNamespace(_cache_path=tmp_path / "shifted")
    saved_scenarios = []

    def save_shifted(scenario):
        saved_scenarios.append(scenario)
        path = metric_cache_path(shifted_processor._cache_path, entry["log_name"], entry["shift_token"])
        path.parent.mkdir(parents=True)
        path.touch()
        return object()

    shifted_processor.compute_and_save_metric_cache = save_shifted
    with patch("navsim.evaluate.continuation_cache.load_metric_cache") as load_cache, patch(
        "navsim.evaluate.continuation_cache.Scene.from_scene_dict_list", return_value=object(),
    ) as build_scene, patch("navsim.evaluate.continuation_cache.NavSimScenario", return_value=object()):
        original_path, shifted_path = cache_shared_scene(
            frames, entry, index, original_processor, shifted_processor, "/maps", tmp_path / "previous",
        )

    assert original_path == previous
    assert shifted_path.is_file()
    assert len(saved_scenarios) == 1
    assert build_scene.call_args.kwargs["num_future_frames"] == 8
    assert build_scene.call_args.args[1] is None
    assert load_cache.call_count == 2
    assert [call.args[1] for call in load_cache.call_args_list] == [frames[3]["timestamp"], frames[11]["timestamp"]]


def test_rejects_non_navsim_v2_metric_cache(tmp_path):
    path = tmp_path / "metric_cache.pkl"
    with lzma.open(path, "wb") as stream:
        pickle.dump(SimpleNamespace(ego_state=SimpleNamespace()), stream)
    with pytest.raises(ValueError, match="Incompatible NAVSIM-v2"):
        load_metric_cache(path, 1_000_000)


def test_cache_reuses_shifted_v1_when_available(tmp_path):
    frames, _ = make_log(tmp_path)
    index = index_current_scenes(tmp_path, SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1))
    entry = index["entries"][0]
    previous_root = tmp_path / "previous"
    for token in (entry["token"], entry["shift_token"]):
        path = metric_cache_path(previous_root, entry["log_name"], token)
        path.parent.mkdir(parents=True)
        path.touch()
    original_processor = SimpleNamespace(_cache_path=tmp_path / "original")
    shifted_processor = SimpleNamespace(_cache_path=tmp_path / "shifted")

    with patch("navsim.evaluate.continuation_cache.load_metric_cache") as load_cache, patch(
        "navsim.evaluate.continuation_cache.Scene.from_scene_dict_list",
    ) as build_scene:
        original_path, shifted_path = cache_shared_scene(
            frames, entry, index, original_processor, shifted_processor, "/maps", previous_root,
        )

    assert original_path == metric_cache_path(previous_root, entry["log_name"], entry["token"])
    assert shifted_path == metric_cache_path(previous_root, entry["log_name"], entry["shift_token"])
    assert load_cache.call_count == 2
    build_scene.assert_not_called()