import json
import os
import pickle
import subprocess
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf

from navsim.evaluate.continuation_cache import source_stamp, write_json_atomic
from navsim.evaluate.continuation_value import ContinuationMode
from navsim.planning.script.run_continuation_value import (
    _cache_log_job, _cached_proposal, _cached_score, _infer_gpu_shard, _inference_shards,
    _score_fingerprint, _scoring_worker_config, _stored_score_fingerprint, analyze_scores, score_scenes,
)
from navsim.planning.script.run_two_stage_oracle import token_artifact_path, write_npz_atomic


def test_proposal_artifact_requires_correct_checkpoint_log_and_shape(tmp_path):
    path = tmp_path / "tokens" / "scene.npz"
    write_npz_atomic(path, {
        "proposals": np.ones((64, 8, 3)),
        "fingerprint": np.asarray("checkpoint-v1"),
        "log_stamp": np.array([100, 200]),
    })

    assert _cached_proposal(path, "checkpoint-v1", [100, 200], 8)
    assert not _cached_proposal(path, "checkpoint-v2", [100, 200], 8)
    assert not _cached_proposal(path, "checkpoint-v1", [100, 300], 8)
    assert not _cached_proposal(path, "checkpoint-v1", [100, 200], 10)


def test_score_artifact_requires_complete_matrix_and_current_inputs(tmp_path):
    path = tmp_path / "tokens" / "scene.npz"
    stamps = {"original_stamp": [1, 2], "shifted_stamp": [3, 4], "proposal_stamp": [5, 6]}
    arrays = {
        "continuation_scores": np.ones((64, 32)),
        "metric_values": np.ones((64, 32, 8)),
        "terminal_overlap": np.zeros(64, dtype=bool),
        "e1_scores": np.ones(64),
        "e1_metrics": np.ones((64, 8)),
        "endpoint_states": np.ones((64, 4)),
        "fingerprint": np.asarray("score-v1"),
        **{key: np.asarray(stamp) for key, stamp in stamps.items()},
    }
    write_npz_atomic(path, arrays)

    assert _cached_score(path, "score-v1", stamps, 32)
    assert not _cached_score(path, "score-v1", stamps, 64)
    assert not _cached_score(path, "score-v2", stamps, 32)
    assert not _cached_score(path, "score-v1", {**stamps, "shifted_stamp": [3, 5]}, 32)


def test_analysis_reuses_saved_scores_for_different_k_lists(tmp_path):
    cfg = OmegaConf.create({"v2_root": str(tmp_path), "max_k": 64, "k_values": [16, 32, 64]})
    entry = {"token": "scene", "log_file": "train_log.pkl", "log_name": "train_log.db", "reason": "eligible"}
    index = {"entries": [entry], "eligible": 1}
    stamps = {"original_stamp": [1, 2], "shifted_stamp": [3, 4], "proposal_stamp": [5, 6]}
    scores = np.zeros((64, 64))
    scores[0, 15], scores[1, 31], scores[0, 63] = 0.9, 0.95, 0.99
    score_path = token_artifact_path(tmp_path / "scores", "scene")
    write_json_atomic(tmp_path / "scores" / "manifest.json", {
        "fingerprint": "scoring", "max_k": 64, "metric": "EPDMS-no-EC",
    })
    write_npz_atomic(score_path, {
        "continuation_scores": scores,
        "metric_values": np.ones((64, 64, 8)),
        "terminal_overlap": np.zeros(64, dtype=bool),
        "e1_scores": np.ones(64),
        "e1_metrics": np.ones((64, 8)),
        "endpoint_states": np.ones((64, 4)),
        "fingerprint": np.asarray("scoring"),
        **{key: np.asarray(value) for key, value in stamps.items()},
    })

    with patch("navsim.planning.script.run_continuation_value._index", return_value=index), patch(
        "navsim.planning.script.run_continuation_value._inference_fingerprint", return_value="inference",
    ), patch("navsim.planning.script.run_continuation_value._cache_fingerprint", return_value="cache"), patch(
        "navsim.planning.script.run_continuation_value._score_fingerprint", return_value="scoring",
    ), patch("navsim.planning.script.run_continuation_value._require_manifest"), patch(
        "navsim.planning.script.run_continuation_value._score_input_stamps", return_value=stamps,
    ):
        analyze_scores(cfg)
        cfg.k_values = [16, 64]
        analyze_scores(cfg)

    with np.load(token_artifact_path(tmp_path / "analysis" / "top25" / "k_16_32_64", "scene")) as data:
        assert data["v2_values"].shape == (64, 3)
        np.testing.assert_array_equal(data["top_counts"], [4, 8, 16])
        np.testing.assert_array_equal(data["v2_top_indices"], [0, 1, 0])
    with np.load(token_artifact_path(tmp_path / "analysis" / "top25" / "k_16_64", "scene")) as data:
        assert data["v2_values"].shape == (64, 2)
    with (tmp_path / "analysis" / "top25" / "k_16_32_64" / "summary.json").open() as stream:
        summary = json.load(stream)
    assert summary["analyzed_scenes"] == 1
    assert summary["value_target"] == "top25_mean"
    assert summary["top_counts"] == [4, 8, 16]
    assert summary["by_k"]["32"]["v2_top1_agreement"] == 0.0
    assert summary["by_k"]["32"]["mean_v2_top_ties"] == 1.0


def test_legacy_score_manifest_reuses_only_matching_scoring_settings(tmp_path):
    cfg = OmegaConf.create({
        "v2_root": str(tmp_path), "max_k": 32,
        "proposal_sampling": {"num_poses": 40, "interval_length": 0.1},
        "agent": {"trajectory_sampling": {"num_poses": 8, "interval_length": 0.5}},
        "simulator": {"model": "pdm"}, "scorer": {"progress_weight": 5.0},
        "traffic_agents_policy": {"non_reactive": {"_target_": "log_replay"}},
    })
    write_json_atomic(tmp_path / "scores" / "manifest.json", {
        "fingerprint": "old-score-fingerprint", "max_k": 32, "metric": "EPDMS-no-EC",
    })
    snapshot = tmp_path / "hydra" / "score" / "code" / "hydra" / "config.yaml"
    snapshot.parent.mkdir(parents=True)
    OmegaConf.save(cfg, snapshot)

    assert _stored_score_fingerprint(cfg, "new-score-fingerprint") == "old-score-fingerprint"
    cfg.scorer.progress_weight = 6.0
    with np.testing.assert_raises_regex(ValueError, "Stored scoring settings differ"):
        _stored_score_fingerprint(cfg, "new-score-fingerprint")


def test_score_fingerprint_excludes_target_aggregation_but_includes_modes():
    cfg = OmegaConf.create({
        "max_k": 32, "proposal_sampling": {"num_poses": 40, "interval_length": 0.1},
        "agent": {"trajectory_sampling": {"num_poses": 8, "interval_length": 0.5}},
        "simulator": {"model": "pdm"}, "scorer": {"progress_weight": 5.0},
        "traffic_agents_policy": {"non_reactive": {"_target_": "log_replay"}},
    })
    with patch("navsim.planning.script.run_continuation_value._cache_fingerprint", return_value="cache"):
        original = _score_fingerprint(cfg, {}, "inference")
        with patch("navsim.evaluate.continuation_value.continuation_values", return_value=None):
            assert _score_fingerprint(cfg, {}, "inference") == original
        with patch("navsim.planning.script.run_continuation_value.CONTINUATION_MODES", (ContinuationMode(5, 1),)):
            assert _score_fingerprint(cfg, {}, "inference") != original


def test_gpu_shards_balance_logs_and_persist_model_across_scene_chunks(tmp_path):
    data_root = tmp_path / "logs"
    data_root.mkdir()
    assignments = []
    stamps = {}
    for log_name in ("first.pkl", "second.pkl"):
        with (data_root / log_name).open("wb") as stream:
            pickle.dump([{"token": log_name}], stream)
        stamps[log_name] = source_stamp(data_root / log_name)
        assignments.append((log_name, [{"token": f"{log_name[:-4]}-{index}"} for index in range(3)]))
    shards = _inference_shards(dict(assignments), [0, 1, 2, 3])
    assert set(shards) == {0, 1}

    cfg = OmegaConf.create({
        "v2_root": str(tmp_path), "original_sensor_path": str(tmp_path),
        "gpu_inference_chunk_scenes": 2, "gpu_batch_size": 16, "gpu_num_workers": 8,
        "agent": {"_target_": "unused"}, "train_test_split": {"scene_filter": {"_target_": "unused"}},
    })
    model = SimpleNamespace(initialize=lambda: None, get_sensor_config=lambda: "cameras")
    calls = []

    def predict(agent, loader, tokens, device, batch_size, num_workers):
        assert agent is model and loader.sensor_config == "cameras" and device.type == "cpu"
        calls.append(list(tokens))
        assert (batch_size, num_workers) == (16, 8)
        return {token: np.zeros((64, 8, 3), dtype=np.float32) for token in tokens}

    with patch("navsim.planning.script.run_continuation_value.instantiate", side_effect=[model, object()]) as create, patch(
        "navsim.planning.script.run_continuation_value.frames_for_entry",
        side_effect=lambda frames, entry, window: ([entry], []),
    ), patch("navsim.planning.script.run_continuation_value.SceneLoader", side_effect=lambda **kwargs: SimpleNamespace(
        sensor_config=kwargs["sensor_config"],
    )), patch("navsim.planning.script.run_continuation_value.predict_proposals", side_effect=predict):
        complete, failures = _infer_gpu_shard(
            cfg, assignments, "fingerprint", None, str(data_root), stamps,
            {"history_frames": 4, "original_future_frames": 10},
        )

    assert create.call_count == 2
    assert complete == 6 and not failures
    assert [len(chunk) for chunk in calls] == [2, 2, 2]
    for log_name, entries in assignments:
        for entry in entries:
            path = token_artifact_path(tmp_path / "proposals", entry["token"])
            assert _cached_proposal(path, "fingerprint", stamps[log_name], 8)


def test_log_cache_worker_reuses_processors_and_isolates_scene_failures(tmp_path):
    path = tmp_path / "drive.pkl"
    with path.open("wb") as stream:
        pickle.dump([{"token": "frame"}], stream)
    entries = [{"token": "good"}, {"token": "bad"}]
    processors = []

    def new_processor(cache_path, force, sampling):
        processor = SimpleNamespace(cache_path=cache_path)
        processors.append(processor)
        return processor

    def save_cache(frames, entry, window, original_processor, shifted_processor, map_root, v1_root):
        assert frames == [{"token": "frame"}]
        assert original_processor is processors[0] and shifted_processor is processors[1]
        if entry["token"] == "bad":
            raise ValueError("bad scene")

    with patch("navsim.planning.script.run_continuation_value.MetricCacheProcessor", side_effect=new_processor), patch(
        "navsim.planning.script.run_continuation_value.cache_shared_scene", side_effect=save_cache,
    ):
        complete, failures = _cache_log_job(
            path, entries, source_stamp(path), {"history_frames": 4}, object(), tmp_path / "output", None, "/maps",
        )

    assert len(processors) == 2
    assert complete == 1 and failures == {"bad": "bad scene"}


def test_cache_stage_caps_pool_at_number_of_logs(tmp_path):
    from navsim.planning.script.run_continuation_value import cache_gt

    log_names = ("one.pkl", "two.pkl")
    index = {
        "entries": [
            {"token": f"scene-{name}", "log_file": name, "reason": "eligible"} for name in log_names
        ],
        "data_root": str(tmp_path), "source_stamps": {name: [1, 2] for name in log_names},
        "history_frames": 4, "original_future_frames": 10, "eligible": 2,
    }
    cfg = OmegaConf.create({
        "v2_root": str(tmp_path), "v2_workers": 72, "v1_cache_root": None,
        "proposal_sampling": {"_target_": "unused"},
    })
    created = []

    class InlinePool:
        def __init__(self, max_workers, mp_context):
            created.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, *args):
            future = Future()
            future.set_result(function(*args))
            return future

    with patch.dict(os.environ, {"NUPLAN_MAPS_ROOT": str(tmp_path)}), patch(
        "navsim.planning.script.run_continuation_value._index", return_value=index,
    ), patch("navsim.planning.script.run_continuation_value.instantiate", return_value=object()), patch(
        "navsim.planning.script.run_continuation_value._cache_fingerprint", return_value="cache",
    ), patch("navsim.planning.script.run_continuation_value._manifest"), patch(
        "navsim.planning.script.run_continuation_value._cache_log_job", return_value=(1, {}),
    ), patch("navsim.planning.script.run_continuation_value.ProcessPoolExecutor", InlinePool):
        cache_gt(cfg)

    with (tmp_path / "cache_status.json").open() as stream:
        status = json.load(stream)
    assert created == [2]
    assert status == {"complete": 2, "failed": {}}


def test_scoring_tasks_omit_the_large_navtrain_filter(tmp_path):
    cfg = OmegaConf.create({
        "v2_root": str(tmp_path), "v1_cache_root": None, "max_k": 64,
        "train_test_split": {"scene_filter": {"tokens": ["token"] * 100_000}},
        "agent": {"trajectory_sampling": {"time_horizon": 4, "interval_length": 0.5}},
        "scorer": {"progress_weight": 5.0},
        "simulator": {"proposal_sampling": {"num_poses": 40, "interval_length": 0.1}},
        "traffic_agents_policy": {"non_reactive": {"_target_": "log_replay"}},
    })

    worker_cfg = _scoring_worker_config(cfg)

    assert "train_test_split" not in worker_cfg
    assert len(pickle.dumps(worker_cfg)) < 4096
    assert worker_cfg.max_k == 64
    assert worker_cfg.simulator.proposal_sampling.num_poses == 40
    assert worker_cfg.traffic_agents_policy.non_reactive._target_ == "log_replay"


def test_scoring_pool_is_capped_by_pending_scenes(tmp_path):
    entries = [
        {"token": "scene-one", "log_file": "one.pkl", "reason": "eligible"},
        {"token": "scene-two", "log_file": "two.pkl", "reason": "eligible"},
    ]
    index = {
        "entries": entries, "source_stamps": {"one.pkl": [1, 2], "two.pkl": [3, 4]}, "eligible": 2,
    }
    cfg = OmegaConf.create({"v2_root": str(tmp_path), "max_k": 1, "v2_workers": 72})
    created = []

    class InlinePool:
        def __init__(self, max_workers, mp_context):
            created.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, *args):
            future = Future()
            future.set_result(function(*args))
            return future

    with patch("navsim.planning.script.run_continuation_value._index", return_value=index), patch(
        "navsim.planning.script.run_continuation_value._inference_fingerprint", return_value="inference",
    ), patch("navsim.planning.script.run_continuation_value._cache_fingerprint", return_value="cache"), patch(
        "navsim.planning.script.run_continuation_value._score_fingerprint", return_value="score",
    ), patch("navsim.planning.script.run_continuation_value._require_manifest"), patch(
        "navsim.planning.script.run_continuation_value._manifest",
    ), patch("navsim.planning.script.run_continuation_value._score_settings", return_value={}), patch(
        "navsim.planning.script.run_continuation_value._score_input_stamps", return_value={},
    ), patch(
        "navsim.planning.script.run_continuation_value._cached_score", return_value=False,
    ), patch("navsim.planning.script.run_continuation_value._scoring_worker_config", return_value=object()), patch(
        "navsim.planning.script.run_continuation_value.score_scene_job", return_value={},
    ), patch("navsim.planning.script.run_continuation_value.ProcessPoolExecutor", InlinePool):
        score_scenes(cfg)

    assert created == [2]
    with (tmp_path / "score_status.json").open() as stream:
        assert json.load(stream) == {"complete": 2, "failed": {}}


def test_first_step_shell_launcher_builds_index_from_raw_log(tmp_path):
    log_root = tmp_path / "logs"
    log_root.mkdir()
    frames = [
        {"token": f"token-{index}", "timestamp": index * 500_000,
         "roadblock_ids": ["lane"], "log_name": "drive.db"}
        for index in range(21)
    ]
    with (log_root / "drive.pkl").open("wb") as stream:
        pickle.dump(frames, stream)

    workspace = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "V2_ROOT": str(tmp_path / "run"), "V2_MAX_SCENES": "1"}
    subprocess.run([
        "bash", "scripts/evaluation/run_drivor_v2_01_index.sh",
        f"navsim_log_path={log_root}", "train_test_split.scene_filter.log_names=null",
        "train_test_split.scene_filter.tokens=null",
    ], cwd=workspace, env=environment, check=True, capture_output=True, text=True)

    with (tmp_path / "run" / "index.json").open() as stream:
        index = json.load(stream)
    assert index["eligible"] == 1
    assert index["entries"][0]["shift_token"] == "token-11"
    assert index["entries"][0]["log_name"] == "drive.db"