from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import mock_open, patch

import numpy as np
import pandas as pd
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from omegaconf import OmegaConf

from navsim.common.dataclasses import SensorConfig, Trajectory
from navsim.common.dataloader import SceneFilter, SceneLoader, filter_synthetic_scenes
from navsim.common.enums import SceneFrameType
from navsim.planning.script.run_pdm_score import build_pdm_score_tasks, run_pdm_score


class TestPDMScoreTasks(TestCase):
    def test_tasks_cover_cached_tokens_once_and_reuse_scene_indexes(self):
        original_scenes = {
            "original-a": [{"token": "end-a", "log_name": "log-a"}],
            "original-b": [{"token": "end-b", "log_name": "log-b"}],
        }
        synthetic_scenes = {
            "synthetic-a1": (Path("/scenes/a1.pkl"), "log-a"),
            "synthetic-a2": (Path("/scenes/a2.pkl"), "log-a"),
            "synthetic-b1": (Path("/scenes/b1.pkl"), "log-b"),
            "synthetic-b2": (Path("/scenes/b2.pkl"), "log-b"),
        }
        scene_loader = SimpleNamespace(
            scene_frames_dicts=original_scenes,
            synthetic_scenes=synthetic_scenes,
            get_tokens_list_per_log=lambda: {
                "log-a": ["original-a", "synthetic-a1", "synthetic-a2"],
                "log-b": ["original-b", "synthetic-b1", "synthetic-b2"],
            },
        )
        cached_tokens = ["original-a", "synthetic-a1", "synthetic-a2", "synthetic-b1", "synthetic-b2"]

        tasks = build_pdm_score_tasks(OmegaConf.create({}), scene_loader, cached_tokens, 2)

        self.assertEqual(Counter(token for task in tasks for token in task["tokens"]), Counter(cached_tokens))
        self.assertEqual(len(tasks), 3)
        self.assertTrue(all(len(task["tokens"]) <= 2 for task in tasks))

        with patch("navsim.common.dataloader.filter_scenes", side_effect=AssertionError("log reopened")), patch(
            "navsim.common.dataloader.filter_synthetic_scenes", side_effect=AssertionError("scenes rescanned")
        ):
            for task in tasks:
                loader = SceneLoader(
                    data_path=Path("/unused"),
                    original_sensor_path=None,
                    scene_filter=SceneFilter(include_synthetic_scenes=True),
                    original_scene_index=task["original_scenes"],
                    synthetic_scene_index=task["synthetic_scenes"],
                )
                self.assertEqual(set(loader.tokens), set(task["tokens"]))
                self.assertEqual(loader.get_tokens_list_per_log(), {task["log_file"]: task["tokens"]})

        self.assertTrue(any(not task["original_scenes"] for task in tasks))

    def test_invalid_task_size(self):
        scene_loader = SimpleNamespace(get_tokens_list_per_log=lambda: {})
        with self.assertRaises(ValueError):
            build_pdm_score_tasks(OmegaConf.create({}), scene_loader, [], 0)

    def test_synthetic_filter_uses_metadata_and_preserves_selection(self):
        scene_path = Path("/synthetic/scene.pkl")
        scene_metadata = {
            "log_name": "log-a",
            "scene_token": "scene",
            "map_name": "map",
            "initial_token": "synthetic-a",
            "num_history_frames": 4,
            "num_future_frames": 8,
            "corresponding_original_scene": "end-a",
        }
        scene_filter = SceneFilter(include_synthetic_scenes=True, log_names=["log-a"])

        with patch.object(Path, "iterdir", return_value=[scene_path]), patch.object(Path, "open", mock_open()), patch(
            "navsim.common.dataloader.pickle.load", return_value={"scene_metadata": scene_metadata}
        ), patch("navsim.common.dataloader.Scene.load_from_disk", side_effect=AssertionError("full scene loaded")):
            self.assertEqual(
                filter_synthetic_scenes(Path("/synthetic"), scene_filter, ["end-a"]),
                {"synthetic-a": (scene_path, "log-a")},
            )
            self.assertEqual(filter_synthetic_scenes(Path("/synthetic"), scene_filter, []), {})

            scene_filter.synthetic_scene_tokens = ["synthetic-a"]
            self.assertEqual(
                filter_synthetic_scenes(Path("/synthetic"), scene_filter, []),
                {"synthetic-a": (scene_path, "log-a")},
            )
            scene_filter.log_names = ["log-b"]
            self.assertEqual(filter_synthetic_scenes(Path("/synthetic"), scene_filter, []), {})


class TestPrecomputedPDMScore(TestCase):
    def test_worker_scores_both_stages_without_loading_agent_or_sensors(self):
        sampling = TrajectorySampling(num_poses=2, interval_length=0.5)
        trajectories = {
            token: Trajectory(np.array([[1, 0, 0], [2, 0, 0]], dtype=np.float32), sampling)
            for token in ("original", "synthetic")
        }
        cfg = OmegaConf.create(
            {
                "simulator": {},
                "scorer": {},
                "agent": {},
                "metric_cache_path": "/unused",
                "train_test_split": {"scene_filter": {}},
                "navsim_log_path": "/unused",
                "original_sensor_path": "/unused",
                "synthetic_sensor_path": "/unused",
                "synthetic_scenes_path": "/unused",
                "traffic_agents_policy": {"reactive": {}},
            }
        )
        scene_filter = SceneFilter(include_synthetic_scenes=True, reactive_synthetic_initial_tokens=["synthetic"])
        simulator = SimpleNamespace(proposal_sampling=sampling)
        scorer = SimpleNamespace(proposal_sampling=sampling)

        def metric_cache_for_token(token):
            return SimpleNamespace(
                token=token,
                scene_type=SceneFrameType.ORIGINAL if token == "original" else SceneFrameType.SYNTHETIC,
                log_name="log-a",
                timepoint=SimpleNamespace(time_s=0),
                ego_state=SimpleNamespace(rear_axle=StateSE2(0, 0, 0)),
            )

        cache_loader = SimpleNamespace(tokens=list(trajectories), get_from_token=metric_cache_for_token)
        task = {
            "cfg": cfg,
            "log_file": "log-a",
            "tokens": list(trajectories),
            "original_scenes": {"original": [{"token": "end", "log_name": "log-a"}]},
            "synthetic_scenes": {"synthetic": (Path("/unused/synthetic.pkl"), "log-a")},
            "model_trajectories": trajectories,
        }
        scored = {}

        def fake_score(*, metric_cache, model_trajectory, **kwargs):
            scored[metric_cache.token] = model_trajectory
            return pd.DataFrame([{"pdm_score": 0.5}]), None

        with patch(
            "navsim.planning.script.run_pdm_score.instantiate",
            side_effect=[simulator, scorer, scene_filter, object(), object()],
        ) as instantiate, patch(
            "navsim.planning.script.run_pdm_score.MetricCacheLoader", return_value=cache_loader
        ), patch(
            "navsim.planning.script.run_pdm_score.SceneLoader", wraps=SceneLoader
        ) as loader, patch(
            "navsim.planning.script.run_pdm_score.pdm_score", side_effect=fake_score
        ), patch(
            "navsim.planning.script.run_pdm_score.relative_to_absolute_poses",
            return_value=[StateSE2(0, 0, 0)],
        ):
            rows = run_pdm_score([task])

        self.assertEqual(instantiate.call_count, 5)
        self.assertEqual(loader.call_args.kwargs["sensor_config"], SensorConfig.build_no_sensors())
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["token"].iloc[0] for row in rows}, set(trajectories))
        self.assertTrue(all(row["valid"].iloc[0] for row in rows))
        self.assertEqual(scored, trajectories)