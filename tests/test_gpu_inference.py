from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.planning.script.gpu_inference import predict_trajectories, predict_trajectories_multi_gpu


class FeatureBuilder:
    def compute_features(self, value):
        return {"features": torch.tensor([[value, 0, 0], [value, 1, 0]], dtype=torch.float32)}


class PredictionAgent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.requires_scene = False
        self._trajectory_sampling = TrajectorySampling(num_poses=2, interval_length=0.5)
        self.batch_sizes = []

    def get_feature_builders(self):
        return [FeatureBuilder()]

    def forward(self, features):
        self.batch_sizes.append(features["features"].shape[0])
        return {"trajectory": features["features"]}


class TestGPUInference(TestCase):
    def test_batches_features_and_maps_predictions_by_token(self):
        agent = PredictionAgent()
        scene_loader = type("SceneLoader", (), {"get_agent_input_from_token": lambda self, token: int(token)})()

        predictions = predict_trajectories(agent, scene_loader, ["3", "1", "2"], torch.device("cpu"), batch_size=2)

        self.assertEqual(agent.batch_sizes, [2, 1])
        self.assertEqual(set(predictions), {"3", "1", "2"})
        for token, trajectory in predictions.items():
            np.testing.assert_array_equal(
                trajectory.poses, [[int(token), 0, 0], [int(token), 1, 0]]
            )
            self.assertEqual(trajectory.trajectory_sampling, agent._trajectory_sampling)

    def test_rejects_invalid_batch_and_privileged_scene_agent(self):
        agent = PredictionAgent()
        for batch_size, num_workers in ((0, 0), (1, -1)):
            with self.assertRaises(ValueError):
                predict_trajectories(agent, None, [], torch.device("cpu"), batch_size, num_workers)

        agent.requires_scene = True
        with self.assertRaises(ValueError):
            predict_trajectories(agent, None, [], torch.device("cpu"), batch_size=1)

    def test_multi_gpu_shards_indexes_and_merges_predictions(self):
        tokens = ["original-0", "synthetic-0", "original-1", "synthetic-1", "original-2", "synthetic-2", "original-3", "synthetic-3"]
        scene_loader = SimpleNamespace(
            scene_frames_dicts={token: [{"token": token}] for token in tokens if token.startswith("original")},
            synthetic_scenes={token: (Path(f"/{token}.pkl"), "log") for token in tokens if token.startswith("synthetic")},
        )
        jobs = []

        def run_jobs(function, configs, shards, originals, synthetics, device_indexes):
            for shard, original_index, synthetic_index, device_index in zip(shards, originals, synthetics, device_indexes):
                jobs.append((shard, original_index, synthetic_index, device_index))
                yield {token: device_index for token in shard}

        with patch("navsim.planning.script.gpu_inference.ProcessPoolExecutor") as executor, patch(
            "navsim.planning.script.gpu_inference.torch.cuda.device_count", return_value=4
        ):
            executor.return_value.__enter__.return_value.map.side_effect = run_jobs
            predictions = predict_trajectories_multi_gpu(None, scene_loader, tokens, torch.device("cuda:0"), 4)

        self.assertEqual(set(predictions), set(tokens))
        self.assertEqual([device_index for _, _, _, device_index in jobs], [0, 1, 2, 3])
        for shard, originals, synthetics, device_index in jobs:
            self.assertEqual(set(shard), set(originals) | set(synthetics))
            self.assertEqual({predictions[token] for token in shard}, {device_index})

        with patch("navsim.planning.script.gpu_inference.torch.cuda.device_count", return_value=3):
            with self.assertRaises(ValueError):
                predict_trajectories_multi_gpu(None, scene_loader, tokens, torch.device("cuda:0"), 4)