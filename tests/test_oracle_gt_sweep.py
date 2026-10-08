import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from omegaconf import OmegaConf

from navsim.common.dataclasses import Trajectory
from navsim.planning.script.oracle_gt_sweep import (
    evaluation_mappings, load_ground_truth, run_oracle_gt_sweep,
)


class TestOracleGTSweep(TestCase):
    def setUp(self):
        self.sampling = TrajectorySampling(num_poses=2, interval_length=0.5)
        self.cfg = OmegaConf.create({
            "gpu_inference": True,
            "output_dir": "/unused",
            "max_scenarios_per_task": 2,
            "agent": {"trajectory_sampling": {
                "_target_": "nuplan.planning.simulation.trajectory.trajectory_sampling.TrajectorySampling",
                "num_poses": 2, "interval_length": 0.5,
            }},
            "train_test_split": {"reactive_all_mapping": [["original", "previous", [["synthetic", "synthetic_previous"]]]]},
            "oracle_gt": {
                "missing_gt": "error", "gt_path": None, "prepare_only": False,
                "lambdas": [0, 1], "lambda_multipliers": [0, 1], "distance": "fde", "include_min_distance": True,
            },
        })
        self.tokens = ["original", "previous", "synthetic", "synthetic_previous"]
        self.loader = SimpleNamespace(
            tokens=self.tokens,
            scene_frames_dicts={"original": [], "previous": []},
            get_scene_from_token=lambda token: SimpleNamespace(
                get_future_trajectory=lambda count: Trajectory(np.zeros((2, 3)), self.sampling)
            ),
        )

    def test_missing_gt_requires_explicit_baseline_policy(self):
        with self.assertRaisesRegex(ValueError, "no human GT"):
            load_ground_truth(self.cfg, self.loader, self.tokens, self.sampling)
        self.cfg.oracle_gt.missing_gt = "baseline"
        ground_truth, sources = load_ground_truth(self.cfg, self.loader, self.tokens, self.sampling)
        self.assertEqual(set(ground_truth), {"original", "previous"})
        self.assertEqual(set(sources.values()), {"original_human_future"})

    def test_external_gt_covers_synthetic_and_checks_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "gt.npz"
            np.savez(archive, tokens=np.array(self.tokens[2:]), poses=np.zeros((2, 2, 3)), interval_length=0.5)
            self.cfg.oracle_gt.gt_path = str(archive)
            ground_truth, sources = load_ground_truth(self.cfg, self.loader, self.tokens, self.sampling)
            self.assertEqual(set(ground_truth), set(self.tokens))
            self.assertEqual(sources["synthetic"], "external_gt")
            with self.assertRaisesRegex(ValueError, "interval"):
                load_ground_truth(self.cfg, self.loader, self.tokens, TrajectorySampling(num_poses=2, interval_length=1))

    def test_mapping_coverage_is_exact_and_cache_complete(self):
        self.assertEqual(len(evaluation_mappings(self.cfg, self.tokens, self.tokens)), 1)
        with self.assertRaisesRegex(ValueError, "metric caches"):
            evaluation_mappings(self.cfg, self.tokens[:-1], self.tokens)
        with self.assertRaisesRegex(ValueError, "exactly"):
            evaluation_mappings(self.cfg, self.tokens[:-1], self.tokens[:-1])
        with self.assertRaisesRegex(ValueError, "exactly"):
            evaluation_mappings(self.cfg, self.tokens + ["unmapped"], self.tokens + ["unmapped"])

    def test_sweep_preserves_missing_gt_and_writes_comparable_summary(self):
        proposals = np.zeros((4, 2, 2, 3))
        proposals[:, 0, :, 0] = 5
        proposals[:, 1, :, 0] = 1
        scores = np.tile([2., 1.], (4, 1))
        evaluated = []

        def evaluate(cfg, loader, tasks, worker, tokens, trajectories, output):
            evaluated.append({token: trajectory.poses[0, 0] for token, trajectory in trajectories.items()})
            return dict(combined_score=0.5 + 0.1 * (len(evaluated) - 1), stage_one_score=0.8, stage_two_score=0.7, valid_scenes=4)

        self.cfg.oracle_gt.missing_gt = "baseline"
        with tempfile.TemporaryDirectory() as directory:
            self.cfg.output_dir = directory
            with patch("navsim.planning.script.oracle_gt_sweep.load_or_predict_candidates", return_value=(proposals, scores)), patch(
                "navsim.planning.script.oracle_gt_sweep.build_worker"
            ), patch("navsim.planning.script.run_pdm_score.build_pdm_score_tasks", return_value=[]), patch(
                "navsim.planning.script.oracle_gt_sweep.evaluate_selection", side_effect=evaluate
            ):
                run_oracle_gt_sweep(self.cfg, self.loader, self.tokens)
            summary = pd.read_csv(Path(directory) / "oracle_summary.csv")
            self.assertEqual(len(summary), 3)
            np.testing.assert_allclose(summary["delta_baseline"], [0, .1, .2])
            np.testing.assert_array_equal(summary["gt_scenes"].to_numpy(), [2, 2, 2])
            np.testing.assert_allclose(summary["mean_selected_distance_m"], [5, 1, 1])
            self.assertEqual(evaluated[0]["original"], 5)
            self.assertEqual(evaluated[1]["original"], 1)
            self.assertTrue(all(result["synthetic"] == 5 for result in evaluated))
            self.assertTrue((summary["scope"] == "partial_oracle_missing_gt_keeps_baseline").all())