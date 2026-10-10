from unittest.mock import patch
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from omegaconf import OmegaConf

from navsim.planning.script.gt_two_stage_experiment import checked_rows, score_candidate_task
from navsim.planning.script.gt_two_stage_experiment import run_gt_two_stage_experiment
from navsim.planning.script.gt_two_stage_diagnostics import SAFETY_METRICS


def test_gt_priority_and_exhaustive_actual_selection_with_stable_ties():
    sampling = TrajectorySampling(num_poses=2, interval_length=.5)
    proposals = np.zeros((3, 2, 3))
    proposals[:, :, 0] = np.arange(3)[:, None]
    ground_truth = np.full((2, 3), 10.0)
    task = {
        "cfg": OmegaConf.create({"agent": {"trajectory_sampling": {}}}),
        "tokens": ["original", "synthetic"],
        "candidate_proposals": {token: proposals for token in ("original", "synthetic")},
        "candidate_scores": {token: np.array([0, 1, 2]) for token in ("original", "synthetic")},
        "ground_truth": {"original": ground_truth},
        "original_scenes": {"original": []}, "synthetic_scenes": {"synthetic": None},
    }
    visits = []

    def fake_score(args):
        job = args[0]
        rows = []
        for token in job["tokens"]:
            poses = job["model_trajectories"][token].poses
            if token == "original":
                np.testing.assert_array_equal(poses, ground_truth)
                score = 1.0
            else:
                proposal_index = int(poses[0, 0])
                visits.append(proposal_index)
                score = [.9, .9, .4][proposal_index]
            rows.append(pd.DataFrame([{"token": token, "valid": True, "pdm_score": score}]))
        return rows

    with patch("navsim.planning.script.gt_two_stage_experiment.instantiate", return_value=sampling), patch(
        "navsim.planning.script.run_pdm_score.run_pdm_score", side_effect=fake_score
    ):
        result = score_candidate_task([task])
    assert visits == [2, 0, 1]
    selection = pd.DataFrame(result["selection"]).set_index("token")
    assert selection.loc["original", "actual_pdm_index"] == -1
    assert selection.loc["synthetic", "model_index"] == 2
    assert selection.loc["synthetic", "actual_pdm_index"] == 0
    assert result["model"].set_index("token").loc["synthetic", "pdm_score"] == .4
    assert result["actual"].set_index("token").loc["synthetic", "pdm_score"] == .9
    assert len(result["candidate_scores"]) == 3


@pytest.mark.parametrize("rows", [
    [{"token": "scene", "valid": False, "pdm_score": .5}],
    [{"token": "other", "valid": True, "pdm_score": .5}],
    [{"token": "scene", "valid": True, "pdm_score": np.nan}],
    [{"token": "scene", "valid": True, "pdm_score": .5}] * 2,
])
def test_failed_incomplete_duplicate_or_nonfinite_evaluation_is_rejected(rows):
    with pytest.raises((RuntimeError, ValueError)):
        checked_rows([pd.DataFrame(rows)], ["scene"])


def test_experiment_writes_both_settings_and_coverage_groups(tmp_path):
    tokens = ["original", "previous", "second_a", "second_b"]
    cfg = OmegaConf.create({
        "gt_two_stage": {"candidate_cache_path": "cache", "gt_cache_path": None},
        "navsim_log_path": "logs", "synthetic_scenes_path": "synthetic",
        "metric_cache_path": "metrics", "output_dir": str(tmp_path),
        "max_scenarios_per_task": 2,
        "agent": {"trajectory_sampling": {}}, "simulator": {"proposal_sampling": {}},
        "train_test_split": {"scene_filter": {},
                             "reactive_all_mapping": [["original", "previous", [["second_a", "second_b"]]]]},
    })
    rows = pd.DataFrame([
        {"token": token, "valid": True, "pdm_score": .5, "score": .5, "weight": 1,
         "endpoint_x": 10 if token == "previous" else 0, "endpoint_y": 0,
         "start_point_x": 0, "start_point_y": 0,
         **{metric: 1 for metric in SAFETY_METRICS}}
        for token in tokens
    ])
    worker = SimpleNamespace(map=lambda *args: [{
        "model": rows, "actual": rows,
        "candidate_scores": [{"token": "second_a", "proposal_index": 0, "pdm_score": .5}],
        "selection": [{"token": token, "has_gt": token in ("original", "previous"),
                       "model_index": 0, "actual_pdm_index": 0} for token in tokens],
    }])
    proposals = {token: np.zeros((2, 2, 3)) for token in tokens}
    scores = {token: np.ones(2) for token in tokens}
    with patch("navsim.planning.script.gt_two_stage_experiment.SceneLoader", return_value=SimpleNamespace(tokens=tokens)), patch(
        "navsim.planning.script.gt_two_stage_experiment.MetricCacheLoader", return_value=SimpleNamespace(tokens=tokens)
    ), patch("navsim.planning.script.gt_two_stage_experiment.instantiate"), patch(
        "navsim.planning.script.gt_two_stage_experiment.load_candidates", return_value=(proposals, scores)
    ), patch("navsim.planning.script.gt_two_stage_experiment.load_ground_truth", return_value={
        token: np.zeros((2, 3)) for token in ("original", "previous")
    }), patch("navsim.planning.script.gt_two_stage_experiment.build_worker", return_value=worker), patch(
        "navsim.planning.script.run_pdm_score.build_pdm_score_tasks", return_value=[{"tokens": tokens}]
    ), patch("navsim.planning.script.run_pdm_score.create_scene_aggregators", side_effect=lambda mappings, raw, sampling: raw), patch(
        "navsim.planning.script.run_pdm_score.compute_final_scores", side_effect=lambda raw: raw
    ):
        run_gt_two_stage_experiment(cfg)
    summary = json.loads((tmp_path / "gt_two_stage_summary.json").read_text())
    assert summary["gt_scenes"] == 2
    for setting in ("model", "actual"):
        groups = summary["settings"][f"gt_{setting}_top1"]["groups"]
        assert groups["all"]["combined_score"] == .25
        assert groups["non_fallback"]["branches"] == 1
        assert groups["fallback"]["branches"] == 1
        assert (tmp_path / f"gt_{setting}_top1_scores.csv").exists()
        assert (tmp_path / f"gt_{setting}_top1_safety_zero_causes.csv").exists()