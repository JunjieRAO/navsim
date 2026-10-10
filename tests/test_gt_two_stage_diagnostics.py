import numpy as np
import pandas as pd

from navsim.planning.script.gt_two_stage_diagnostics import SAFETY_METRICS, summarize_branches


def test_group_scores_and_overlapping_safety_causes():
    rows = []
    for token, endpoint, start, score, weight in (
        ("original", 0, 0, .8, 1), ("previous", 10, 0, 1, 1),
        ("second_a", 0, 0, .5, 1), ("second_b", 0, 0, 0, 1),
    ):
        rows.append({"token": token, "endpoint_x": endpoint, "endpoint_y": 0,
                     "start_point_x": start, "start_point_y": 0,
                     "score": score, "weight": weight,
                     **{metric: 0 if token == "second_b" else 1 for metric in SAFETY_METRICS}})
    summary, branches, safety = summarize_branches(
        pd.DataFrame(rows), {("original", "previous"): [("second_a", "second_b")]}
    )
    assert summary["non_fallback"]["branches"] == 1
    assert summary["fallback"]["branches"] == 1
    assert np.isclose(summary["all"]["combined_score"], .2)
    assert np.isclose(summary["all"]["stage_two_score"], .25)
    assert len(safety) == 4
    assert summary["fallback"]["safety_zero_causes"]["stage_two"]["drivable_area_compliance"] == {
        "zero_scene_count": 1, "weighted_zero_mass_per_branch": 1.0
    }
    assert branches.fallback.tolist() == [False, True]