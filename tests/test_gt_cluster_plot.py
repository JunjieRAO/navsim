import numpy as np
import pandas as pd
import pytest

from scripts.evaluation.plot_gt_stage2_cluster import local_poses, select_cluster
from nuplan.common.actor_state.state_representation import StateSE2


def cluster_data():
    config = {"train_test_split": {"reactive_all_mapping": [
        ["first", "previous", [["near", "other"], ["far", "last"]]]
    ]}}
    candidates = pd.DataFrame({"token": ["near", "near", "far", "other", "last"],
                               "pdm_score": [0, 0, .8, .4, .7]})
    scores = pd.DataFrame({"token": ["first", "previous", "near", "far", "other", "last"],
                          "endpoint_x": [0, 0, 0, 0, 0, 0], "endpoint_y": [0] * 6,
                          "start_point_x": [0, 0, 1, 5, 2, 3], "start_point_y": [0] * 6,
                          "weight": [1, 1, .99, .01, .5, .5]}).set_index("token")
    return config, candidates, scores


def test_default_cluster_includes_all_members_distance_sorted():
    first, members, distances = select_cluster(*cluster_data())
    assert first == "first"
    assert members == ["near", "far"]
    assert distances == {"near": 1, "far": 5}


def test_scene_token_selects_parent_and_rejects_nonfailure_cluster():
    assert select_cluster(*cluster_data(), scene_token="far")[0] == "first"
    with pytest.raises(ValueError, match="no all-candidate-zero"):
        select_cluster(*cluster_data(), first_token="previous")


def test_global_to_cluster_coordinates():
    result = local_poses(StateSE2(10, 20, np.pi / 2), [[10, 22, np.pi / 2]])
    np.testing.assert_allclose(result, [[2, 0, 0]], atol=1e-10)