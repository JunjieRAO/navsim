import pandas as pd
import pytest

from scripts.evaluation.list_gt_failure_clusters import SAFETY_METRICS, coverage_order, failure_clusters


def data():
    config = {"train_test_split": {"reactive_all_mapping": [
        ["first", "previous", [["zero", "good"], ["other", "last"]]]
    ]}}
    candidates = pd.DataFrame([
        {"token": token, "proposal_index": index, "pdm_score": score}
        for token, score in (("zero", 0), ("good", .5), ("other", .5), ("last", .5))
        for index in range(64)
    ])
    scores = pd.DataFrame({"token": ["zero", "good", "other", "last"],
                           "weight": [.5, .5, .5, .5],
                           **{metric: [0, 1, 1, 1] for metric in SAFETY_METRICS.values()}})
    return config, candidates, scores


def test_lists_parent_cluster_and_all_members_not_just_failures():
    clusters = failure_clusters(*data())
    assert clusters.first_token.tolist() == ["first"]
    assert clusters.scenes.tolist() == [2]
    assert clusters.all_zero_scenes.tolist() == [1]
    assert clusters.all_zero_scene_tokens.tolist() == ["zero"]
    assert clusters.qualifying_scenes.tolist() == [1]
    assert clusters.qualifying_scene_tokens.tolist() == ["zero"]


def test_incomplete_and_duplicate_candidates_rejected():
    config, candidates, scores = data()
    with pytest.raises(ValueError, match="exactly 64"):
        failure_clusters(config, candidates.iloc[1:], scores)
    with pytest.raises(ValueError, match="Duplicate"):
        failure_clusters(config, pd.concat([candidates, candidates.iloc[:1]]), scores)


@pytest.mark.parametrize("weight, expected", [(0, 0), (.0009, 0), (.001, 0), (.00101, 1)])
def test_weight_threshold_is_strict_and_applies_to_failed_scene(weight, expected):
    config, candidates, scores = data()
    scores.loc[scores.token == "zero", "weight"] = weight
    assert len(failure_clusters(config, candidates, scores)) == expected


def test_representatives_cover_all_categories_before_filling():
    clusters = pd.DataFrame([
        {"first_token": str(index), **{f"{label}_zero_scenes": int(label in categories)
                                      for label in SAFETY_METRICS}}
        for index, categories in enumerate([{"NC"}, {"DAC"}, {"DDC"}, {"NC", "DAC"}, {"DDC", "TLC"}])
    ])
    ordered = coverage_order(clusters)
    assert ordered.first_token.tolist()[:2] == ["3", "4"]
    assert all(ordered.head(2)[f"{label}_zero_scenes"].sum() > 0 for label in SAFETY_METRICS)
    assert set(ordered.first_token) == set(clusters.first_token)


def test_empty_coverage_order():
    config, candidates, scores = data()
    scores.weight = 0
    assert failure_clusters(config, candidates, scores).empty