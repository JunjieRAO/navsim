import numpy as np


def proposal_ade(proposals: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    proposals = np.asarray(proposals, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    if proposals.ndim != 3 or proposals.shape[-1] != 3:
        raise ValueError("Expected proposals with shape [candidates, poses, 3]")
    if ground_truth.shape != proposals.shape[1:]:
        raise ValueError("GT and proposals must have identical pose sampling")
    if not np.isfinite(proposals).all() or not np.isfinite(ground_truth).all():
        raise ValueError("GT and proposals must be finite")
    return np.linalg.norm(proposals[..., :2] - ground_truth[None, :, :2], axis=-1).mean(axis=-1)


def validate_scores(scores: np.ndarray, ade: np.ndarray) -> tuple:
    scores = np.asarray(scores, dtype=np.float64)
    ade = np.asarray(ade, dtype=np.float64)
    if scores.ndim != 1 or not scores.size or scores.shape != ade.shape:
        raise ValueError("Scores and ADE must be matching nonempty vectors")
    if not np.isfinite(scores).all() or not np.isfinite(ade).all() or (ade < 0).any():
        raise ValueError("Scores must be finite and ADE finite and nonnegative")
    return scores, ade


def select_proposal(scores: np.ndarray, ade: np.ndarray, strength: float = 0.0, min_ade: bool = False) -> int:
    scores, ade = validate_scores(scores, ade)
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("Lambda must be finite and nonnegative")
    return int(np.argmin(ade) if min_ade else np.argmax(scores - strength * ade))


def switching_threshold(scores: np.ndarray, ade: np.ndarray) -> float:
    scores, ade = validate_scores(scores, ade)
    baseline = int(np.argmax(scores))
    closer = ade < ade[baseline]
    if not closer.any():
        return float("inf")
    return float(np.min((scores[baseline] - scores[closer]) / (ade[baseline] - ade[closer])))


def calibrate_lambda(score_vectors, ade_vectors) -> tuple:
    thresholds = np.array([switching_threshold(scores, ade) for scores, ade in zip(score_vectors, ade_vectors)])
    positive = thresholds[np.isfinite(thresholds) & (thresholds > 0)]
    reference = float(np.median(positive)) if positive.size else None
    return reference, thresholds