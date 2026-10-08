import numpy as np


DISTANCE_KINDS = ("ade", "fde")


def proposal_distance(proposals: np.ndarray, ground_truth: np.ndarray, kind: str = "ade") -> np.ndarray:
    proposals = np.asarray(proposals, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    if kind not in DISTANCE_KINDS:
        raise ValueError(f"Distance kind must be one of {DISTANCE_KINDS}")
    if proposals.ndim != 3 or proposals.shape[-1] != 3:
        raise ValueError("Expected proposals with shape [candidates, poses, 3]")
    if ground_truth.shape != proposals.shape[1:]:
        raise ValueError("GT and proposals must have identical pose sampling")
    if not np.isfinite(proposals).all() or not np.isfinite(ground_truth).all():
        raise ValueError("GT and proposals must be finite")
    displacement = np.linalg.norm(proposals[..., :2] - ground_truth[None, :, :2], axis=-1)
    return displacement.mean(axis=-1) if kind == "ade" else displacement[:, -1]


def validate_scores(scores: np.ndarray, distance: np.ndarray) -> tuple:
    scores = np.asarray(scores, dtype=np.float64)
    distance = np.asarray(distance, dtype=np.float64)
    if scores.ndim != 1 or not scores.size or scores.shape != distance.shape:
        raise ValueError("Scores and distances must be matching nonempty vectors")
    if not np.isfinite(scores).all() or not np.isfinite(distance).all() or (distance < 0).any():
        raise ValueError("Scores must be finite and distances finite and nonnegative")
    return scores, distance


def select_proposal(scores: np.ndarray, distance: np.ndarray, strength: float = 0.0, min_distance: bool = False) -> int:
    scores, distance = validate_scores(scores, distance)
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("Lambda must be finite and nonnegative")
    return int(np.argmin(distance) if min_distance else np.argmax(scores - strength * distance))


def switching_threshold(scores: np.ndarray, distance: np.ndarray) -> float:
    scores, distance = validate_scores(scores, distance)
    baseline = int(np.argmax(scores))
    closer = distance < distance[baseline]
    if not closer.any():
        return float("inf")
    return float(np.min((scores[baseline] - scores[closer]) / (distance[baseline] - distance[closer])))


def calibrate_lambda(score_vectors, distance_vectors) -> tuple:
    thresholds = np.array([switching_threshold(scores, distance) for scores, distance in zip(score_vectors, distance_vectors)])
    positive = thresholds[np.isfinite(thresholds) & (thresholds > 0)]
    reference = float(np.median(positive)) if positive.size else None
    return reference, thresholds