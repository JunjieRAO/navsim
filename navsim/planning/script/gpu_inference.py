import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import Trajectory
from navsim.common.dataloader import FrameList, SceneLoader
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder


class EvaluationFeatures(Dataset):
    def __init__(self, scene_loader: SceneLoader, tokens: List[str], feature_builders: List[AbstractFeatureBuilder]):
        self.scene_loader = scene_loader
        self.tokens = tokens
        self.feature_builders = feature_builders

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, index: int) -> Tuple[str, Dict[str, torch.Tensor]]:
        token = self.tokens[index]
        agent_input = self.scene_loader.get_agent_input_from_token(token)
        features: Dict[str, torch.Tensor] = {}
        for builder in self.feature_builders:
            features.update(builder.compute_features(agent_input))
        return token, features


def predict_trajectories(
    agent: AbstractAgent,
    scene_loader: SceneLoader,
    tokens: List[str],
    device: torch.device,
    batch_size: int,
    num_workers: int = 0,
) -> Dict[str, Trajectory]:
    if batch_size < 1 or num_workers < 0:
        raise ValueError("GPU prediction batch size must be positive and data workers must be non-negative")
    if agent.requires_scene:
        raise ValueError("GPU prediction requires an agent that does not use privileged scene inputs")

    agent.to(device).eval()
    dataset = EvaluationFeatures(scene_loader, tokens, agent.get_feature_builders())
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    trajectories: Dict[str, Trajectory] = {}
    with torch.inference_mode():
        for batch_tokens, batch_features in tqdm(dataloader, desc="Predicting trajectories"):
            features = {key: value.to(device, non_blocking=True) for key, value in batch_features.items()}
            poses = agent.forward(features)["trajectory"].detach().cpu().numpy()
            for token, trajectory_poses in zip(batch_tokens, poses):
                trajectories[token] = Trajectory(trajectory_poses, agent._trajectory_sampling)

    return trajectories


def predict_proposals(
    agent: AbstractAgent,
    scene_loader: SceneLoader,
    tokens: List[str],
    device: torch.device,
    batch_size: int,
    num_workers: int = 0,
) -> Dict[str, np.ndarray]:
    if batch_size < 1 or num_workers < 0:
        raise ValueError("GPU prediction batch size must be positive and data workers must be non-negative")
    if agent.requires_scene:
        raise ValueError("GPU prediction requires an agent that does not use privileged scene inputs")
    if len(set(tokens)) != len(tokens):
        raise ValueError("Prediction tokens must be unique")

    agent.to(device).eval()
    dataset = EvaluationFeatures(scene_loader, tokens, agent.get_feature_builders())
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    proposals: Dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for batch_tokens, batch_features in tqdm(dataloader, desc="Predicting proposals"):
            features = {key: value.to(device, non_blocking=True) for key, value in batch_features.items()}
            predictions = agent.forward(features)
            if "proposals" not in predictions:
                raise ValueError("Agent must return proposals for oracle evaluation")
            poses = predictions["proposals"].detach().cpu().numpy()
            expected_shape = (len(batch_tokens), 64, agent._trajectory_sampling.num_poses, 3)
            if poses.shape != expected_shape or not np.isfinite(poses).all():
                raise ValueError(f"Expected finite proposals with shape {expected_shape}, got {poses.shape}")
            for token, scene_proposals in zip(batch_tokens, poses):
                proposals[token] = scene_proposals

    return proposals


def _predict_on_gpu(
    cfg: DictConfig,
    tokens: List[str],
    original_scenes: Dict[str, FrameList],
    synthetic_scenes: Dict[str, Tuple[Path, str]],
    device_index: int,
    all_proposals: bool = False,
) -> Union[Dict[str, Trajectory], Dict[str, np.ndarray]]:
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device)
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    scene_loader = SceneLoader(
        synthetic_sensor_path=Path(cfg.synthetic_sensor_path),
        original_sensor_path=Path(cfg.original_sensor_path),
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=agent.get_sensor_config(),
        synthetic_scene_index=synthetic_scenes,
        original_scene_index=original_scenes,
    )
    if all_proposals:
        return predict_proposals(agent, scene_loader, tokens, device, cfg.gpu_batch_size, cfg.gpu_num_workers)
    return predict_trajectories(agent, scene_loader, tokens, device, cfg.gpu_batch_size, cfg.gpu_num_workers)


def predict_trajectories_multi_gpu(
    cfg: DictConfig,
    scene_loader: SceneLoader,
    tokens: List[str],
    device: torch.device,
    num_devices: int,
) -> Dict[str, Trajectory]:
    start_index = device.index or 0
    if device.type != "cuda" or num_devices < 1 or start_index + num_devices > torch.cuda.device_count():
        raise ValueError("GPU prediction requires the requested number of visible CUDA devices")

    shards = [tokens[index::num_devices] for index in range(num_devices)]
    jobs = [
        (
            cfg,
            shard,
            {token: scene_loader.scene_frames_dicts[token] for token in shard if token in scene_loader.scene_frames_dicts},
            {token: scene_loader.synthetic_scenes[token] for token in shard if token in scene_loader.synthetic_scenes},
            start_index + index,
        )
        for index, shard in enumerate(shards)
        if shard
    ]
    trajectories: Dict[str, Trajectory] = {}
    if not jobs:
        return trajectories
    with ProcessPoolExecutor(max_workers=len(jobs), mp_context=multiprocessing.get_context("spawn")) as pool:
        for predictions in pool.map(_predict_on_gpu, *zip(*jobs)):
            trajectories.update(predictions)
    return trajectories


def predict_proposals_multi_gpu(
    cfg: DictConfig,
    scene_loader: SceneLoader,
    tokens: List[str],
    device: torch.device,
    num_devices: int,
) -> Dict[str, np.ndarray]:
    start_index = device.index or 0
    if device.type != "cuda" or num_devices < 1 or start_index + num_devices > torch.cuda.device_count():
        raise ValueError("GPU prediction requires the requested number of visible CUDA devices")
    if len(set(tokens)) != len(tokens):
        raise ValueError("Prediction tokens must be unique")

    shards = [tokens[index::num_devices] for index in range(num_devices)]
    jobs = [
        (
            cfg,
            shard,
            {token: scene_loader.scene_frames_dicts[token] for token in shard if token in scene_loader.scene_frames_dicts},
            {token: scene_loader.synthetic_scenes[token] for token in shard if token in scene_loader.synthetic_scenes},
            start_index + index,
            True,
        )
        for index, shard in enumerate(shards)
        if shard
    ]
    proposals: Dict[str, np.ndarray] = {}
    if not jobs:
        return proposals
    with ProcessPoolExecutor(max_workers=len(jobs), mp_context=multiprocessing.get_context("spawn")) as pool:
        for predictions in pool.map(_predict_on_gpu, *zip(*jobs)):
            proposals.update(predictions)
    if set(proposals) != set(tokens):
        raise ValueError("GPU proposal inference did not return every token")
    return proposals