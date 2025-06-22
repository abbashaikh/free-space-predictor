'''SceneProcessor class'''
from pathlib import Path
from typing import Optional, Dict, Any

import random
import numpy as np
import torch

from trajdata.data_structures.agent import AgentType
from traj_pred.modules import ModelRegistrar
from traj_pred import TrajectoryPredictor
from data_generation import (
    SceneAgentBatch,
    transform_coords_np,
    is_within_distance,
    get_constraint_coeffs,
    find_sample_size
)

import support_pred.halfplane_module as hm

class SceneProcessor:
    """
    Encapsulates loading and interfacing with a trained TrajectoryPredictor
    for a given SceneAgentBatch (only one scene per batch!).

    Notations:
    N: number of scenes
    M: number of agents per scene
    T: prediction horizon
    """

    def __init__(
        self,
        batch: SceneAgentBatch,
        col_radius: float = 1.0,
        prediction_horizon: int = 12,
        scene_limits: float = 20.0,
        scene_scale: int = 20,
        support_tolearance: float = 10.0,
    ) -> None:
        self.batch = batch
        self.data_idx = torch.unique(batch.data_idx)
        self.num_scenes = self.data_idx.numel()

        self.col_radius = col_radius

        self.model: Optional[TrajectoryPredictor] = None
        self.prediction_horizon = prediction_horizon

        self.scene_limits = scene_limits
        self.scale = scene_scale
        self.support_tol = support_tolearance

        self.logger = hm.Logger(log_file="logfile.txt", enabled=True)
        self.processor = hm.HalfplaneIntersectionProcessor(
            scene_size=int(2*scene_limits*scene_scale),
            tolerance=support_tolearance
        )
        self.processor.set_logger(self.logger)

    def _find_latest_checkpoint(self, model_dir: Path, epoch: int) -> Path:
        """
        Starting from `epoch`, search downward for the first existing
        'model_registrar-{epoch}.pt' file.
        """
        for e in range(epoch, 0, -1):
            candidate = model_dir / f"model_registrar-{e}.pt"
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"No checkpoint found in {model_dir} up to epoch {epoch}"
        )

    def load_model(
        self,
        model_dir: str,
        epoch: int,
        hyperparams: Dict[str, Any],
    ) -> None:
        """
        Loads the latest checkpoint (≤ epoch) into a TrajectoryPredictor.
        
        Args:
            model_dir: Directory containing 'model_registrar-<N>.pt' files.
            epoch:  Maximum epoch number to consider.
            hyperparams: Dict
        """
        model_dir_path = Path(model_dir)
        if not model_dir_path.is_dir():
            raise NotADirectoryError(f"{model_dir} is not a valid directory")

        # Find checkpoint file
        ckpt_path = self._find_latest_checkpoint(model_dir_path, epoch)

        # Initialize registrar & model
        device = hyperparams.get("device", torch.device("cpu"))
        registrar = ModelRegistrar(str(model_dir_path), device)
        predictor = TrajectoryPredictor(
            model_registrar=registrar,
            hyperparams=hyperparams,
            log_writer=None,
            device=device,
        )
        predictor.set_environment()

        # Load weights
        checkpoint = torch.load(ckpt_path, map_location=device)
        predictor.load_state_dict(checkpoint["model_state_dict"], strict=False)

        self.model = predictor

    def _get_true_future(self) -> np.ndarray:
        batch = self.batch

        coords = batch.agent_fut[:, :self.prediction_horizon, :2].cpu().numpy() # [M, T, 2]
        transforms = batch.world_from_agent_tf # [M, 3, 3]

        transformed = transform_coords_np(coords, transforms) # [M, T, 2]

        return transformed

    def _get_robot_positions(
            self,
            agent_pos: np.ndarray,  # [M, T, 2]
            samples_per_scene=1
    ) -> np.ndarray:
        agent_pos = agent_pos.reshape(-1, 2)
        valid = ~np.isnan(agent_pos).any(axis=1)
        agent_pos = agent_pos[valid]

        radius = self.col_radius + 0.5

        robot_positions = np.zeros((samples_per_scene, 2), dtype=agent_pos.dtype)
        for idx in range(samples_per_scene):
            while True: # Generate a random position for the robot
                # Ensure the robot's position does not overlap with any occupied positions
                robot_pos = np.array([
                    random.uniform(-self.scene_limits-2.0, self.scene_limits-2.0),
                    random.uniform(-self.scene_limits-2.0, self.scene_limits-2.0)
                ], dtype=agent_pos.dtype)

                in_collision = is_within_distance(agent_pos, robot_pos, radius)
                if not np.any(in_collision):
                    robot_positions[idx, :] = robot_pos
                    break
        return robot_positions

    def _get_support_estimate(
            self,
            agent_pos: np.ndarray,      # [M, T, 2]
            robot_positions: np.ndarray # [samples, 2]
        ) -> np.ndarray:
        agent_pos = np.transpose(agent_pos, (1, 0, 2))  # [T, M, 2]

        T = agent_pos.shape[0]
        num_samples = robot_positions.shape[0]
        support_estimates = np.zeros((num_samples, T), dtype=int)

        for idx, robot_pos in enumerate(robot_positions):
            coeffs = get_constraint_coeffs(robot_pos, agent_pos, self.col_radius)   # [T, M, 3]
            support_estimates[idx, :] = self.processor.get_support(coeffs)          # [T,]

        return support_estimates.max(axis=1)    # [num_samples,]

    def scene_supports(self, samples_per_scene=5, risk=0.05, confidence=0.01) -> np.ndarray:
        '''Get support for all ego agent samples in a given scene'''
        batch = self.batch

        agent_pos = self._get_true_future()
        robot_positions = self._get_robot_positions(agent_pos, samples_per_scene)   # [num_samples, 2]

        support_estimates = self._get_support_estimate(agent_pos, robot_positions)  # [num_samples,]

        #TODO: Currently used for pedestrian datasets only!
        agent_type = AgentType.PEDESTRIAN
        names = batch.agent_name
        data_ids = batch.data_idx
        timestamps = batch.scene_ts

        keys = [
            f"{str(agent_type)}/agent_{name}/idx_{id.item()}/ts_{ts.item()}"
            for name, id, ts in zip(names, data_ids, timestamps)
        ]

        for idx, support in enumerate(support_estimates):
            robot_pos = robot_positions[idx]

            calc_support = float('inf')
            est_support = support-1
            while calc_support>est_support:
                est_support+=1

                assert est_support<=5, "Estimated support exceeded 5" # TODO

                S = int(find_sample_size(est_support, risk, confidence))
                predictions_dict: Dict[str, np.ndarray] = self.model.predict(
                    batch=batch,
                    num_samples=S,
                    prediction_horizon=self.prediction_horizon,
                    full_dist=True,
                    output_dists=False
                )

                per_agent_pred = [predictions_dict[k] for k in keys]
                pred_array = np.stack(per_agent_pred, axis=0)  # (M, S, T, 2)

                M, S, T, D = pred_array.shape
                pred_array_flat = pred_array.reshape(M, S*T, D)

                transforms = batch.world_from_agent_tf # [M, 3, 3]
                transformed = transform_coords_np(pred_array_flat, transforms) # [M, S*T, 2]

                tmp = transformed.reshape(M, S, T, D)
                per_step_pred = np.transpose(tmp, (2, 0, 1, 3)) # (T, M, S, 2)

                N = M * S
                flat_preds = per_step_pred.reshape(T, N, 2)

                pred_coeffs = get_constraint_coeffs(robot_pos, flat_preds, self.col_radius) # [T, M, 3]
                supports = self.processor.get_support(pred_coeffs)    # [T,]

                calc_support = max(supports)

            support_estimates[idx] = est_support

        return support_estimates

