import os
import json
from functools import partial
from collections import defaultdict
from typing import Tuple, List, Callable

import numpy as np
# import torch

from torch.utils import data
from trajdata import UnifiedDataset, SceneBatch
from trajdata.data_structures.batch_element import SceneBatchElement
from trajdata.data_structures.agent import AgentMetadata, AgentType
from trajdata.data_structures.state import StateArray
from trajdata.augmentation import NoiseHistories

def all_current_states(
    batch_elem: SceneBatchElement,
) -> np.ndarray:
    agents: List[AgentMetadata] = batch_elem.agents
    curr_pos = []
    for agent in agents:
        raw_state: StateArray = batch_elem.cache.get_raw_state(
            agent.name, batch_elem.scene_ts
        )
        state = np.asarray(raw_state)
        curr_pos.append(state)
    return np.stack(curr_pos, axis=0)

def get_neighs(
    batch_elem: SceneBatchElement,
    interaction_radius: float,
) -> np.ndarray:
    curr_states = batch_elem.extras["curr_states"]
    is_neigh = []
    for state in curr_states:
        distances = [
            np.linalg.norm(state[:2] - agent_st[:2])
            for agent_st in curr_states
        ]
        is_neigh.append([dist <= interaction_radius for dist in distances])
    return np.stack(is_neigh, axis=0)

def extras_collate_fn(
    batch_elems: List[SceneBatchElement],
    base_collate: Callable[[List[SceneBatchElement]], SceneBatch],
) -> SceneBatch:
    """
    1) Pads each extra (“curr_states”, “is_neigh”, “custom”) 
       to the same first dimension = max_agents_in_batch.
    2) Calls base_collate(...) (i.e. scene_collate_fn) on the padded batch.
    """
    max_agent_num: int = max(elem.num_agents for elem in batch_elems)

    for elem in batch_elems:
        # Pad curr_states: shape (n_i, state_dim) → (max_agents_in_batch, state_dim)
        curr_arr = elem.extras["curr_states"]
        n_i, state_dim = curr_arr.shape
        if n_i < max_agent_num:
            pad_block = np.zeros((max_agent_num - n_i, state_dim), dtype=curr_arr.dtype)
            elem.extras["curr_states"] = np.concatenate([curr_arr, pad_block], axis=0)
        else:
            elem.extras["curr_states"] = curr_arr[:max_agent_num]

        # Pad is_neigh: shape (n_i, n_i) → (max_agents_in_batch, max_agents_in_batch)
        mat = elem.extras["is_neigh"]
        if n_i < max_agent_num:
            pad_mat = np.zeros((max_agent_num, max_agent_num), dtype=mat.dtype)
            pad_mat[:n_i, :n_i] = mat
            elem.extras["is_neigh"] = pad_mat
        else:
            elem.extras["is_neigh"] = mat[:max_agent_num, :max_agent_num]

    return base_collate(batch_elems)

def main():
    log_dir = 'data/trained_models/trajectory_prediction'
    model_dir = os.path.join(log_dir, "eth-28_May_2025_10_28_45")

    with open(os.path.join(model_dir, 'config.json'), 'r', encoding="utf-8") as config_json:
        hyperparams = json.load(config_json)
    # device
    hyperparams["device"] = "cpu"
    hyperparams["trajdata_cache_dir"] = "data/pedestrian_datasets/.unified_data_cache"

    desired_data=[
        "eupeds_eth-train",
    ]
    max_agent_num = 20
    data_dirs = {
        "eupeds_eth": "data/pedestrian_datasets/eth_ucy_peds",
    }

    attention_radius = defaultdict(
        lambda: 20.0
    )  # Default range is 20m unless otherwise specified.
    attention_radius[(AgentType.PEDESTRIAN, AgentType.PEDESTRIAN)] = 5.0

    input_noise = 0.0
    augmentations = list()
    if input_noise > 0.0:
        augmentations.append(NoiseHistories(stddev=input_noise))

    batch_size = 4

    dataset = UnifiedDataset(
        desired_data=desired_data,
        centric="scene",
        history_sec=(0.1, hyperparams["history_sec"]),
        future_sec=(0.1, hyperparams["prediction_sec"]),
        agent_interaction_distances=attention_radius,
        max_agent_num=max_agent_num,
        incl_robot_future=hyperparams["incl_robot_node"],
        incl_raster_map=hyperparams["map_encoding"],
        only_predict=[AgentType.PEDESTRIAN],
        no_types=[AgentType.UNKNOWN],
        augmentations=augmentations if len(augmentations) > 0 else None,
        standardize_data=False,
        num_workers=hyperparams["preprocess_workers"],
        cache_location=hyperparams["trajdata_cache_dir"],
        data_dirs=data_dirs,
        verbose=True,
        extras={
            "curr_states": all_current_states,
            "is_neigh": partial(get_neighs, interaction_radius=5.0),
        }
    )

    print(f"# Data Samples: {len(dataset)}")

    base_collate = dataset.get_collate_fn(pad_format="right")

    dataloader = data.DataLoader(
        dataset,
        # collate_fn=dataset.get_collate_fn(pad_format="right"),
        collate_fn=partial(extras_collate_fn, base_collate=base_collate),
        pin_memory=False if hyperparams["device"] == "cpu" else True,
        batch_size=batch_size,
        shuffle=True,
        num_workers=hyperparams["preprocess_workers"],
        sampler=None,
    )

    batch: SceneBatch = next(iter(dataloader))

    return batch

if __name__ == '__main__':
    batch = main()
    print(f"Num of agents in scenes: {batch.num_agents}")
    print(f"Shape of all current states array: {batch.extras['curr_states'].shape}")
    print(f"Shape of is_neigh array: {batch.extras['is_neigh'].shape}")
    # print(f"Shape of neigh hist array: {batch.extras['neigh_hist'].shape}")
