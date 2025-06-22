"""
Custom data
"""
from typing import Tuple, List, Callable, Optional

import numpy as np

from trajdata import SceneBatch
from trajdata.data_structures.batch_element import SceneBatchElement
from trajdata.data_structures.agent import AgentMetadata
from trajdata.data_structures.state import StateArray
from trajdata.utils.state_utils import transform_state_np_2d

def custom_world_from_agent_tf(
    batch_elem: SceneBatchElement,
) -> np.ndarray:
    '''Transformation matrix for conversion from agent frame to world frame'''
    world_agent_hist: List[np.ndarray] = batch_elem.agent_histories

    agent_pos_list = [hist[-1,:2] for hist in world_agent_hist]
    agent_sc_list = [hist[-1,-2:] for hist in world_agent_hist]

    tf_list = []
    for pos, sc in zip(agent_pos_list, agent_sc_list):
        cos_agent = sc[-1]
        sin_agent = sc[-2]
        tf: np.ndarray = np.array(
            [
                [cos_agent, -sin_agent, pos[0]],
                [sin_agent, cos_agent, pos[1]],
                [0.0, 0.0, 1.0],
            ]
        )
        tf_list.append(tf)

    return np.array(tf_list)

def get_neigh_idxs(
    batch_elem: SceneBatchElement,
    interaction_radius: float,
) -> np.ndarray:
    """Provides adjacency matrix"""
    agents: List[AgentMetadata] = batch_elem.agents
    curr_states = []
    for agent in agents:
        raw_state: StateArray = batch_elem.cache.get_raw_state(
            agent.name, batch_elem.scene_ts
        )
        state = np.asarray(raw_state)
        curr_states.append(state)

    is_neigh = []
    for state in curr_states:
        distances = [
            np.linalg.norm(state[:2] - agent_st[:2])
            for agent_st in curr_states
        ]
        is_neigh.append([dist <= interaction_radius for dist in distances])
    is_neigh_mat = np.stack(is_neigh, axis=0)
    np.fill_diagonal(is_neigh_mat, False)

    return is_neigh_mat

def custom_agent_neigh_hist(
    batch_elem: SceneBatchElement,
    history_sec: Tuple[Optional[float], Optional[float]],
) -> np.ndarray:
    """
    Provide neighbor history of each agent in scene
    in respective agent-centric frames
    """
    assert batch_elem.standardize_data is False, \
        "Per-agent history requires a non-standarized dataset (set standardize_data=False)"

    dt = batch_elem.dt
    max_hist_len: int = round((history_sec[1]/dt)) + 1
    world_agent_hist: List[np.ndarray] = batch_elem.agent_histories
    state_dim = world_agent_hist[0].shape[-1]
    num_agents = batch_elem.num_agents

    neigh_hists: List[List[np.ndarray]] = []
    for idx in range(num_agents):
        centered_world_from_agent_tf = batch_elem.extras["world_from_agent_tf"][idx]
        centered_agent_from_world_tf: np.ndarray = np.linalg.inv(
            centered_world_from_agent_tf
        )

        row_hists: List[Optional[np.ndarray]] = []
        for jdx, agent_hist in enumerate(world_agent_hist):
            # append if neighbor or self
            if batch_elem.extras["is_neigh"][idx, jdx] or jdx==idx:
                hist_st = transform_state_np_2d(agent_hist, centered_agent_from_world_tf)
                row_hists.append(hist_st)
            # else append None
            else:
                row_hists.append(None)
        neigh_hists.append(row_hists)

    output = np.full(
        (num_agents, num_agents, max_hist_len, state_dim),
        np.nan,
        dtype=np.float32
    )
    # pad arrays to match maximum history length (8 steps),
    # and max possible neighbors (num of agents)
    for i in range(num_agents):
        for k, hist_ij in enumerate(neigh_hists[i]):
            if hist_ij is None:
                continue
            len_j = hist_ij.shape[0]
            output[i, k, :len_j, :] = hist_ij

    return output.astype(np.float32)

def custom_agent_fut(
    batch_elem: SceneBatchElement,
    future_sec: Tuple[Optional[float], Optional[float]],
) -> np.ndarray:
    dt = batch_elem.dt
    max_fut_len: int = round((future_sec[1]/dt))

    world_agent_fut: List[np.ndarray] = batch_elem.agent_futures
    state_dim = world_agent_fut[0].shape[-1]

    agent_fut_list = []
    for idx, agent_fut in enumerate(world_agent_fut):
        centered_world_from_agent_tf = batch_elem.extras["world_from_agent_tf"][idx]
        centered_agent_from_world_tf: np.ndarray = np.linalg.inv(
            centered_world_from_agent_tf
        )

        fut_st = transform_state_np_2d(agent_fut, centered_agent_from_world_tf)

        t_i = fut_st.shape[0]
        if t_i<max_fut_len:
            padding = np.full((max_fut_len-t_i, state_dim), np.nan, dtype=fut_st.dtype)
            fut_st_padded = np.concatenate([fut_st, padding], axis=0)
        else:
            fut_st_padded = fut_st[:max_fut_len]
        agent_fut_list.append(fut_st_padded)

    return np.stack(agent_fut_list, axis=0).astype(np.float32)

def custom_collate_fn(
    batch_elems: List[SceneBatchElement],
    history_sec: Tuple[Optional[float], Optional[float]],
    future_sec: Tuple[Optional[float], Optional[float]],
    base_collate: Callable[[List[SceneBatchElement]], SceneBatch],
) -> SceneBatch:
    """
    1) Pads each extras data 
    2) Calls base_collate(...) (i.e. scene_collate_fn) on the padded batch.
    """
    max_agent_num: int = max(elem.num_agents for elem in batch_elems) # M
    dt = batch_elems[0].dt
    max_hist_len: int = round((history_sec[1]/dt)) + 1
    max_fut_len: int = round((future_sec[1]/dt))
    state_dim = batch_elems[0].agent_histories[0].shape[-1]

    for elem in batch_elems:
        n_i = elem.num_agents
        tf = elem.extras["world_from_agent_tf"]
        is_neigh = elem.extras["is_neigh"]
        agent_neigh_hist = elem.extras["agent_neigh_hist_st"]
        agent_fut = elem.extras["agent_fut_st"]

        if n_i < max_agent_num:
            # Pad "world_from_agent_tf": shape (n_i, 3, 3) -> (max_agents_in_batch, 3, 3)
            padded_tf = np.full(
                (max_agent_num, 3, 3),
                np.nan,
                dtype=tf.dtype
            )
            padded_tf[:n_i] = tf
            # Pad "is_neigh": shape (n_i, n_i) -> (max_agents_in_batch, max_agents_in_batch)
            padded_is_neigh = np.zeros((max_agent_num, max_agent_num), dtype=is_neigh.dtype)
            padded_is_neigh[:n_i, :n_i] = is_neigh
            elem.extras["is_neigh"] = padded_is_neigh
            # Pad "neigh_hist": shape (n_i, n_i, max_hist_len, 8) -> (M, M, max_hist_len, 8)
            padded_agent_neigh_hist = np.full(
                (max_agent_num, max_agent_num, max_hist_len, state_dim),
                np.nan,
                dtype=agent_neigh_hist.dtype
            )
            padded_agent_neigh_hist[:n_i, :n_i, :, :] = agent_neigh_hist
            # Pad "agent_fut_st": shape (n_i, max_fut_len, 8) -> (M, max_fut_len, 8)
            padded_agent_fut = np.full(
                (max_agent_num, max_fut_len, state_dim),
                np.nan,
                dtype=agent_neigh_hist.dtype
            )
            padded_agent_fut[:n_i] = agent_fut
        else:
            padded_tf = tf[:max_agent_num]
            elem.extras["is_neigh"] = is_neigh[:max_agent_num, :max_agent_num]
            padded_agent_neigh_hist = agent_neigh_hist
            padded_agent_fut = agent_fut[:max_agent_num]

        elem.extras["world_from_agent_tf"] = padded_tf
        elem.extras["agent_neigh_hist_st"] = padded_agent_neigh_hist
        elem.extras["agent_fut_st"] = padded_agent_fut

    return base_collate(batch_elems)
