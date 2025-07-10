"""The SceneBatchAgent class"""
from typing import List

import numpy as np
import torch

from trajdata import SceneBatch
from trajdata.data_structures.agent import AgentType

class SceneAgentBatch:
    """
    Extends SceneBatch by exposing per‐agent views:
      - agent_hist (stacked per‐agent histories)
      - agent_hist_len (1D lengths)
      - num_neigh (1D neighbor counts)
      - neigh_hist (per‐agent neighbor histories)
      - neigh_hist_len (per‐agent neighbor history lengths)
      - neigh_types (per‐agent neighbor types)
    All other SceneBatch fields remain available unchanged.
    """
    # Notations:
    # S: number of scenes
    # B: batch size (total agents combined in all scenes)
    # M: max number of agents in a scene in the batch
    # H: max history length; F: max fuutre length
    # N: state dimension
    def __init__(
        self,
        batch: SceneBatch,
        filter_mask=None
    ):
        self.batch = batch

        self.num_agents = batch.num_agents
        self.total_agents = int(self.num_agents.sum().item())   # B

        self.is_neigh = batch.extras['is_neigh']   # shape (S, M, M)

        if filter_mask is None:
            filter_mask = torch.ones((self.total_agents,), dtype=torch.bool)

        self.filter_mask_dict = {}
        self.filter_mask_dict["cpu"] = filter_mask.to("cpu")
        self.filter_mask_dict[str(self.batch.agent_hist.device)] = filter_mask.to(
            self.batch.agent_hist.device
        )

    def to(self, device) -> None:
        '''Duplicate SceneBatch's to() function'''
        self.batch.to(device)

    def _filter(self, tensor: torch.Tensor):
        return tensor[self.filter_mask_dict[str(tensor.device)]]

    def _filter_tensor_or_list(self, tensor_or_list):
        if isinstance(tensor_or_list, torch.Tensor):
            return self._filter(tensor_or_list)
        else:
            return type(tensor_or_list)(
                [
                    el
                    for idx, el in enumerate(tensor_or_list)
                    if self.filter_mask_dict["cpu"][idx]
                ]
            )

    @property
    def dt(self) -> torch.Tensor:
        '''scene dt'''
        dt_list: List[torch.Tensor] = []
        for scene_idx, n_i in enumerate(self.num_agents):
            n_i = n_i.item()
            dt = self.batch.dt[scene_idx]
            for _ in range(n_i):
                dt_list.append(dt)
        dt = torch.stack(dt_list, dim=0)
        return self._filter(dt)

    @property
    def data_idx(self) -> torch.Tensor:
        '''data idx'''
        data_idx_list: List[torch.Tensor] = []
        for scene_idx, n_i in enumerate(self.num_agents):
            n_i = n_i.item()
            data_id = self.batch.data_idx[scene_idx]
            for _ in range(n_i):
                data_idx_list.append(data_id)
        data_idx = torch.stack(data_idx_list, dim=0)
        return self._filter(data_idx)

    @property
    def scene_ts(self) -> torch.Tensor:
        '''scene timestep'''
        scene_ts_list: List[torch.Tensor] = []
        for scene_idx, n_i in enumerate(self.num_agents):
            n_i = n_i.item()
            ts = self.batch.scene_ts[scene_idx]
            for _ in range(n_i):
                scene_ts_list.append(ts)
        scene_ts = torch.stack(scene_ts_list, dim=0)
        return self._filter(scene_ts)

    @property
    def scene_ids(self) -> List[str]:
        '''Scene ID'''
        scene_ids: List[str] = []
        for scene_idx, n_i in enumerate(self.num_agents):
            n_i = n_i.item()
            scene_id = self.batch.scene_ids[scene_idx]
            for _ in range(n_i):
                scene_ids.append(scene_id)

        return self._filter_tensor_or_list(scene_ids)

    @property
    def agent_hist(self) -> torch.Tensor:
        """
        Stack each valid agent history
        into a single (total_agents, hist_len, state_dim) tensor.
        """
        agent_neigh_hist = self.batch.extras['agent_neigh_hist_st'] # (S, M, M, H, N)
        agent_hist_list: List[torch.Tensor] = []
        for scene_idx, n_i in enumerate(self.num_agents):
            n_i = n_i.item()
            for agent_idx in range(n_i):
                agent_hist_list.append(agent_neigh_hist[scene_idx, agent_idx, agent_idx])   # (H, N)
        agent_hist = torch.stack(agent_hist_list, dim=0)    # (B, H, N)
        return self._filter(agent_hist)

    @property
    def agent_hist_len(self) -> torch.Tensor:
        """
        1D tensor of all per‐agent history lengths (skipping zeros).
        """
        # self.agent_hist_len exists as a (B, M) tensor. We flatten and drop zeros.
        grouped_hist_len = self.batch.agent_hist_len  # provided by SceneBatch
        flat = grouped_hist_len[grouped_hist_len != 0]
        return self._filter(flat)

    @property
    def agent_fut(self) -> torch.Tensor:
        """
        Stack each valid agent future
        into a single (total_agents, fut_len, state_dim) tensor.
        """
        agent_fut = self.batch.extras["agent_fut_st"]   # (S, M, F, N)
        agent_fut_list: List[torch.Tensor] = []
        for scene_idx, n_i in enumerate(self.num_agents):
            n_i = n_i.item()
            for agent_idx in range(n_i):
                agent_fut_list.append(agent_fut[scene_idx, agent_idx])   # (F, N)
        agent_fut = torch.stack(agent_fut_list, dim=0)    # (B, F, N)
        return self._filter(agent_fut)

    @property
    def agent_fut_len(self) -> torch.Tensor:
        """
        1D tensor of all per‐agent history lengths (skipping zeros).
        """
        # self.agent_hist_len exists as a (B, M) tensor. We flatten and drop zeros.
        grouped_hist_len = self.batch.agent_hist_len  # provided by SceneBatch
        grouped_fut_len = self.batch.agent_fut_len
        flat = grouped_fut_len[grouped_hist_len != 0]
        return self._filter(flat)

    @property
    def agent_name(self) -> List[str]:
        '''Agent Names'''
        agent_name: List[str] = []
        for scene_idx, n_i in enumerate(self.num_agents):
            n_i = n_i.item()
            for agent_idx in range(n_i):
                agent_name.append(self.batch.agent_names[scene_idx][agent_idx])

        return self._filter_tensor_or_list(agent_name)

    @property
    def num_neigh(self) -> torch.Tensor:
        """
        1D tensor of neighbor counts for each agent in the batch.
        """
        counts_list: List[torch.Tensor] = []

        for scene_idx, n_i in enumerate(self.num_agents):
            # take the n_i×n_i submatrix for this scene
            n_i = n_i.item()
            submat = self.is_neigh[scene_idx, :n_i, :n_i]         # (n_i, n_i)
            counts_per_agent = submat.sum(dim=1).to(torch.long)  # (n_i,)
            counts_list.append(counts_per_agent)
        num_neigh = torch.cat(counts_list, dim=0)  # (B,)
        return self._filter(num_neigh)

    def get_neigh_idxs(self) -> List[torch.Tensor]:
        neigh_idxs_list = []
        for scene_idx, n_i in enumerate(self.num_agents):
            n_i = n_i.item()
            for agent_idx in range(n_i):
                mask = self.is_neigh[scene_idx, agent_idx, :n_i]   # (n_i,)
                neigh_idxs_list.append(torch.nonzero(mask, as_tuple=False).view(-1))

        return neigh_idxs_list

    @property
    def neigh_hist(self) -> torch.Tensor:
        """
        For each agent, grab its row in extras['neigh_hist'], truncated to max_num_neigh.
        Returns a (total_agents, max_num_neigh, hist_len, state_dim_nh) tensor.
        """
        max_num_neigh: int = int(self.num_neigh.max().item())
        agent_neigh_hist: torch.Tensor = self.batch.extras['agent_neigh_hist_st']  # (S, M, M, H, N)

        _, _, _, H, N = agent_neigh_hist.shape
        device = agent_neigh_hist.device
        dtype = agent_neigh_hist.dtype

        neigh_idxs_list = self.get_neigh_idxs()

        count = 0
        neigh_hist_list: List[torch.Tensor] = []
        for scene_idx, n_i in enumerate(self.num_agents):
            n_i = n_i.item()
            #TODO
            for agent_idx in range(n_i):
                neigh_idxs = neigh_idxs_list[count]
                neigh_hist_list.append(
                    agent_neigh_hist[scene_idx, agent_idx, neigh_idxs]
                )   # (#neighbors, H, N)
                count+=1

        padded_neigh_hist = torch.full(
            (self.total_agents, max_num_neigh, H, N),
            np.nan,
            dtype=dtype,
            device=device
        )

        for idx, neigh_hist in enumerate(neigh_hist_list):
            k = neigh_hist.size(0)
            if k > 0:
                padded_neigh_hist[idx, :k] = neigh_hist.to(device)

        return self._filter(padded_neigh_hist)

    @property
    def neigh_hist_len(self) -> torch.Tensor:
        """
        Returns a (total_agents, max_num_neigh) tensor where each row contains
        the history‐lengths of that agent’s neighbors, zero‐padded.
        """
        agent_hist_len: torch.Tensor = self.batch.agent_hist_len  # (S, M) from SceneBatch
        max_num_neigh = int(self.num_neigh.max().item())

        device = agent_hist_len.device
        dtype = agent_hist_len.dtype

        neigh_idxs_list = self.get_neigh_idxs()

        # Build a list of per‐agent neighbor‐history lengths
        count = 0
        neigh_len_list: List[torch.Tensor] = []
        for scene_idx, n_i in enumerate(self.num_agents):
            n_i = n_i.item()
            for _ in range(n_i):
                neigh_idxs = neigh_idxs_list[count]
                hist_lens = agent_hist_len[scene_idx, neigh_idxs]        # (#neighbors,)
                neigh_len_list.append(hist_lens)
                count+=1

        # Allocate output, fill with zeros
        output = torch.zeros((self.total_agents, max_num_neigh),
                             dtype=dtype, device=device)

        for idx, hist_lens in enumerate(neigh_len_list):
            k = hist_lens.size(0)
            if k > 0:
                output[idx, :k] = hist_lens.to(device)

        return self._filter(output)

    @property
    def neigh_types(self) -> torch.Tensor:
        """
        Returns a (total_agents, max_num_neigh) tensor where each row lists
        the neighbor types of that agent, padded with -1.
        """
        agent_types: torch.Tensor = self.batch.agent_type   # (S, M)
        max_num_neigh = int(self.num_neigh.max().item())

        device = agent_types.device
        dtype = agent_types.dtype

        neigh_idxs_list = self.get_neigh_idxs()

        count = 0
        neigh_types_list: List[torch.Tensor] = []
        for scene_idx, n_i in enumerate(self.num_agents):
            n_i = n_i.item()
            for _ in range(n_i):
                neigh_idxs = neigh_idxs_list[count]
                types = agent_types[scene_idx, neigh_idxs]  # (#neighbors,)
                neigh_types_list.append(types)
                count+=1

        output = torch.full(
            (self.total_agents, max_num_neigh),
            -1,
            dtype=dtype,
            device=device
        )

        for idx, types in enumerate(neigh_types_list):
            k = types.size(0)
            if k > 0:
                output[idx, :k] = types.to(device)

        return self._filter(output)

    @property
    def world_from_agent_tf(self) -> np.ndarray:
        tf = self.batch.extras["world_from_agent_tf"]
        tf_list: List[np.ndarray] = []
        for scene_idx, n_i in enumerate(self.num_agents):
            n_i = n_i.item()
            for agent_idx in range(n_i):
                tf_list.append(tf[scene_idx, agent_idx])

        tf = np.stack(tf_list, axis=0)
        return tf

    @property
    def agent_type(self) -> torch.Tensor:
        agent_type_list: List[torch.Tensor] = []
        for scene_idx, n_i in enumerate(self.num_agents):
            n_i = n_i.item()
            for agent_idx in range(n_i):
                agent_type_list.append(self.batch.agent_type[scene_idx, agent_idx])

        return  torch.stack(agent_type_list, dim=0)

    def agent_types(self) -> List[AgentType]:
        unique_types: torch.Tensor = torch.unique(self.agent_type)
        return [
            AgentType(unique_type.item())
            for unique_type in unique_types
            if unique_type >= 0
        ]

    def for_agent_type(self, agent_type: AgentType):
        match_type = self.agent_type == agent_type
        return SceneAgentBatch(self.batch, match_type)
