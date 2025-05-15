import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model_utils import ModeKeys


#### Helper Functions ####
def unpack_rnn_state(state_tuple):
    """
    Convert output of bi-directional LSTMs to batch first 
    and squeeze along feature dimension
    """
    # PyTorch returned LSTM states have 3 dims:
    # (num_layers * num_directions, batch, hidden_size)

    state = torch.cat(state_tuple, dim=0).permute(1, 0, 2)
    # Now state is (batch, 2 * num_layers * num_directions, hidden_size)

    state_size = state.size()
    return torch.reshape(state, (-1, state_size[1] * state_size[2]))

def roll_by_gather(mat: torch.Tensor, dim: int, shifts: torch.LongTensor):
    """Shift up rows of arrays by specified amount"""
    # assumes 3D array
    batch, ts, dim = mat.shape

    arange1 = (
        torch.arange(ts, device=shifts.device)
        .unsqueeze(0)
        .unsqueeze(-1)
        .expand(batch, -1, dim)
    )
    # print(arange1)
    arange2 = (arange1 - shifts[:, None, None]) % ts
    # print(arange2)
    return torch.gather(mat, 1, arange2)

def get_agent_neigh_joint_state(
    node_history_st: torch.Tensor,
    node_history_len: torch.Tensor,
    neigh_hist: torch.Tensor,
    neigh_hist_len: torch.Tensor,
    neigh_types: torch.Tensor,
):
    """
    Creates node-neighbor history pairs
    """
    # pad to have equal sequence lengths
    if neigh_hist.shape[2] < node_history_st.shape[1]:
        neigh_hist = F.pad(
            neigh_hist,
            pad=(0, 0, 0, node_history_st.shape[1] - neigh_hist.shape[2]),
            value=np.nan,
        )
    elif neigh_hist.shape[2] > node_history_st.shape[1]:
        node_history_st = F.pad(
            node_history_st,
            pad=(0, 0, 0, neigh_hist.shape[2] - node_history_st.shape[1]),
            value=np.nan,
        )
    # repeat node history length, num. of neighbor times
    node_hist_lens_for_cat = node_history_len.unsqueeze(1).expand(
        (-1, neigh_hist.shape[1])
    )
    # find minimum history for each node-neighbor pair
    joint_history_len = torch.minimum(
        neigh_hist_len, node_hist_lens_for_cat
    ).flatten()
    has_data: torch.Tensor = joint_history_len > 0
    # repeat node history, num. of neighbor times and keep those with minimum history > 0
    node_hist_for_cat = node_history_st.repeat_interleave(
        neigh_hist.shape[1], dim=0, output_size=has_data.shape[0]
    )[has_data]
    # squeeze neigbor history along num. of neigbors dim.
    neigh_hist_for_cat = neigh_hist.reshape(-1, *neigh_hist.shape[2:])[has_data]
    # history lenght and neighbor types for joint node-neigh pairs under consideration
    joint_history_len = joint_history_len[has_data]
    joint_neigh_types = neigh_types.flatten()[has_data]

    # calculate shift in timesteps
    # e.g. if node history length > neigh. history length,
    # shift node history up to only consider common timesteps
    node_shifts = joint_history_len - node_hist_lens_for_cat.flatten()[has_data]
    neigh_shifts = joint_history_len - neigh_hist_len.flatten()[has_data]
    # execute the shifts
    node_hist_for_cat = roll_by_gather(
        node_hist_for_cat, dim=1, shifts=node_shifts.to(node_hist_for_cat.device)
    )
    neigh_hist_for_cat = roll_by_gather(
        neigh_hist_for_cat, dim=1, shifts=neigh_shifts.to(neigh_hist_for_cat.device)
    )
    # concatenate node and neigh. states
    joint_history = torch.cat([neigh_hist_for_cat, node_hist_for_cat], dim=-1)

    return joint_history, joint_history_len, joint_neigh_types

############################

############################
#   Node History Encoder   #
############################
class NodeHistoryEncoder(nn.Module):
    """
    Node History Encoder using LSTM
    """
    def __init__(self, state_length, hidden_dim):
        super(NodeHistoryEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=state_length,
            hidden_size=hidden_dim,
            batch_first=True
        )

    def forward(self, hp, mode, node_hist, node_hist_len) -> torch.Tensor:
        """
        Encodes the nodes history.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param node_hist: Historic and current state of the node. [bs, mhl, state]
        :param node_hist_len: Number of timesteps for which data is available [bs]
        :return: Encoded node history tensor. [bs, enc_rnn_dim]
        """
        packed_input = nn.utils.rnn.pack_padded_sequence(
            node_hist, node_hist_len, batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=node_hist.shape[1]
        )
        output = F.dropout(
            output,
            p=1.0 - hp["rnn_kwargs"]["dropout_keep_prob"],
            training=(mode == ModeKeys.TRAIN),
        )  # [bs, max_time, enc_rnn_dim]
        return output[torch.arange(output.shape[0]), node_hist_len - 1]

############################
#   Node Future Encoder   #
############################
class NodeFutureEncoder(nn.Module):
    """
    Node Future Encoder using Bi-directional LSTM
    """
    def __init__(self, state_length, pred_state_length, hidden_dim):
        super(NodeFutureEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=pred_state_length,
            hidden_size=hidden_dim,
            bidirectional=True,
            batch_first=True
        )
        self.initial_h_model = nn.Linear(state_length, hidden_dim)
        self.initial_c_model = nn.Linear(state_length, hidden_dim)

    def forward(self, hp, mode, node_present, node_future, future_lens) -> torch.Tensor:
        """
        Encodes the node future (during training) using a bi-directional LSTM

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param node_present: Current state of the node. [bs, state]
        :param node_future: Future states of the node. [bs, ph, state]
        :return: Encoded future.
        """
        # Here we're initializing the forward hidden states,
        # but zeroing the backward ones.
        initial_h = self.initial_h_model(node_present)
        initial_h = torch.stack( #TODO: verify node_present.device
            [initial_h, torch.zeros_like(initial_h, device=node_present.device)], dim=0
        )
        initial_c = self.initial_c_model(node_present)
        initial_c = torch.stack( #TODO: verify node_present.device
            [initial_c, torch.zeros_like(initial_c, device=node_present.device)], dim=0
        )

        initial_state = (initial_h, initial_c)

        node_future_packed = nn.utils.rnn.pack_padded_sequence(
            node_future, future_lens, batch_first=True, enforce_sorted=False
        )

        _, state = self.lstm(node_future_packed, initial_state)
        state = unpack_rnn_state(state)
        state = F.dropout(
            state,
            p=1.0 - hp["rnn_kwargs"]["dropout_keep_prob"],
            training=(mode == ModeKeys.TRAIN),
        )

        return state

####################
#   Edge Encoder   #
####################
#TODO: implement edge encoder modules
class EdgeEncoder(nn.Module):
    """
    Edge Encoder using LSTM
    """
    def __init__(self, state_length, neighbor_state_length, rnn_hidden_dim):
        super(EdgeEncoder, self).__init__()
        edge_encoder_input_size = state_length + neighbor_state_length
        self.lstm = nn.LSTM(
            edge_encoder_input_size,
            rnn_hidden_dim,
            batch_first=True
        )

    def forward(
        self,
        mode,
        hp,
        joint_history,
        joint_history_len,
    ) -> torch.Tensor:
        """Encode all agent-neighbor joint histories"""
        #TODO: input should be only for a given neigh. type
        packed_input = nn.utils.rnn.pack_padded_sequence(
            joint_history, joint_history_len, batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed_input)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=joint_history.shape[1]
        )
        outputs = F.dropout(
            outputs,
            p=1.0 - hp["rnn_kwargs"]["dropout_keep_prob"],
            training=(mode == ModeKeys.TRAIN),
        )  # [bs, max_time, enc_rnn_dim]

        return outputs[torch.arange(outputs.shape[0]), joint_history_len - 1]
    
################################
#   Edge Interaction Encoder   #
################################
#TODO: implement edge encoder modules
class EdgeInteractionEncoder(nn.Module):
    """
    Edge Interaction Encoder using Transformer with Edge Encoding
    """
    def __init__(self, edge_types, hyperparams):
        super(EdgeInteractionEncoder, self).__init__()

    def forward(self, mode, node_hist, node_hist_len) -> torch.Tensor:
        return None

###################
#   Map Encoder   #
###################
class CNNMapEncoder(nn.Module):
    """
    Convolutional Neural Network (CNN) based Map Encoder
    """
    def __init__(
        self, map_channels, hidden_channels, output_size, masks, strides, patch_size
    ):
        super(CNNMapEncoder, self).__init__()
        self.convs = nn.ModuleList()
        input_size = (map_channels, patch_size, patch_size)
        x_dummy = torch.ones(input_size).unsqueeze(0) * torch.tensor(float("nan"))

        for i, _ in enumerate(hidden_channels):
            self.convs.append(
                nn.Conv2d(
                    map_channels if i == 0 else hidden_channels[i - 1],
                    hidden_channels[i],
                    masks[i],
                    stride=strides[i],
                )
            )
            x_dummy = self.convs[i](x_dummy)

        self.fc = nn.Linear(x_dummy.numel(), output_size)

    def forward(self, hp, nt, mode, x):
        """Forward pass through the CNN Map Encoder"""
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.2)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        x = F.dropout(
            x,
            p=hp["map_encoder"][nt]["dropout"],
            training=(mode == ModeKeys.TRAIN),
        )
        return x

###################
# p_z_x OR q_z_xy #
###################
class LatentDistEncoder(nn.Module):
    """
    Latent Distribution Encoder of the CVAE
    """
    def __init__(self, p_input_dim, p_hidden_dim, latent_dim):
        super(LatentDistEncoder, self).__init__()
        self.p_hidden_dim = p_hidden_dim
        if self.p_hidden_dim != 0:
            self.p_dist = nn.Linear(p_input_dim, p_hidden_dim)
            h_input_dim = self.p_hidden_dim
        else:
            h_input_dim = p_input_dim
        self.h_dist = nn.Linear(h_input_dim, latent_dim)

    def forward(self, hp, mode, enc):
        """Forward pass (encoding -> latent value)"""
        if self.p_hidden_dim != 0:
            h = F.dropout(
                F.relu(self.p_dist(enc)),
                p=1.0 - hp["MLP_dropout_keep_prob"],
                training=(mode == ModeKeys.TRAIN),
            )
        else:
            h = enc
        latent = self.h_dist(h)

        return latent

###################
#   Decoder GRU   #
###################
class DecoderGRU(nn.Module):
    """
    Gated Recurrent Unit Decoder
    """
    def __init__(
            self,
            state_length,
            pred_state_length,
            z_size,
            x_size,
            rnn_hidden_dim,
            decoder_input_dim
    ):
        super(DecoderGRU, self).__init__()
        self.state_action = nn.Linear(state_length, pred_state_length)
        self.rnn = nn.GRU(z_size + x_size, rnn_hidden_dim, batch_first=True)
        self.rnn_cell = nn.GRUCell(decoder_input_dim, rnn_hidden_dim)
        self.initial_h = nn.Linear(z_size + x_size, rnn_hidden_dim)

    def forward(self):
        """Forward pass of the GRU decoder unit"""

        return
    
###################
#   Decoder GMM   #
###################
class DecoderGMM(nn.Module):
    """
    Gausian Mixture Model Decoder
    """
    def __init__(
            self,
            pred_state_length,
            gmm_components
    ):
        super(DecoderGMM, self).__init__()
        gmm_mus_dim = pred_state_length
        gmm_log_sigmas_dim = pred_state_length
        gmm_corrs_dim = 1
        gmm_dims = gmm_mus_dim + gmm_log_sigmas_dim + gmm_corrs_dim

        self.proj_to_log_pis = nn.Linear(_, gmm_components) #TODO: one_layer_equivalent?
        self.proj_to_mus = nn.Linear(_, gmm_components*pred_state_length) #TODO: ""
        self.proj_to_log_sigmas = nn.Linear(_, gmm_components*pred_state_length) #TODO: ""
        self.proj_to_corrs = nn.Linear(_, gmm_components) #TODO: ""

    def forward(self):
        """Forward pass of the GMM decoder module"""
        
        return
