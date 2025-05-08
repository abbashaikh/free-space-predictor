import numpy as np
import torch
import torch.nn as nn

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
