'''Support Predictor'''
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
from trajdata import AgentBatch, AgentType

from fsp.modules import (
    PositionalBias,
    MultiheadAttention,
    TransformerEncoder
)
from data_generation import SceneAgentBatch


def init_params(module):

    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(
            data.cpu().normal_(mean=0.0, std=0.02).to(data.device)
        )

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


class SupportPredictor(nn.Module):
    def __init__(
        self,
        hyperparams: Dict,
        num_edges: int,
        edge_type: str,
        num_encoder_layers: int = 6,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        layerdrop: float = 0.0,
        apply_init: bool = False,
        activation_fn: str = "relu",
        learned_pos_embedding: bool = True,
        embed_scale: float = None,
        export: bool = False,
        traceable: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        sandwich_ln: bool = False,
        droppath_prob: float = 0.0,
        num_3d_bias_kernel: int = 128,
    ) -> None:
        super(SupportPredictor, self).__init__()
        self.layerdrop = layerdrop
        self.embedding_dim = embedding_dim
        self.apply_init = apply_init
        self.traceable = traceable

        # remove_head is set to true during fine-tuning
        self.load_softmax = not getattr(hyperparams, "remove_head", False)

        self.positional_bias = PositionalBias(
            num_heads=num_attention_heads,
            num_edges=num_edges,
            n_layers=num_encoder_layers,
            embed_dim=embedding_dim,
            num_kernel=num_3d_bias_kernel,
            no_share_rpe=False,
        )

        self.embed_scale = embed_scale

        droppath_probs = [
            x.item() for x in torch.linspace(0, droppath_prob, num_encoder_layers)
        ]

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_transformer_m_encoder_layer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout_module.p, # TODO
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    export=export,
                    q_noise=q_noise,
                    qn_block_size=qn_block_size,
                    sandwich_ln=sandwich_ln,
                    droppath_prob=droppath_probs[_],
                )
                for _ in range(num_encoder_layers)
            ]
        )

        # Apply initialization of model params after building the model
        if self.apply_init:
            self.apply(init_params)

    def build_transformer_m_encoder_layer(
        self,
        embedding_dim,
        ffn_embedding_dim,
        num_attention_heads,
        dropout,
        attention_dropout,
        activation_dropout,
        activation_fn,
        export,
        q_noise,
        qn_block_size,
        sandwich_ln,
        droppath_prob,
    ):
        return TransformerEncoder(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            activation_fn=activation_fn,
            export=export,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            sandwich_ln=sandwich_ln,
            droppath_prob=droppath_prob,
        )
    
    def forward(
        self,
        batch: SceneAgentBatch,
        last_state_only: bool = False,
    ):
        attn_bias, merged_edge_features, delta_pos = self.positional_bias(batch)

        # TODO
        if mask_3d is not None:
            merged_edge_features, delta_pos = merged_edge_features * mask_3d[:, None, None], delta_pos * mask_3d[:, None, None, None]
            attn_bias_3d = attn_bias_3d.masked_fill_(((attn_bias_3d != float('-inf')) * (1 - mask_3d[:, None, None, None])).bool(), 0.0)

        x = batch.enc #TODO
        x[:, 1:, :] = x[:, 1:, :] + merged_edge_features * 0.01

        if self.embed_scale is not None:
            x = x * self.embed_scale

        # B x T x C -> T x B x C
        x = x.transpose(0, 1) # TODO

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        for layer in self.layers:
            x, _ = layer(x, self_attn_padding_mask=padding_mask, self_attn_mask=attn_mask, self_attn_bias=attn_bias)
            if not last_state_only:
                inner_states.append(x)

        atom_output = None
        if delta_pos is not None:
            atom_output = self.atom_proc(x[1:, :, :], attn_bias[:, :, 1:, 1:], delta_pos)
            if mask_3d is not None:
                mask_3d_only = (mask == torch.tensor([0.0, 1.0]).to(mask)[None, :]).all(dim=-1)
                atom_output = atom_output * mask_3d_only[:, None, None]

        if last_state_only:
            inner_states = [x]

        if self.traceable:
            inner_states, atom_output = torch.stack(inner_states), atom_output
        else:
            inner_states, atom_output = inner_states, atom_output

        x = inner_states[-1].transpose(0, 1)

        if self.load_softmax:
            x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))
            x = self.embed_out(x)
            x = x + self.lm_output_learned_bias
        else:
            x = self.proj_out(x)

        return x, atom_output, {
            "inner_states": inner_states,
        }
