import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


#######################
#   Support Module    #
#######################
class EMB2S(nn.Module):
    """Linear layer to calculate support from human & robot embeddings

    Assume human-robot embeddings have been concatenated for easy input

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        num_humans,
        embedding_size,
    ):
        super(EMB2S, self).__init__()
        self.layers = nn.ModuleDict(
            {
                "layer1": nn.Linear(num_humans * embedding_size, 256),
                "layer2": nn.Linear(256, 256),
                "layer3": nn.Linear(256, 256),
                "layer4": nn.Linear(256, 1),
            }
        )

    def forward(
        self,
        embeddings_human_robot,
    ):
        """Forward Pass of Linear Layer"""
        x = embeddings_human_robot
        for layer_name, layer in self.layers.items():
            x = F.relu(layer(x)) if layer_name != "layer4" else layer(x)
        return x
