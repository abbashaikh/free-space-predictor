import numpy as np
import torch
import torch.nn as nn
from trajdata import AgentBatch, AgentType


class SupportPredictor:
    """
    Class to predict the support of a given scenorio
    """

    def __init__(self, model):
        self.model = model

    def predict(self, data):
        """
        Predicts support values using the provided model.

        Parameters:
        - data: Input data for prediction.



        Returns:
        - Predicted support values.
        """
        return self.model.predict(data)
