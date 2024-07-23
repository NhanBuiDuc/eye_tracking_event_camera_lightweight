import torch
from torch import nn
from metrics.metric_base import Metric

class AngularError(Metric):
    """
    Implementation of Angular Error metric.
    """
    def __init__(self, distance, screen_size=(1080, 1920), diagonal = 40):
        super().__init__()
        self.distance = distance
        self.screen_size = screen_size

    def forward(self, inputs, targets):
        """
        Compute Angular Error metric.

        Parameters:
            inputs (torch.Tensor): Input tensor of shape (batch, 2).
            targets (torch.Tensor): Target tensor of shape (batch, 2).

        Returns:
            dict: Dictionary containing mean horizontal and vertical angular errors.
        """
        # Extract x and y coordinates
        x1, y1 = inputs[:, 0] * self.screen_size[0], inputs[:, 1] * self.screen_size[1]
        x2, y2 = targets[:, 0] * self.screen_size[0], targets[:, 1] * self.screen_size[1]

