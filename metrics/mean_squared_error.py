import torch
from torch import nn
import torch.nn.functional as F
from metrics.metric_base import Metric

class MeanSquaredError(Metric):
    """
    Implementation of Mean Squared Error metric.
    """

    def forward(self, inputs, targets):
        """
        Compute Mean Squared Error metric.

        Parameters:
            inputs (list of torch.Tensor): List of input tensors.
            targets (list of torch.Tensor): List of target tensors.

        Returns:
            float: Mean Squared Error value.
        """
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.stack(inputs)
        if not isinstance(targets, torch.Tensor):
            targets = torch.stack(targets)
        mse_value = F.mse_loss(inputs, targets).item()
        return mse_value
