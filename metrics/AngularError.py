import torch
from torch import nn
from metrics.metric_base import Metric
import math

class AngularError(Metric):
    """
    Implementation of Angular Error metric.
    """
    def __init__(self, distance, screen_size=(1080, 1920), diagonal=40):
        super().__init__()
        """

        Parameters:
            distance (int, float): Distance from user's eyes to center of the screen, in centimeter (cm)
            screen_size (tuple[int, int]): Pixel unit width and height of the screen
            diagonal (int): the diagonal line length from the top to the bottom of the screen (inch)

        Returns:
            dict: Dictionary containing mean horizontal and vertical angular errors.
        """
        self.distance = distance / 2.54
        self.screen_size = screen_size
        self.diagonal = diagonal
        self.diagonal_pixels = math.sqrt(screen_size[0]**2 + screen_size[1]**2)
        # Calculate pixel size in inches
        self.pixel_size_inches = diagonal / self.diagonal_pixels
        
        # Calculate center of the screen in inches
        self.cx, self.cy = (self.screen_size[1] * self.pixel_size_inches) / 2, (self.screen_size[0] * self.pixel_size_inches) / 2

    def forward(self, inputs, targets):
        """
        Compute Angular Error metric.

        Parameters:
            inputs (torch.Tensor): Input tensor of shape (batch, 2).
            targets (torch.Tensor): Target tensor of shape (batch, 2).

        Returns:
            dict: Dictionary containing mean horizontal and vertical angular errors.
        """
        # Convert pixels to inches
        inputs_inch = inputs * torch.tensor([self.screen_size[1], self.screen_size[0]]) * self.pixel_size_inches
        targets_inch = targets * torch.tensor([self.screen_size[1], self.screen_size[0]]) * self.pixel_size_inches

        # Extract x and y coordinates
        x1, y1 = inputs_inch[:, 0], inputs_inch[:, 1]
        x2, y2 = targets_inch[:, 0], targets_inch[:, 1]

        # Calculate horizontal and vertical angular errors
        theta_input = (180 / math.pi) * torch.atan(abs(x1 - self.cx) / self.distance)
        theta_target = (180 / math.pi) * torch.atan(abs(x2 - self.cx) / self.distance)
        phi_input = (180 / math.pi) * torch.atan(abs(y1 - self.cy) / self.distance)
        phi_target = (180 / math.pi) * torch.atan(abs(y2 - self.cy) / self.distance)

        # Calculate the absolute differences
        horizontal_errors = torch.abs(theta_input - theta_target)
        vertical_errors = torch.abs(phi_input - phi_target)

        # Calculate mean errors
        mean_horizontal_error = torch.mean(horizontal_errors).item()
        mean_vertical_error = torch.mean(vertical_errors).item()

        return {
            'mean_horizontal_error': mean_horizontal_error,
            'mean_vertical_error': mean_vertical_error
        }
