import torch
from torch import nn
import torch.nn.functional as F
from metrics.metric_base import Metric
from loss.YoloLoss import YoloLoss
def euclidean_distance(tensor1, tensor2):
    # Ensure tensors have the same shape (batch, 2)
    assert tensor1.shape == tensor2.shape, "Tensors must have the same shape (batch, 2)"
    
    # Compute squared differences along the second dimension (axis=1)
    squared_diff = (tensor1 - tensor2)**2
    
    # Sum along the second dimension to get squared Euclidean distances
    squared_dist = squared_diff.sum(dim=0)
    
    # Take the square root to get Euclidean distances
    distances = torch.sqrt(squared_dist)
    
    return distances

class YoloMetric(Metric):
    """
    Implementation of Mean Squared Error metric.
    """
    def __init__(self, dataset_params, training_params, detection_rate_thresholds = [3, 5, 10], image_size = (640, 480)):
        super().__init__()
        self.yolo_loss = YoloLoss(dataset_params, training_params)
        self.detection_rate_thresholds = detection_rate_thresholds
        self.image_size = image_size
    def forward(self, inputs, targets):
        """
        Compute Mean Squared Error metric.

        Parameters:
            inputs (list of torch.Tensor): List of input tensors.
            targets (list of torch.Tensor): List of target tensors.

        Returns:
            float: Mean Squared Error value.
        """
        total_yolo_loss = self.yolo_loss(inputs, targets)
        memory = self.yolo_loss.memory
        loss = self.yolo_loss.loss
        detection_rate = self.calculate_detection_rate(memory["points"]["target"], memory["points"]["pred"])
        return {
            "total_yolo_loss": total_yolo_loss, 
            "memory": memory,
            "loss": loss,
            "detection_rate": detection_rate
        }

    def calculate_detection_rate(self, inputs, targets):
        inputs[:, 0] = inputs[:, 0] * self.image_size[0]
        inputs[:, 1] = inputs[:, 1] * self.image_size[1]
        
        inputs[:, 0] = inputs[:, 0] * (512 // self.image_size[0])
        inputs[:, 1] = inputs[:, 1] * (512 // self.image_size[1])      

        targets[:, 0] = targets[:, 0] * self.image_size[0]
        targets[:, 1] = targets[:, 1] * self.image_size[1]
        
        targets[:, 0] = targets[:, 0] * (512 // self.image_size[0])
        targets[:, 1] = targets[:, 1] * (512 // self.image_size[1])      
        results = {}
        for threshold in self.detection_rate_thresholds:
            count = 0
            for inp, target in zip(inputs, targets):
                distances_error = euclidean_distance(inp, target)
                if distances_error < threshold:
                    count += 1
            rate = (count / len(inputs))
            results[threshold] = rate
        return results

    def angular_error(self, inputs, targets):
        """
        Calculate the angular error between predicted and ground truth gaze vectors.
        
        Args:
            inputs (torch.Tensor): Predicted gaze vectors of shape (batch, 2).
            targets (torch.Tensor): Ground truth gaze vectors of shape (batch, 2).
        
        Returns:
            torch.Tensor: Angular errors for each element in the batch of shape (batch,).
        """
        # Normalize the vectors
        inputs[:, 0] = inputs[:, 0] * self.image_size[0]
        inputs[:, 1] = inputs[:, 1] * self.image_size[1]
        
        inputs[:, 0] = inputs[:, 0] * (512 // self.image_size[0])
        inputs[:, 1] = inputs[:, 1] * (512 // self.image_size[1])      

        targets[:, 0] = targets[:, 0] * self.image_size[0]
        targets[:, 1] = targets[:, 1] * self.image_size[1]
        
        targets[:, 0] = targets[:, 0] * (512 // self.image_size[0])
        targets[:, 1] = targets[:, 1] * (512 // self.image_size[1])      
        
        # Calculate the dot product between the normalized vectors
        dot_product = (inputs * targets).sum(dim=1)
        
        # Clamp the dot product to avoid numerical errors
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        
        # Calculate the angular error in radians
        angular_error_radians = torch.acos(dot_product)
        
        # Convert the angular error from radians to degrees
        angular_error_degrees = torch.rad2deg(angular_error_radians)
        
        return angular_error_degrees