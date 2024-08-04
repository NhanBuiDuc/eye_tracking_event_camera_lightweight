import torch
import torch.nn as nn
import torch.nn.functional as F

class EyeGazeLoss(nn.Module):
    def __init__(self):
        super(EyeGazeLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, output, target):
        # Split the output and target tensors
        output_mse = output[:, :2]
        target_mse = target[:, :2]
        
        output_softmax = output[:, 2]
        target_softmax = target[:, 2].long()  # Ensure the target for softmax is of type long
        
        # Compute the MSE loss for the first two dimensions
        mse_loss = self.mse_loss(output_mse, target_mse)
        
        # Compute the CrossEntropy loss for the third dimension
        cross_entropy_loss = self.cross_entropy_loss(output_softmax.unsqueeze(1), target_softmax)
        
        # Combine the losses
        loss = 0.7 * mse_loss + 0.3 * cross_entropy_loss
        return loss