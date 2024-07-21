import os
import numpy as np
from metrics.AngularError import AngularError
from metrics.metric_base import MetricSequence
import json
import torch

def read_output_target(folder_path):
    output_arrays = []
    target_arrays = []
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npy'):
            if 'output' in file_name:
                array = np.load(os.path.join(folder_path, file_name))
                output_arrays.append(array)
            elif 'target' in file_name:
                array = np.load(os.path.join(folder_path, file_name))
                target_arrays.append(array)
    
    return output_arrays, target_arrays

# Example usage:
folder_path = 'cache/train'
output_arrays, target_arrays = read_output_target(folder_path)
outputs = np.concatenate(output_arrays, axis=0)
targets = np.concatenate(target_arrays, axis=0)
outputs = outputs.reshape([outputs.shape[0] * outputs.shape[1], outputs.shape[2], outputs.shape[3], outputs.shape[4], outputs.shape[5]])
targets = targets.reshape([targets.shape[0] * targets.shape[1], targets.shape[2], targets.shape[3], targets.shape[4], targets.shape[5]])

outputs = torch.tensor(outputs)
targets = torch.tensor(targets)
# outputs = torch.rand([1 * 64, 4, 4, 10])
config_path = "config/ini_30.json"

if config_path is not None:
    with open(config_path, 'r') as f:
        config_params = json.load(f)
dataset_params = config_params["dataset_params"]
training_params = config_params["training_params"]

metrics = []
metrics.append(AngularError(dataset_params, training_params))
metric_sequence = MetricSequence(metrics)
metrics = metric_sequence(outputs, targets)
metric_sequence.to_csv(path="log/metrics.csv", keys_to_log=["total_yolo_loss", "loss", "detection_rate"])