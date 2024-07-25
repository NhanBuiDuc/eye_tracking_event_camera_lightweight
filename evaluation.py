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

outputs = torch.tensor(outputs)
targets = torch.tensor(targets)
# outputs = torch.rand([1 * 64, 4, 4, 10])
config_path = "config/evb_eye.json"

if config_path is not None:
    with open(config_path, 'r') as f:
        config_params = json.load(f)
dataset_params = config_params["dataset_params"]
training_params = config_params["training_params"]

metrics = []
metrics.append(AngularError(dataset_params["distance_user_camera"]))
metric_sequence = MetricSequence(metrics)
metrics = metric_sequence(outputs, targets)
metric_sequence.to_csv(path="log/metrics.csv", keys_to_log=["mean_horizontal_error", "mean_vertical_error"])