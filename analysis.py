import torch
from data.dataset.ini_30_dataset import Ini30Dataset
import time
import psutil  # To measure memory usage
import os

def run_inference_on_cpu(dataset_config_path):
    # Initialize the dataset
    device = "cpu"  # Run on CPU
    split= "train"
    dataset = Ini30Dataset(split, device, dataset_config_path)

    # Choose an index to process
    index = 0

    # Measure inference time
    start_time = time.time()

    # Get a single batch item (batch size of 1)
    dataset[index]
    # event_tensor = dataset.prepare_x(index)

    # Calculate inference time
    inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds

    # Calculate memory consumption
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024 ** 2  # Memory usage in MB

    # Print results
    print(f"Inference time: {inference_time:.4f} milliseconds")
    print(f"Memory usage: {memory_usage:.2f} MB")

    return event_tensor

# Example usage:
config_path = "config\ini_30.json"
event_tensor= run_inference_on_cpu(config_path)
print("Event tensor shape:", event_tensor[0].float().shape)