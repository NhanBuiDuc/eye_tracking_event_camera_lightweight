import torch
import onnx
import onnxruntime as ort
import numpy as np
import os
import time
import platform
import psutil
from model.simple_convlstm import SimpleConvLSTM

def get_cpu_info():
    # Basic CPU info
    cpu_info = {
        "CPU Name": platform.processor(),
        "Physical Cores": psutil.cpu_count(logical=False),
        "Logical Cores": psutil.cpu_count(logical=True),
        "CPU Frequency (GHz)": psutil.cpu_freq().current / 1000
    }
    return cpu_info

def print_cpu_info():
    cpu_info = get_cpu_info()
    print("CPU Information:")
    for key, value in cpu_info.items():
        print(f"{key}: {value}")

def measure_inference_time_onnx(session, input_tensor, num_runs=100):
    # Warm-up
    for _ in range(10):
        _ = session.run(None, {"input": input_tensor})

    # Measure inference time
    start_time = time.time()
    for _ in range(num_runs):
        _ = session.run(None, {"input": input_tensor})
    end_time = time.time()

    avg_inference_time = (end_time - start_time) / num_runs
    return avg_inference_time

def model_analysis_onnx(session, input_shapes):
    # Measure inference time for different input shapes
    for input_shape in input_shapes:
        input_tensor = np.random.randn(*input_shape).astype(np.float32)
        print(f"Testing input shape: {input_tensor.shape}")
        avg_inference_time = measure_inference_time_onnx(session, input_tensor)
        print(f"Average inference time for input shape {input_shape}: {avg_inference_time * 1000:.2f} ms")

    # Print CPU information
    print_cpu_info()

# Define your model and input size here
model = SimpleConvLSTM(height=64, width=64, input_dim=2)
dummy_input = torch.randn(1, 8, 2, 64, 64)

# Save the model to an ONNX file with dynamic axes
os.makedirs("checkpoints/onnx", exist_ok=True)
onnx_model_path = "checkpoints/onnx/simple_model.onnx"
dynamic_axes = {'input': {0: 'batch_size', 1: 'timesteps'}, 'output': {0: 'batch_size', 1: 'timesteps'}}
torch.onnx.export(model, dummy_input, onnx_model_path, verbose=True, input_names=['input'], output_names=['output'], dynamic_axes=dynamic_axes)
print(f"Model saved to {onnx_model_path}")

# Load and check the ONNX model
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)
print("ONNX model is valid.")

# Create an ONNX Runtime session with CPU only
session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

# Define input shapes with different batch sizes and timesteps
batch_sizes = [1, 8, 16, 32, 64]
timesteps = [8, 16, 32, 64]

input_shapes = [(batch_size, timestep, 2, 64, 64) for batch_size in batch_sizes for timestep in timesteps]

# Perform model analysis and measure inference time for ONNX model with different batch sizes and timesteps
model_analysis_onnx(session, input_shapes)
