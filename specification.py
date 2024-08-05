import torch
import onnx
import onnxruntime as ort
import numpy as np
import os
import time
import platform
import psutil
from model.simple_convlstm2 import SimpleConvLSTM2
from torchsummary import summary
from fvcore.nn import FlopCountAnalysis, parameter_count_table

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

def measure_inference_time_onnx(session, input_feed, num_runs=100):
    # Warm-up
    for _ in range(10):
        results, hidden = session.run(["output", "hidden"], input_feed)

    # Measure inference time
    start_time = time.time()
    for _ in range(num_runs):
        results, hidden = session.run(["output", "hidden"], input_feed)
    end_time = time.time()

    avg_inference_time = (end_time - start_time) / num_runs
    return avg_inference_time

def model_analysis_onnx(session, input_shapes):
    # Measure inference time for different input shapes
    for input_shape in input_shapes:
        input_tensor = np.random.randn(*input_shape).astype(np.float32)
        batch_size = input_tensor.shape[0]
        hidden_state_numpy = [
            (np.random.randn(batch_size, 8, 64, 64).astype(np.float32), np.random.randn(batch_size, 8, 64, 64).astype(np.float32)),
            (np.random.randn(batch_size, 16, 32, 32).astype(np.float32), np.random.randn(batch_size, 16, 32, 32).astype(np.float32)),
            (np.random.randn(batch_size, 32, 16, 16).astype(np.float32), np.random.randn(batch_size, 32, 16, 16).astype(np.float32)),
            (np.random.randn(batch_size, 64, 8, 8).astype(np.float32), np.random.randn(batch_size, 64, 8, 8).astype(np.float32)),
        ]
        last_out_numpy = np.random.randn(batch_size, 3).astype(np.float32)
        # Prepare input feed
        input_feed = {
            "input": input_tensor,
            "hidden_state_0_h": hidden_state_numpy[0][0],
            "hidden_state_0_c": hidden_state_numpy[0][1],
            "hidden_state_1_h": hidden_state_numpy[1][0],
            "hidden_state_1_c": hidden_state_numpy[1][1],
            "hidden_state_2_h": hidden_state_numpy[2][0],
            "hidden_state_2_c": hidden_state_numpy[2][1],
            "hidden_state_3_h": hidden_state_numpy[3][0],
            "hidden_state_3_c": hidden_state_numpy[3][1],
            "last_out": last_out_numpy
        }
        print(f"Testing input shape: {input_tensor.shape}")
        avg_inference_time = measure_inference_time_onnx(session, input_feed)
        print(f"Average inference time for input shape {input_shape}: {avg_inference_time * 1000:.2f} ms")

    # Print CPU information
    print_cpu_info()

# Initialize your model
model = SimpleConvLSTM2(height=64, width=64, input_dim=2)
# Print model summary (parameters)
summary(model, input_size=(30, 2, 64, 64))
# Create dummy input to calculate FLOPs
dummy_input = torch.randn(1, 30, 2, 64, 64).to('cuda')

# Calculate FLOPs
flop_counter = FlopCountAnalysis(model, dummy_input)
flops = flop_counter.total()

time_steps = [30, 60]
for time_step in time_steps:

    # Create dummy input tensors
    dummy_input = torch.randn(1, time_step, 2, 64, 64)
    hidden_state = [
        [(torch.rand(1, 8, 64, 64), torch.rand(1, 8, 64, 64))],
        [(torch.rand(1, 16, 32, 32), torch.rand(1, 16, 32, 32))],
        [(torch.rand(1, 32, 16, 16), torch.rand(1, 32, 16, 16))],
        [(torch.rand(1, 64, 8, 8), torch.rand(1, 64, 8, 8))],
    ]
    last_out = torch.randn(1, 3)  # Adjust dimensions if necessary


    # Save the model to an ONNX file with dynamic axes
    os.makedirs("checkpoints/onnx", exist_ok=True)
    onnx_model_path = "checkpoints/onnx/simple_model.onnx"
    dynamic_axes = {'input': {0: 'batch_size', 1: 'timesteps'}, 'output': {0: 'batch_size', 1: 'timesteps'}}
    torch.onnx.export(
        model,
        (dummy_input, hidden_state, last_out),
        onnx_model_path,
        verbose=True,
        input_names=['input', 'hidden_state_0_h', 'hidden_state_0_c', 
                    'hidden_state_1_h', 'hidden_state_1_c', 
                    'hidden_state_2_h', 'hidden_state_2_c', 
                    'hidden_state_3_h', 'hidden_state_3_c', 'last_out'],
        output_names=['output', 'hidden'],
        opset_version=11,  # Make sure to set the opset version if necessary
        dynamic_axes=dynamic_axes
    )
    # torch.onnx.export(model, dummy_input, onnx_model_path, verbose=True, input_names=['input'], output_names=['output'], dynamic_axes=dynamic_axes)
    print(f"Model saved to {onnx_model_path}")

    # Load and check the ONNX model
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid.")

    # Create an ONNX Runtime session with CPU only
    session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

    input_tensor = np.random.randn(1, time_step, 2, 64, 64).astype(np.float32)
    batch_size = input_tensor.shape[0]
    hidden_state_numpy = [
        (np.random.randn(batch_size, 8, 64, 64).astype(np.float32), np.random.randn(batch_size, 8, 64, 64).astype(np.float32)),
        (np.random.randn(batch_size, 16, 32, 32).astype(np.float32), np.random.randn(batch_size, 16, 32, 32).astype(np.float32)),
        (np.random.randn(batch_size, 32, 16, 16).astype(np.float32), np.random.randn(batch_size, 32, 16, 16).astype(np.float32)),
        (np.random.randn(batch_size, 64, 8, 8).astype(np.float32), np.random.randn(batch_size, 64, 8, 8).astype(np.float32)),
    ]
    last_out_numpy = np.random.randn(batch_size, 3).astype(np.float32)
    # Prepare input feed
    input_feed = {
        "input": input_tensor,
        "hidden_state_0_h": hidden_state_numpy[0][0],
        "hidden_state_0_c": hidden_state_numpy[0][1],
        "hidden_state_1_h": hidden_state_numpy[1][0],
        "hidden_state_1_c": hidden_state_numpy[1][1],
        "hidden_state_2_h": hidden_state_numpy[2][0],
        "hidden_state_2_c": hidden_state_numpy[2][1],
        "hidden_state_3_h": hidden_state_numpy[3][0],
        "hidden_state_3_c": hidden_state_numpy[3][1],
        "last_out": last_out_numpy
    }
    print(f"Testing input shape: {input_tensor.shape}")
    avg_inference_time = measure_inference_time_onnx(session, input_feed)
    print(f"Average inference time for input shape {input_tensor.shape}: {avg_inference_time * 1000:.2f} ms")

    # Print CPU information
    print_cpu_info()