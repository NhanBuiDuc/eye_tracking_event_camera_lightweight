import torch
from torchsummary import summary
from model.B_3ET import Baseline_3ET
from fvcore.nn import FlopCountAnalysis
import time
import os

def get_device_info():
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
    else:
        device_name = os.popen("cat /proc/cpuinfo | grep 'model name' | uniq").read().strip()
    return device_name

def measure_inference_time(model, input_tensor, dtype, device, num_runs=100):
    input_tensor = input_tensor.to(dtype).to(device)
    
    # Warm-up
    for _ in range(10):
        _ = model(input_tensor)
    
    # Measure inference time
    start_time = time.time()
    for _ in range(num_runs):
        _ = model(input_tensor)
    end_time = time.time()
    
    avg_inference_time = (end_time - start_time) / num_runs
    return avg_inference_time

def model_analysis(model, input_size, batch_sizes, device):
    model.to(device)
    
    # Print model summary
    summary(model, input_size=input_size)
    
    # Number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")
    
    # FLOPs
    input_tensor = torch.randn(1, *input_size).to(device)
    flops = FlopCountAnalysis(model, input_tensor)
    print(f"FLOPs: {flops.total() / 1e9:.2f} GFLOPs")
    
    model.eval()
    
    # Measure inference time for different batch sizes
    for batch_size in batch_sizes:
        input_tensor_float32 = torch.randn(batch_size, *input_size, dtype=torch.float32)
        avg_inference_time_float32 = measure_inference_time(model, input_tensor_float32, device, torch.float32 )
        print(f"Average inference time for float32 with batch size {batch_size}: {avg_inference_time_float32 * 1000:.2f} ms")
    
    # Print device information
    device_info = get_device_info()
    print(f"Device: {device_info}")

# Define your model and input size here
model = Baseline_3ET(
    height=64,
    width=64,
    input_dim=2
)
input_size = (40, 2, 64, 64)  # Example input size
batch_sizes = [1, 8, 16, 32, 64]  # List of batch sizes to test

model_analysis(model, input_size, batch_sizes, device = "cpu")
