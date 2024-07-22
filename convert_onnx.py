from model.simple_convlstm import SimpleConvLSTM
import torch
import onnx
import onnxruntime as ort
import numpy as np
import os

model = SimpleConvLSTM(height=64, width=64, input_dim=2)
dummy_input = torch.randn(1, 64, 2, 64, 64)
os.makedirs("checkpoints/onnx", exist_ok=True)
# Save the model to an ONNX file
onnx_model_path = "checkpoints/onnx/simple_model.onnx"
torch.onnx.export(model, dummy_input, onnx_model_path, verbose=True, input_names=['input'], output_names=['output'])
print(f"Model saved to {onnx_model_path}")
onnx_model = onnx.load(onnx_model_path)

# Check that the model is valid
onnx.checker.check_model(onnx_model)
print("ONNX model is valid.")

# Create an ONNX Runtime session
session = ort.InferenceSession(onnx_model_path)

# Perform inference

results = session.run(["output"], {"input": dummy_input.numpy()})

print("Inference results:")
print(results[0])