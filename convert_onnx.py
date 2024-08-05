from model.simple_convlstm2 import SimpleConvLSTM2
import torch
import onnx
import onnxruntime as ort
import numpy as np
import os

# Initialize your model
model = SimpleConvLSTM2(height=64, width=64, input_dim=2)

# Create dummy input tensors
dummy_input = torch.randn(1, 30, 2, 64, 64)
hidden_state = [
    [(torch.rand(1, 8, 64, 64), torch.rand(1, 8, 64, 64))],
    [(torch.rand(1, 16, 32, 32), torch.rand(1, 16, 32, 32))],
    [(torch.rand(1, 32, 16, 16), torch.rand(1, 32, 16, 16))],
    [(torch.rand(1, 64, 8, 8), torch.rand(1, 64, 8, 8))],
]
last_out = torch.randn(1, 3)  # Adjust dimensions if necessary

# Create the ONNX export directory if it doesn't exist
os.makedirs("checkpoints/onnx", exist_ok=True)

# Save the model to an ONNX file
onnx_model_path = "checkpoints/onnx/simple_model.onnx"
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
    opset_version=11  # Make sure to set the opset version if necessary
)
print(f"Model saved to {onnx_model_path}")

# Load the ONNX model
onnx_model = onnx.load(onnx_model_path)

# Check that the model is valid
onnx.checker.check_model(onnx_model)
# Print a human readable representation of the graph
print(onnx.helper.printable_graph(onnx_model.graph))
print("ONNX model is valid.")

# Create an ONNX Runtime session
session = ort.InferenceSession(onnx_model_path)

dummy_input = np.random.randn(1, 30, 2, 64, 64).astype(np.float32)
# Convert hidden states and last_out to numpy arrays for ONNX Runtime
hidden_state_numpy = [
    (np.random.randn(1, 8, 64, 64).astype(np.float32), np.random.randn(1, 8, 64, 64).astype(np.float32)),
    (np.random.randn(1, 16, 32, 32).astype(np.float32), np.random.randn(1, 16, 32, 32).astype(np.float32)),
    (np.random.randn(1, 32, 16, 16).astype(np.float32), np.random.randn(1, 32, 16, 16).astype(np.float32)),
    (np.random.randn(1, 64, 8, 8).astype(np.float32), np.random.randn(1, 64, 8, 8).astype(np.float32)),
]
last_out_numpy = np.random.randn(1, 3).astype(np.float32)

# Prepare input feed
input_feed = {
    "input": dummy_input,
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

# Perform inference
results = session.run(["output", "hidden"], input_feed)

print("Inference results:")
print("Output:", results[0])
print("Hidden states:", results[1])
