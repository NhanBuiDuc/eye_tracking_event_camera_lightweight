from model.simple_convlstm2 import SimpleConvLSTM2
import torch
import torch.nn as nn


input = torch.rand([1, 30, 2, 64, 64])


model = SimpleConvLSTM2(64, 64, 2)

result, hidden = model(input, None)
print(result)
result, hidden = model(input, hidden, result)