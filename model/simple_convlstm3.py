import tqdm
import tables
import os
import pdb
import cv2
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from thop import profile
from scipy.ndimage import median_filter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader

# from training.models.utils import get_summary

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state 
        combined = torch.cat([input_tensor, h_cur.to(input_tensor.device)], dim=1)  # concatenate along channel axis
        combined_conv = torch.relu(self.conv(combined.to(self.conv.weight.device)))
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, height, width, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.height = height
        self.width = width
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

        # Define a linear layer to transform tensor_2
        self.hidden_linear = nn.Linear(6, hidden_dim[0] * height * width)  # Transform 3 features to 4096 (64*6
        # Define a convolutional layer to reduce back to 8 channels
        self.hidden_conv_h = nn.Conv2d(in_channels=hidden_dim[0] * 2, out_channels=hidden_dim[0], kernel_size=1, stride=1, padding=0)
        self.hidden_conv_c = nn.Conv2d(in_channels=hidden_dim[0] * 2, out_channels=hidden_dim[0], kernel_size=1, stride=1, padding=0)
    def forward(self, input_tensor, hidden_state=None, last_out = None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
            
        b, t, c, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is None:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))
            # Apply the linear layer to tensor_2
        if last_out is not None:
            hidden_state_list = []
            for layer_idx in range(self.num_layers):
                h, c = hidden_state[layer_idx]
                out_feat = self.hidden_linear(last_out)  # Shape: (1, 4096)
                out_feat = out_feat.view(h.shape[0], h.shape[1], h.shape[2], h.shape[3])
                concatenated_h_feat = torch.cat((h, out_feat), dim=1)  # Shape: (1, 16, 64, 64)
                concatenated_c_feat = torch.cat((c, out_feat), dim=1)  # Shape: (1, 16, 64, 64)
                h = self.hidden_conv_h(concatenated_h_feat)
                c = self.hidden_conv_c(concatenated_c_feat)
                hidden_state_list.append((h, c))

            hidden_state = hidden_state_list

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    # init hidden states for sub cell layers
    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
        
class SimpleConvLSTM2(nn.Module):
    def __init__(self, height, width, input_dim):
        super(SimpleConvLSTM2, self).__init__() 

        self.convlstm1 = ConvLSTM(input_dim=input_dim, hidden_dim=8, height = height, width = width, kernel_size=(3, 3), num_layers=1, batch_first=True)
        self.bn1 = nn.BatchNorm3d(8)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.convlstm2 = ConvLSTM(input_dim=8, hidden_dim=16, height = height//2, width = width//2, kernel_size=(3, 3), num_layers=1, batch_first=True)
        self.bn2 = nn.BatchNorm3d(16)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.convlstm3 = ConvLSTM(input_dim=16, hidden_dim=32, height = height//4, width = width//4, kernel_size=(3, 3), num_layers=1, batch_first=True)
        self.bn3 = nn.BatchNorm3d(32)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.convlstm4 = ConvLSTM(input_dim=32, hidden_dim=64, height = height//8, width = width//8, kernel_size=(3, 3), num_layers=1, batch_first=True)
        self.bn4 = nn.BatchNorm3d(64)
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.fc1 = nn.Linear(1024, 512)
        # self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 6)
        # get_summary(self)

    def forward(self, x, hidden_states_input=None, last_out = None):

        hidden_states = []
        if hidden_states_input is None or hidden_states_input[0] is None:
            h1 = None
        else:
            h1 = hidden_states_input[0]

        if last_out is None:
            x, h1 = self.convlstm1(x, h1, None)
        else:
            x, h1 = self.convlstm1(x, h1, last_out.clone())
        x = x[0].permute(0, 2, 1, 3, 4)
        # x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        hidden_states.append(h1)

        if hidden_states_input is None or hidden_states_input[1] is None:
            h2 = None
        else:
            h2 = hidden_states_input[1]
        x = x.permute(0, 2, 1, 3, 4)

        if last_out is None:
            x, h2 = self.convlstm2(x, h2, None)
        else:
            x, h2 = self.convlstm2(x, h2, last_out.clone())

        x = x[0].permute(0, 2, 1, 3, 4)
        # x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        hidden_states.append(h2)
        
        if hidden_states_input is None or hidden_states_input[2] is None:
            h3 = None
        else:
            h3 = hidden_states_input[2]

        x = x.permute(0, 2, 1, 3, 4)

        if last_out is None:
            x, h3 = self.convlstm3(x, h3, None)
        else:
            x, h3 = self.convlstm3(x, h3, last_out.clone())

        x = x[0].permute(0, 2, 1, 3, 4)
        # x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        hidden_states.append(h3)

        if hidden_states_input is None or hidden_states_input[3] is None:
            h4 = None
        else:
            h4 = hidden_states_input[3]

        x = x.permute(0, 2, 1, 3, 4)

        if last_out is None:
            x, h4 = self.convlstm4(x, h4, None)
        else:
            x, h4 = self.convlstm4(x, h4, last_out.clone())

        x = x[0].permute(0, 2, 1, 3, 4)
        # x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)
        hidden_states.append(h4)

        # Flatten and apply LSTM layer
        x_list=[]
        b, c, seq, h, w = x.size()
        for t in range(seq): 
            data = x[:,:,t,:,:]
            data = data.reshape(b, -1)
            data = F.relu(self.fc1(data))
            # data = self.drop(data)
            data = self.fc2(data)

        coordinate_pred = data[:, :2]  # Shape: (batch_size, 2, h, w)

        # 2. Apply softmax to the last four channels
        state_pred = data[:, 2:]  # Shape: (batch_size, 4, h, w)
        state_pred = F.softmax(state_pred, dim=1)  # Apply softmax on the class dimension

        # 3. Concatenate the results back together
        data = torch.cat((coordinate_pred, state_pred), dim=1)
        return data, hidden_states




# if __name__ == "__main__":
#     import torch
#     from thop import profile

#     # Assuming 'Baseline_3ET' is the model class and 'device' is defined
#     model = Baseline_3ET(64, 64, 1, torch.device("cuda"))  # Create an instance of the model
#     input_tensor = torch.randn(1, 2, 1, 64, 64).to(torch.device("cuda"))  # Define an example input tensor
#     macs, params = profile(model, inputs=(input_tensor,))
#     print(f"Number of MAC operations: {macs}")