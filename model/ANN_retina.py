import torch
import torch.nn as nn
from utils.ini_30.util import get_summary

# class DecimationLayer(nn.Module):
#     def __init__(
#         self,
#         spike_layer_class,
#         batch_size: int,
#         num_channels: int,
#         decimation_rate: float
#     ):
#         super().__init__() 
#         self.spike_layer_class = eval(spike_layer_class)  

#         self.conv = nn.Conv2d(
#             in_channels=num_channels,
#             out_channels=num_channels,
#             kernel_size=(1, 1),
#             stride=(1, 1),
#             bias=False)
        
#         self.spk = self.spike_layer_class(
#             batch_size=batch_size,
#             min_v_mem=-1.0,
#             spike_threshold=decimation_rate)
        
#         # Prevent kernel from being trained
#         self.conv.requires_grad_(False)
#         self.conv.weight.data = (torch.eye(num_channels, num_channels).unsqueeze(-1).unsqueeze(-1) )

#     def forward(self, x):

#         # Conv expects 4D input
#         input_dims = len(x.shape)
#         if input_dims == 5:
#             (n, t, c, h, w) = x.shape
#             x = x.reshape((n * t, c, h, w))

#         x = self.conv(x)  # (nt, c, h, w) 
#         x = self.spk(x)  # (n, t, c, h, w)

#         if input_dims == 5:
#             # Bring to original shape
#             x = x.reshape((n, t, c, h, w))
#         return x

class DecimationLayer(nn.Module):
    def __init__(self, decimation_rate):
        super().__init__()
        self.decimation_rate = decimation_rate
        self.pool = nn.MaxPool3d(kernel_size=self.decimation_rate, stride=self.decimation_rate)

    def forward(self, x):
        return self.pool(x)

class Retina(nn.Module):
    def __init__(self, dataset_params, training_params, layers_config):
        super(Retina, self).__init__()

        self.train_with_mem = training_params["train_with_mem"]
        self.train_with_exodus = training_params["train_with_exodus"]

        # Data configs
        self.num_bins = dataset_params["num_bins"]
        self.input_channel = dataset_params["input_channel"]
        self.img_width = dataset_params["img_width"]
        self.img_height = dataset_params["img_height"]

        # Train configs
        self.training_params = training_params

        # Modules initialization
        modules = []
        for i, layer in enumerate(layers_config):
            if layer["name"] == "Input":
                c_x, c_y, c_in = (
                    layer["img_width"],
                    layer["img_height"],
                    layer["input_channel"],
                )
                print(str(i), layer["name"], " in: \n ", c_x, c_y, c_in)

            elif layer["name"] == "Conv":
                # Input dimensions
                print(str(i), layer["name"], " in: \n ", c_x, c_y, c_in)
                modules.append(
                    nn.Conv2d(
                        in_channels=c_in,
                        out_channels=layer["out_dim"],
                        kernel_size=(layer["k_xy"], layer["k_xy"]),
                        stride=(layer["s_xy"], layer["s_xy"]),
                        padding=(layer["p_xy"], layer["p_xy"]),
                        bias=False,
                    )
                )

                # Output dimensions
                c_in = layer["out_dim"]
                c_x = ((c_x - layer["k_xy"] + 2 * layer["p_xy"]) // layer["s_xy"]) + 1
                c_y = ((c_y - layer["k_xy"] + 2 * layer["p_xy"]) // layer["s_xy"]) + 1
                print(str(i), layer["name"], " out: \n ", c_x, c_y, c_in)

                # Weights initialization
                torch.nn.init.xavier_uniform_(modules[-1].weight)

            elif layer["name"] == "ReLu":
                modules.append(nn.ReLU())

            elif layer["name"] == "BatchNorm":
                modules.append(nn.BatchNorm2d(c_in))

            elif layer["name"] == "AvgPool":
                # Input dimensions
                print(str(i), layer["name"], " in: \n ", c_x, c_y, c_in)
                modules.append(nn.AvgPool2d(layer["k_xy"], layer["s_xy"]))

                # Output dimensions
                c_x = (c_x - layer["k_xy"]) // layer["s_xy"] + 1
                c_y = (c_y - layer["k_xy"]) // layer["s_xy"] + 1
                print(str(i), layer["name"], " out: \n ", c_x, c_y, c_in)

            elif layer["name"] == "Flat":
                modules.append(nn.Flatten())

                # Output dimensions
                c_in = c_x * c_y * c_in

            elif layer["name"] == "Linear":
                # Input dimensions
                print(layer["name"], " in: \n ", c_in)
                modules.append(nn.Linear(c_in, layer["out_dim"], bias=False))

                # Output dimensions
                print(layer["name"], " out: \n ", layer["out_dim"])
                c_in = layer["out_dim"]

                # Weights initialization
                torch.nn.init.xavier_uniform_(modules[-1].weight)

            elif layer["name"] == "SumPool":
                # Input dimensions
                print(str(i), layer["name"], " in: \n ", c_x, c_y, c_in)
                modules.append(sinabs.layers.SumPool2d(layer["k_xy"], layer["s_xy"]))

                # Output dimensions
                c_x = (c_x - layer["k_xy"]) // layer["s_xy"] + 1
                c_y = (c_y - layer["k_xy"]) // layer["s_xy"] + 1
                print(str(i), layer["name"], " out: \n ", c_x, c_y, c_in)

            elif layer["name"] == "Decimation":
                modules.append(
                    DecimationLayer(
                        decimation_rate=layer["decimation_rate"]
                    )
                )
            else:
                raise NotImplementedError("Unknown Layer")

        self.seq = nn.Sequential(*modules)
        get_summary(self)

    def compute_mac_operations(self):
        total_mac_ops = 0
        input_size = (
            self.training_params["batch_size"] * self.num_bins,
            self.input_channel,
            self.img_width,
            self.img_height,
        )
        with torch.no_grad():
            x = torch.zeros(*input_size)
            for module in self.seq:
                x = module(x)
                if isinstance(module, nn.Conv2d):
                    total_mac_ops += (
                        module.in_channels
                        * module.out_channels
                        * module.kernel_size[0] ** 2
                        * x.size(-1)
                        * x.size(-2)
                    )
                elif isinstance(module, nn.Linear):
                    total_mac_ops += module.in_features * module.out_features

        return total_mac_ops

    def forward(self, x):
        b, t, c, w, h = x.shape
        x = x.reshape(b * t, c, w, h)
        x = self.seq(x)
        x = x.reshape(b, t, 4, 4, 10)
        return x
