import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        # Define a 3x3 convolutional layer for mask generation
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        # x is of shape (batch, 2, 64, 64)
        batch_size = x.size(0)

        # Step 1: Create an empty tensor of shape (batch, 1, 64, 64)
        empty_tensor = torch.zeros(batch_size, 1, 64, 64, device=x.device, dtype=x.dtype)

        # Step 2: Add the two channels together
        sum_tensor = x[:, 0:1, :, :] + x[:, 1:2, :, :]  # Shape: (batch, 1, 64, 64)

        # Step 3: Generate a mask by applying a 3x3 convolution followed by a sigmoid function
        mask = torch.sigmoid(self.conv(sum_tensor))

        # Step 4: Add the mask to the sum_tensor
        result_tensor = sum_tensor + mask  # Shape: (batch, 1, 64, 64)

        # Step 5: Concatenate the result with the original input tensor along the channel dimension
        output = torch.cat((x, result_tensor), dim=1)  # Shape: (batch, 3, 64, 64)

        return output

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.embedding = Embedding()
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Merge the first two dimensions: 1 and 2 -> (batch_size, 2, 64, 64)
        x = x.view(x.size(0), -1, x.size(3), x.size(4))
        x = self.embedding(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def resnet18(num_classes=6):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)