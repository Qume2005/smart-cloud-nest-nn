from typing import final

import torch
from torch import nn

@final
class MainNnModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.padding = nn.ZeroPad2d((0, 0, 1, 1))
        self.conv1 = nn.Conv2d(1, 1, 3, padding=1)
        self.conv2 = nn.Conv2d(4, 4, 3, padding=1)
        self.pool1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=2, stride=2, padding=0)
        self.pool2 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=2, stride=2, padding=0)
        self.pool3 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=2, stride=2, padding=0)
        self.pool4 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2, stride=2, padding=0)
        self.flatten = nn.Flatten()
        self.to_proto = nn.Linear(in_features=16, out_features=8)
        self.protos = nn.Linear(in_features=8, out_features=3)

    def forward(self, x):
        x = self.relu(self.conv1(x)) + x
        x = self.padding(x)
        x = self.pool1(x)
        x = self.relu(x)
        x = self.padding(x)
        x = self.pool2(x)
        x = self.relu(self.conv2(x)) + x
        x = self.pool3(x)
        x = self.relu(x)
        x = self.pool4(x)
        x = self.flatten(x)
        x = self.to_proto(x)
        x = -torch.cdist(x, self.protos.weight, p=2) - self.protos.bias
        return x