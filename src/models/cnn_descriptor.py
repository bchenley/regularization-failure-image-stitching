# Author: Brandon Henley

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNDescriptor(nn.Module):
    def __init__(self, input_shape=(3, 64, 64), use_dropout=False, dropout_rate=0.3,
                 conv_channels=[32, 64, 64, 128], kernel_sizes=[3, 3, 3, 3], descriptor_dim=(64, 32)):
        super(CNNDescriptor, self).__init__()

        self.use_dropout = use_dropout
        c, h, w = input_shape

        assert len(conv_channels) == len(kernel_sizes), "conv_channels and kernel_sizes must match in length"

        layers = []
        in_channels = c
        for i, (out_channels, k) in enumerate(zip(conv_channels, kernel_sizes)):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=k // 2))
            layers.append(nn.ReLU())
            if i < len(conv_channels) - 1:
                layers.append(nn.MaxPool2d(2))
            else:
                layers.append(nn.AdaptiveAvgPool2d((1, 1)))
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.head = nn.Sequential(
            nn.Linear(conv_channels[-1], descriptor_dim[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if use_dropout else nn.Identity(),
            nn.Linear(descriptor_dim[0], descriptor_dim[1])  # Final descriptor dimensionality
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.head(x)
        return F.normalize(x, p=2, dim=1)
