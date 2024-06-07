import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, channels, batch_norm=False):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(channels, channels, 3, stride=1, padding=1)
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.batch_norm_layer = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv(x)
        if self.batch_norm:
            y = self.batch_norm_layer(y)
        y = self.relu(y)
        return y + x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),

            nn.Dropout2d(0.5),
            nn.MaxPool2d(2, 2),

            ConvBlock(16, batch_norm=True),
            nn.MaxPool2d(2, 2),
            ConvBlock(16, batch_norm=True),

            nn.Dropout2d(0.5),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),

            ConvBlock(32, batch_norm=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.5),
            ConvBlock(32, batch_norm=True),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 16 * 16, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x
