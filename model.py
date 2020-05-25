import torch
import torch.nn as nn
import torch.nn.functional as F

class Siamese(nn.Module):

    def __init__(self):
        super(Siamese, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 10), # 64 @ 96 X 96
            nn.ReLU(inplace=True), # 64 @ 96 X 96
            nn.MaxPool2d(2), # 64 @ 48 X 48
            nn.Conv2d(64, 128, 7), # 128 @ 42 X 42
            nn.ReLU(), # 128 @ 42 X 42
            nn.MaxPool2d(2), # 128 @ 21 X 21
            nn.Conv2d(128, 128, 4), # 128 @ 18 X 18
            nn.MaxPool2d(2), # 128 @ 9 X 9
            nn.ReLU(), # 128 @ 9 X 9
            nn.Conv2d(128, 256, 4), # 256 @ 6 X 6 
            nn.ReLU(),# 256 @ 6 X 6 
        )
        self.linear = nn.Sequential(nn.Linear(256*6*6, 4096), nn.Sigmoid())
        self.out = nn.Linear(4096, 1)

    def forward_each(self, x):
        """
        Siamese forwards seperately each a support and a query
        """
        x = self.conv(x)
        x = self.linear(x)
        return x

    def forward(self, x1, x2):
        """
        After forward_each, computes weighted L1 distance
        """
        x1 = self.forward_each(x1)
        x2 = self.forward_each(x2)
        
        # Train weighted L1 distance
        x = torch.abs(x1, x2)
        self.out(x)
        return x