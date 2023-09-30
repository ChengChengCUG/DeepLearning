import torch
from torch import nn


class Cc(nn.Module):
    def __init__(self):
        super(Cc,self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(3,32,5,stride=1,padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,kernel_size=5,stride=1,padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,10)
        )
    def forward(self,x):
        x = self.module(x)
        return x
