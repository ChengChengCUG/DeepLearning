import torch
import torchvision.datasets
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from torch import nn

dataset = torchvision.datasets.CIFAR10("./data", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=1)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.moderl1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10))

    def forward(self, x):  # 前向传播
        x = self.moderl1(x)
        return x

loss = nn.CrossEntropyLoss()
tudui = Tudui()
optim = torch.optim.SGD(tudui.parameters(),lr=0.01)
for epoch in range(100):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        # 求预测值与真实值之间的损失函数值
        optim.zero_grad()
        results_loss = loss(outputs, targets)
        # 根据损失函数的值进行反向传播
        results_loss.backward()  # 反向传播是用来计算损失函数的梯度的，根据梯度来优化参数
        optim.step()
        # print(results_loss)
        running_loss = running_loss+results_loss
    print(running_loss)

