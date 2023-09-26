import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# 数据集
dataset = torchvision.datasets.CIFAR10("./data", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
#将数据集按批次加载
dataloader = DataLoader(dataset, batch_size=64)

# 卷积神经网络
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        # 卷积核的大小是3*3
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


writer = SummaryWriter("./logs")
# 实例化卷积网络
tudui = Tudui()
print(tudui)
step = 0
for data in dataloader:
    imgs, targets = data
    # 将张量数据输入卷积网络
    output = tudui(imgs)
    writer.add_images("input_data", imgs, step)

    output=torch.reshape(output,[-1,3,30,30])
    writer.add_images("output_data", output, step)
    step=step+1
writer.close()