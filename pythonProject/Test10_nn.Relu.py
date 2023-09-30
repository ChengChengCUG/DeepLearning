import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./Test10_data", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64)
torch.reshape()

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output


writer = SummaryWriter("Test10_logs")
tudui = Tudui()
step = 0
for data in dataloader:
    imgs, targets = data
    output = tudui(imgs)
    writer.add_images("input", imgs, step)
    writer.add_images("output", output, step)
    step = step + 1
writer.close()
