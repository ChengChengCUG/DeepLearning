import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# input = torch.tensor([[1,2,0,3,1],
#                                       [0,1,2,3,1],
#                                       [1,2,1,0,0],
#                                       [5,2,3,1,1],
#                                       [2,1,0,1,1]],dtype=float)
# input = torch.reshape(input,(-1,1,5,5))
# print(input.shape)
dataset = torchvision.datasets.CIFAR10("./data",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,batch_size=64)
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui,self).__init__()
        self.maxpool = MaxPool2d(kernel_size=3,ceil_mode=True)

    def forward(self,input):
        output = self.maxpool(input)
        return output
writer = SummaryWriter("logs_maxpool")
tuidui= Tudui()
step = 0
for data in dataloader:
    imgs,targets = data
    output = tuidui(imgs)
    writer.add_images("input",imgs,step)
    writer.add_images("output",output,step)
    step = step+1
# output = tuidui(input)
# print(output)
writer.close()
