import torchvision
from torch.utils.tensorboard import SummaryWriter

transform_set = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
# 这行代码的主要目的是加载CIFAR-10数据集，
# 经过转换操作后将数据集中的图像变换为张量（tensor）类型
# 然后将其保存在指定的root目录下的"data"文件夹中
train_set = torchvision.datasets.CIFAR10("./data",True,transform_set,download=True)

test_set = torchvision.datasets.CIFAR10("./data",False,transform_set,download=True)
# test_set[0] 返回一个包含两个值的元组或列表，其中第一个值是图像，第二个值是目标标签。
# img,target = test_set[0]

writer = SummaryWriter("Test5_data")
for i in range(10):
    img,target = train_set[i]
    writer.add_image("test_set",img,i)
writer.close()
