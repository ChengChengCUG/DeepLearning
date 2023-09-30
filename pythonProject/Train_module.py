import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from module import Cc

# 训练数据集的下载
train_data = torchvision.datasets.CIFAR10("./data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
# 测试数据集的下载
test_data = torchvision.datasets.CIFAR10("./data", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# 训练集和测试集的大小
train_len = len(train_data)
test_len = len(test_data)

# 利用DataLoader来加载数据集

train_data_dataloader = DataLoader(train_data, 64)
test_data_dataloader = DataLoader(test_data)

# 创建神经网络模型对象
cc = Cc()

#  记录训练和测试的次数
train_step = 0
test_step = 0

# 训练轮次
epoch = 10

# 学习速率
learning_rate = 0.001

# 创建损失函数
loss_fn = nn.CrossEntropyLoss()

# 创建tensorboard
writer = SummaryWriter("./Test11_logs")

# 创建优化器
optimizer = torch.optim.SGD(cc.parameters(), lr=learning_rate)

for i in range(epoch):
    print("-------------第{}轮训练-------------".format(i + 1))
    for data in train_data_dataloader:
        imgs, targets = data
        output = cc(imgs)
        # 求预测值与真实值之间的损失函数值
        loss = loss_fn(output, targets)
        # 优化器调优
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_step = train_step + 1
        if train_step % 100 == 0:
            print("训练次数：{} ，loss={}".format(train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), train_step)

     # 测试步骤
    sum_test_loss = 0.0

    # 整体正确率
    total_accuracy = 0
    with torch.no_grad():
        for data in test_data_dataloader:
            imgs, targets = data
            output = cc(imgs)

            # 每一次测试的正确率
            accuracy = (output.argmax(1) ==targets).sum()
            total_accuracy = accuracy + total_accuracy
            loss = loss_fn(output, targets)
            sum_test_loss = sum_test_loss + loss.item()
        # 整体测试机上的loss
    print("测试集的Loss={}".format(sum_test_loss))
    print("整体测试集上的正确率：{}".format((total_accuracy/test_len)))
    writer.add_scalar("test_loss",sum_test_loss, test_step)
    writer.add_scalar("test_accuracy",total_accuracy/test_len,test_step)
    test_step = test_step + 1
writer.close()
