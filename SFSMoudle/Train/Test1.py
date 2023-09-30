import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# 步骤1: 数据准备
class SFSDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_files = sorted(
            [os.path.join(data_dir, 'images', filename) for filename in os.listdir(os.path.join(data_dir, 'images'))])
        self.normal_files = sorted(
            [os.path.join(data_dir, 'normals', filename) for filename in os.listdir(os.path.join(data_dir, 'normals'))])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert('RGB')
        normal = Image.open(self.normal_files[idx])

        if self.transform:
            image = self.transform(image)
            normal = transforms.ToTensor()(normal)

        return image, normal


data_dir = '/path/to/MIT_Intrinsic_Dataset'  # 数据集目录
transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])

dataset = SFSDataset(data_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)


# 步骤2: 构建深度估计模型
class SFSModel(nn.Module):
    def __init__(self):
        super(SFSModel, self).__init__()
        # 定义深度估计模型的架构
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()

        self.fc1 = nn.Linear(256 * 28 * 28, 512)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, 1)  # 输出深度值

    def forward(self, x):
        # 模型前向传播逻辑
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = x.view(x.size(0), -1)  # 将特征展平
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        return x


# 步骤3: 训练过程
model = SFSModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, targets in data_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(data_loader):.4f}')

print('Finished Training')

# 步骤4: 保存模型
torch.save(model.state_dict(), 'sfs_model.pth')
