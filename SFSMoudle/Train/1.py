import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO




# 步骤1: 数据准备
# 下载数据集（如果尚未下载）
data_dir = '/path/to/MIT_Intrinsic_Dataset'


# ...（以下与之前的代码相同）
