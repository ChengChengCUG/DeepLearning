from PIL import Image
from torchvision import transforms

img_path = "dataset/hymenoptera_data/train/ants/0013035.jpg"
img = Image.open(img_path)
tensor_trans = transforms.ToTensor()
tensor_imag = tensor_trans(img)
print(tensor_imag)