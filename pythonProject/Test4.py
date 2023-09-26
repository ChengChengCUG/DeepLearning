from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter('logs')
img = Image.open("dataset/hymenoptera_data/train/bees/39747887_42df2855ee.jpg")
print(img)
# ToTensor
trans_totensor= transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor,0)
print(img_tensor[0][0][0])

# Normalize
tensor_normal = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_normal = tensor_normal(img_tensor)
print(img_normal[0][0][0])
writer.add_image("Normalizer", img_normal,1)

# Resize
print(img.size)
tensor_resize = transforms.Resize((512,512))
img_resize = tensor_resize(img) # 此时返回的是img PIL类型的数据，赋值给变量img_resize
img_resize = trans_totensor(img_resize) # 这是重新赋值,此时返回的是tensor类型的数据
print(img_resize)
writer.add_image("Resize", img_resize,2)
writer.close()


