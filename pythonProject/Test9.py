import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取输入图像
image_path = "my_img.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 转为灰度图像

# 定义光照方向，这里使用示例值
tilt_degrees = 50
slant_degrees = 30

# 将角度转换为弧度
tilt = np.deg2rad(tilt_degrees)
slant = np.deg2rad(slant_degrees)

# 获取图像尺寸
h, w = img.shape

# 计算光照方向的单位向量
ix = np.cos(tilt) * np.tan(slant)
iy = np.sin(tilt) * np.tan(slant)

# 初始化深度图和法线图
depth_map = np.zeros_like(img, dtype=np.float32)
normal_map = np.zeros((h, w, 3), dtype=np.float32)

# 最大迭代次数
max_iterations = 100

# Jacobi迭代
for _ in range(max_iterations):
    # 计算法线图
    gradient_x, gradient_y = np.gradient(depth_map)
    normal_map[:, :, 0] = -gradient_x
    normal_map[:, :, 1] = -gradient_y
    normal_map[:, :, 2] = 1.0 / np.sqrt(1 + gradient_x**2 + gradient_y**2)

    # 计算反射率
    cos_theta = np.maximum(0, np.sum(normal_map * np.array([ix, iy, 1]), axis=2))
    reflectance_map = cos_theta / np.sqrt(1 + gradient_x**2 + gradient_y**2)

    # 更新深度图
    depth_map -= (img - reflectance_map) / (1 + gradient_x**2 + gradient_y**2)

# 绘制深度图
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(depth_map, cmap='jet')
plt.title("Depth Map")
plt.colorbar()

# 绘制三维表面
plt.subplot(122)
xx, yy = np.meshgrid(range(w), range(h))
zz = depth_map
plt.gca().invert_yaxis()
ax = plt.gcf().add_subplot(122, projection='3d')
ax.plot_surface(xx, yy, zz, cmap='viridis')
plt.title("3D Surface")
plt.show()
