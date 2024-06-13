'''import cv2
import matplotlib.pyplot as plt
import numpy as np

original = cv2.imread(r"D:\\PAT_167_258_479.png")  # 转为灰度图

dft = cv2.dft(np.float32(original), flags=cv2.DFT_COMPLEX_OUTPUT)
dftShift = np.fft.fftshift(dft)     #  将图像中的低频部分移动到图像的中心
result = 20 * np.log(cv2.magnitude(dftShift[:, :, 0], dftShift[:, :, 1]))   # 将实部和虚部转换为实部，乘以20是为了使得结果更大

plt.subplot(121), plt.imshow(original, cmap='gray')
plt.title('original')
plt.axis('off')

plt.subplot(122), plt.imshow(result)
plt.title('fft')
plt.axis('off')
plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取彩色图像
image = cv2.imread("D:\\PAT_167_258_479.png")

# 将彩色图像转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用傅里叶变换
f_transform = np.fft.fft2(gray_image)
f_shift = np.fft.fftshift(f_transform)  # 将FFT输出的直流分量移到频谱中心

# 取模并归一化
magnitude_spectrum = 20*np.log(np.abs(f_shift))

# 显示原始图像
plt.subplot(122), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# 可视化频域图像
plt.subplot(121), plt.imshow(magnitude_spectrum, cmap='jet')
plt.title('Magnitude Spectrum')
plt.axis('off')



plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取彩色图像
image = cv2.imread("D:\\PAT_167_258_479.png")

# 将彩色图像转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用傅里叶变换
f_transform = np.fft.fft2(gray_image.astype(np.float32))
f_shift = np.fft.fftshift(f_transform)

# 取模并进行对数变换以可视化频谱图
magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)  # 避免对数为负无穷

# 可视化原始图像和频谱图
plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image'), plt.axis('off')

plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='hot')
plt.title('Magnitude Spectrum'), plt.colorbar(), plt.axis('off')
plt.show()

# 从频谱图中提取中心的低频分量
center = int(np.sqrt(f_shift.shape[0] * f_shift.shape[1] / 2))
f_shift_center = f_shift[center:center+1, center:center+1]

# 将提取的低频分量重构回空间域
f_ishift_center = np.fft.ifftshift(f_shift_center)
f_itransform_center = np.fft.ifft2(f_ishift_center)

# 取实部并转换为uint8类型
reconstructed_image = np.abs(f_itransform_center).astype(np.uint8)

# 可视化重构的低频分量图像
plt.figure(figsize=(5, 5))
plt.imshow(reconstructed_image, cmap='gray')
plt.title('Reconstructed Low Frequency Image'), plt.axis('off')
plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread("D:\\PAT_167_258_479.png")

# 转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用傅里叶变换
f_transform = np.fft.fft2(gray_image)
f_shift = np.fft.fftshift(f_transform)

# 可视化原始图像
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.show()

# 可视化频谱图
plt.imshow(20 * np.log(np.abs(f_shift) + 1), cmap='gray')
plt.title('Log Magnitude Spectrum')
plt.colorbar()
plt.show()

# 定义低频区域的边界
low_freq_mask = np.zeros(f_shift.shape, dtype=np.uint8)
low_freq_mask[25:-25, 25:-25] = 1  # 选择中间区域作为低频分量

# 应用低频掩码
low_freq_f_shift = f_shift * low_freq_mask

# 逆傅里叶变换
low_freq_f_ishift = np.fft.ifftshift(low_freq_f_shift)
low_freq_f_itransform = np.fft.ifft2(low_freq_f_ishift)

# 取实部并转换为8位无符号整数
reconstructed_image = np.abs(low_freq_f_itransform).astype(np.uint8)

# 可视化重构的低频分量图像
plt.imshow(reconstructed_image, cmap='gray')
plt.title('Reconstructed Low Frequency Image')
plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('D:\\PAT_167_258_479.png', cv2.IMREAD_GRAYSCALE)

# 应用傅里叶变换
f_transform = np.fft.fft2(image)
f_shift = np.fft.fftshift(f_transform)

# 可视化频谱图
plt.imshow(np.log1p(np.abs(f_shift)), cmap='jet')
plt.title('Log Magnitude Spectrum')
plt.colorbar()
plt.show()

# 定义低频区域的掩码
rows, cols = image.shape
mask = np.zeros((rows, cols), np.uint8)
mask[10:10, 10:10] = 255  # 定义中心区域

# 应用掩码到频域数据
f_shift_masked = f_shift * np.fft.fftshift(mask)

# 逆傅里叶变换
#f_ishift_masked = np.fft.ifftshift(f_shift_masked)
reconstructed = np.fft.ifft2(f_shift_masked)

# 取实部并确保像素值在0-255范围内
reconstructed = np.abs(reconstructed).astype(np.uint8)

# 可视化重构的低频分量图像
plt.imshow(reconstructed, cmap='jet')
plt.title('Reconstructed Low Frequency Image')
plt.colorbar()
plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('D:\\PAT_167_258_479.png')

# 如果图像是彩色的，转换为灰度图像
if len(image.shape) == 3:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
else:
    gray_image = image

# 应用傅里叶变换
f_transform = np.fft.fft2(gray_image.astype(np.float32))
f_shift = np.fft.fftshift(f_transform)

# 可视化原始图像
plt.figure(figsize=(10, 6))
plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cmap='gray')
plt.title('Original Image'), plt.axis('off')

# 可视化频谱图
plt.subplot(122), plt.imshow(np.log(np.abs(f_shift) + 1), cmap='gray')
plt.title('Frequency Spectrum'), plt.axis('off')
plt.show()

# 创建掩码以选择频域的一部分
height, width = gray_image.shape
mask = np.zeros((height, width), dtype=np.float32)

# 定义掩码中心区域的大小
mask_size = 30  # 可以根据需要调整
mask[height//2 - mask_size:height//2 + mask_size, 
     width//2 - mask_size:width//2 + mask_size] = 1

# 应用掩码到频域数据
f_shift_masked = f_shift * mask

# 逆傅里叶变换
f_ishift_masked = np.fft.ifftshift(f_shift_masked)
reconstructed = np.fft.ifft2(f_ishift_masked)

# 取实部，确保像素值在0到255的范围内
reconstructed = np.clip(np.real(reconstructed), 0, 255).astype(np.uint8)

# 可视化重构的图像
plt.figure(figsize=(5, 5))
plt.imshow(reconstructed, cmap='gray')
plt.title('Reconstructed Image from Frequency Domain Selection'), plt.axis('off')
plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 图像路径
image_path = 'D:\\PAT_167_258_479.png'

# 读取彩色图像
image = cv2.imread(image_path)

# 检查图像是否正确读取
if image is None:
    print("Image not found. Please check the file path.")
else:
    # 分离颜色通道
    b_channel, g_channel, r_channel = cv2.split(image)

    # 对每个通道进行傅里叶变换和频域中心化
    f_b = np.fft.fft2(b_channel.astype(np.float32))
    f_b_shift = np.fft.fftshift(f_b)

    f_g = np.fft.fft2(g_channel.astype(np.float32))
    f_g_shift = np.fft.fftshift(f_g)

    f_r = np.fft.fft2(r_channel.astype(np.float32))
    f_r_shift = np.fft.fftshift(f_r)

    # 创建掩码以选择频域的低频部分
    mask_size = 10  # 掩码大小可以根据需要调整
    height, width = f_b_shift.shape
    center_y, center_x = height // 2, width // 2  # 确保是整数

    # 定义掩码
    mask = np.zeros(f_b_shift.shape, dtype=np.float32)
    start_y = max(center_y - mask_size, 0)
    end_y = min(center_y + mask_size + 1, height)
    start_x = max(center_x - mask_size, 0)
    end_x = min(center_x + mask_size + 1, width)
    mask[start_y:end_y, start_x:end_x] = 1

    # 应用掩码到每个通道的频域数据
    f_b_shift_masked = f_b_shift * mask
    f_g_shift_masked = f_g_shift * mask
    f_r_shift_masked = f_r_shift * mask

    # 对掩码后的频域数据进行逆傅里叶变换
    f_b_ishift_masked = np.fft.ifftshift(f_b_shift_masked)
    f_g_ishift_masked = np.fft.ifftshift(f_g_shift_masked)
    f_r_ishift_masked = np.fft.ifftshift(f_r_shift_masked)

    b_reconstructed = np.fft.ifft2(f_b_ishift_masked).real.astype(np.uint8)
    g_reconstructed = np.fft.ifft2(f_g_ishift_masked).real.astype(np.uint8)
    r_reconstructed = np.fft.ifft2(f_r_ishift_masked).real.astype(np.uint8)

    # 合并通道得到重构的彩色图像
    reconstructed_image = cv2.merge((b_reconstructed, g_reconstructed, r_reconstructed))

    # 可视化原始图像和重构后的图像
    plt.figure(figsize=(12, 6))
    plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(122), plt.imshow(cv2.cvtColor(reconstructed_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 图像路径
image_path = 'D:\\PAT_167_258_479.png'

# 读取彩色图像
image = cv2.imread(image_path)

# 检查图像是否正确读取
if image is None:
    print("Image not found. Please check the file path.")
else:
    # 分离颜色通道
    b_channel, g_channel, r_channel = cv2.split(image)

    # 对每个通道进行傅里叶变换和频域中心化
    f_b = np.fft.fft2(b_channel.astype(np.float32))
    f_b_shift = np.fft.fftshift(f_b)

    f_g = np.fft.fft2(g_channel.astype(np.float32))
    f_g_shift = np.fft.fftshift(f_g)

    f_r = np.fft.fft2(r_channel.astype(np.float32))
    f_r_shift = np.fft.fftshift(f_r)

    # 可视化原始图像
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    plt.axis('off')

    # 可视化频谱图像
    plt.subplot(1, 3, 2)
    plt.imshow(np.log1p(np.abs(f_b_shift) + np.abs(f_g_shift) + np.abs(f_r_shift)), cmap='copper')

    plt.axis('off')

    # 创建掩码以选择频域的低频部分
    mask_size = 10  # 掩码大小可以根据需要调整
    height, width = f_b_shift.shape
    center_y, center_x = height // 2, width // 2

    mask = np.zeros(f_b_shift.shape, dtype=np.float32)
    mask[center_y - mask_size:center_y + mask_size + 1,
         center_x - mask_size:center_x + mask_size + 1] = 1

    # 应用掩码到每个通道的频域数据
    f_b_shift_masked = f_b_shift * mask
    f_g_shift_masked = f_g_shift * mask
    f_r_shift_masked = f_r_shift * mask

    # 对掩码后的频域数据进行逆傅里叶变换
    f_b_ishift_masked = np.fft.ifftshift(f_b_shift_masked)
    f_g_ishift_masked = np.fft.ifftshift(f_g_shift_masked)
    f_r_ishift_masked = np.fft.ifftshift(f_r_shift_masked)

    b_reconstructed = np.fft.ifft2(f_b_ishift_masked).real.astype(np.uint8)
    g_reconstructed = np.fft.ifft2(f_g_ishift_masked).real.astype(np.uint8)
    r_reconstructed = np.fft.ifft2(f_r_ishift_masked).real.astype(np.uint8)

    # 合并通道得到重构的彩色图像
    reconstructed_image = cv2.merge((b_reconstructed, g_reconstructed, r_reconstructed))

    # 可视化重构图像
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(reconstructed_image, cv2.COLOR_BGR2RGB))

    plt.axis('off')

    plt.tight_layout()
    plt.show()