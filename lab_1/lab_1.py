import numpy as np
import matplotlib.pyplot as plt
import cv2

K_custom = np.array([
    [1,0,-1],
    [2,0,-2],
    [1,0,-1]
],dtype=np.float64)

K_Sobel_x = np.array([
    [-1,0,1],
    [-2,0,2],
    [-1,0,1]
],dtype=np.float64)

K_Sobel_y = np.array([
    [-1,-2,-1],
    [0,0,-0],
    [1,2,1]
],dtype=np.float64)

try:
    I_rgb = np.array(cv2.imread('test.jpg'))
except:
    print("未找到文件")
    exit()

def rgb2gray(img_rgb):
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

I_gray = rgb2gray(I_rgb)
I_float = I_gray.astype(np.float64)

#手动图像滤波
def convolve2d(img, kernel):
    img = img.astype(np.float64)
    kernel = kernel.astype(np.float64)
    H,W = img.shape
    k_size = kernel.shape[0]
    padding = k_size // 2
    padded_img = np.pad(img, padding, mode='constant')
    output = np.zeros_like(img,dtype=np.float64)
    for i in range(H):
        for j in range(W):
            roi = padded_img[i:i+k_size, j:j+k_size]
            output[i,j] = np.sum(roi * kernel)
    return output
#Sobel 滤波
G_x = convolve2d(I_float, K_Sobel_x)
G_y = convolve2d(I_float, K_Sobel_y)
I_sobel = np.sqrt(G_x**2 + G_y**2)
#归一化到 0-255 范围进行可视化
I_sobel_vis = (255* (I_sobel/I_sobel.max())).astype(np.uint8)

#自定义卷积核滤波
I_custom = convolve2d(I_float, K_custom)
I_custom_vis = (255* (I_custom/np.abs(I_custom).max())).astype(np.uint8)

#颜色直方图计算
def color_histogram(img_rgb):
    img_rgb = img_rgb.astype(np.uint8)
    histograms = np.zeros((3,256),dtype=np.float64)
    H,W,C = img_rgb.shape

    for i in range(H):
        for j in range(W):
            r,g,b = img_rgb[i,j,0],img_rgb[i,j,1],img_rgb[i,j,2]
            histograms[0,r] += 1
            histograms[1,g] += 1
            histograms[2,b] += 1

    return histograms

Histograms = color_histogram(I_rgb)

# 可视化
plt.figure(figsize=(15, 4))
colors = ['red', 'green', 'blue']
titles = ['Red Channel Histogram', 'Green Channel Histogram', 'Blue Channel Histogram']

for i in range(3):
    plt.subplot(1, 3, i + 1)
    # plt.bar(x轴：0-255, y轴：对应计数)
    plt.bar(np.arange(256), Histograms[i, :], color=colors[i], width=1)
    plt.title(titles[i])
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
plt.suptitle('Color Histograms')
plt.show()

def texture_features(img_gray,N = 16, dx = 1, dy = 0):
    # 1.灰度级量化
    # 将 0-255 映射到 0 到 N-1
    img_quantized = (img_gray / 256 * N).astype(np.uint8)
    img_quantized[img_quantized == N] = N - 1  # 处理 256 的情况
    H, W = img_quantized.shape

#     2.构建GLCM矩阵
    GLCM = np.zeros((N,N), dtype=np.int64)
    for i in range(H):
        for j in range(W):
            i2 = i + dy
            j2 = j + dx
            if 0 <= i2 < H and 0 <= j2 < W:
                g1 = img_quantized[i, j]
                g2 = img_quantized[i2, j2]
                GLCM[g1, g2] += 1
                GLCM[g2, g1] += 1

#     3. 归一化GLCM
    if GLCM.sum() == 0:
        return np.array([0.0, 0.0, 0.0])

    P = GLCM / GLCM.sum()

    # 4. 特征计算
    features = {}
    i_coords, j_coords = np.meshgrid(np.arange(N), np.arange(N),indexing='ij')
    #能量
    features['Energy'] = np.sum(P**2)
    #对比度
    features['Contrast'] = np.sum((i_coords - j_coords)**2 * P)
    #相关性
    mu_x = np.sum(i_coords * P)
    mu_y = np.sum(j_coords * P)

    sigma_x = np.sqrt(np.sum((i_coords - mu_x) ** 2 * P))
    sigma_y = np.sqrt(np.sum((j_coords - mu_y) ** 2 * P))
    if sigma_x * sigma_y == 0:
        features['Correlation'] = 0.0
    else:
        features['Correlation'] = np.sum((i_coords - mu_x) * (j_coords - mu_y) * P) / (sigma_x * sigma_y)

    # 将特征值保存为 NumPy 数组
    texture_features_array = np.array([
        features['Energy'],
        features['Contrast'],
        features['Correlation']
    ])
    return texture_features_array
Texture_Features = texture_features(I_gray)
np.save('texture_features.npy', Texture_Features)
print(f"纹理特征已保存到 texture_features.npy: {Texture_Features}")

#结果输出
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(15, 5))
# 1. 原始图像
plt.subplot(1, 3, 1)
plt.imshow(I_rgb)
plt.title('1. 原始输入图像')
plt.axis('off')
# 2. Sobel 滤波结果
plt.subplot(1, 3, 2)
plt.imshow(I_sobel_vis, cmap='gray')
plt.title('2. Sobel 滤波结果')
plt.axis('off')
# 3. 自定义核滤波结果
plt.subplot(1, 3, 3)
plt.imshow(I_custom_vis, cmap='gray')
plt.title('3. 自定义核滤波结果')
plt.axis('off')
plt.show()

print(f"任务输入：您的彩色图像 ('your_image.jpg')")
print(f"任务输出 1：Sobel 滤波图像已生成并展示。")
print(f"任务输出 2：给定卷积核滤波图像已生成并展示。")
print(f"任务输出 3：彩色图像的直方图已生成并展示。")
print(f"任务输出 4：纹理特征 (Energy, Contrast, Correlation) 已保存到 texture_features.npy 文件中。")