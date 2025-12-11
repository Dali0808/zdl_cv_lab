import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from train import SimpleCNN
import torch.nn as nn



# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
try:
    # 确保 mnist_cnn.pth 文件在同一目录下
    model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
    print("模型加载成功！")
except FileNotFoundError:
    print("错误：找不到 mnist_cnn.pth，请先运行训练代码。")
    exit()
model.eval()

# 定义预处理转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def predict_student_id_advanced(image_path):
    # 1. 读取图片并转为灰度
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误：无法读取图片 {image_path}")
        return
    # 适当缩小图片，加快处理速度，有时也能减少噪点
    # img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ================== 核心改进区域 ==================

    # 2.【改进】自适应二值化 (Adaptive Thresholding)
    # ADAPTIVE_THRESH_GAUSSIAN_C 比 mean 方法更好
    # blockSize (11): 邻域大小，必须是奇数。越小越关注局部细节，越大越像全局阈值。
    # C (2): 从计算出的平均值中减去的常数。用于微调。
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # 3.【改进】形态学去噪 (Morphological Operations)
    # 定义一个核结构
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # 开运算：先腐蚀后膨胀，用于去除小的白噪声点
    thresh_cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # 如果笔画断裂严重，取消下面这行的注释尝试“闭运算”
    # thresh_cleaned = cv2.morphologyEx(thresh_cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    # =================================================

    # 可视化调试：查看二值化结果 (非常关键!)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("1. Adaptive Thresholding (Black background?)")
    plt.imshow(thresh, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title("2. Cleaned (Noise Removed?)")
    plt.imshow(thresh_cleaned, cmap='gray')
    plt.axis('off')
    plt.show()

    # 4. 查找轮廓
    contours, _ = cv2.findContours(thresh_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digit_rects = []
    img_h, img_w = gray.shape

    # 【调试】在原图上画出所有找到的轮廓，看看是否准确
    debug_img = img.copy()

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # 【改进】更智能的筛选条件
        # 1. 高度不能太小 (例如小于图片高度的 1/50)
        # 2. 高度不能太大 (例如超过图片高度的 1/2，那是背景噪音)
        # 3. 长宽比不能太夸张 (防止检测到长条形的污渍)
        aspect_ratio = w / float(h)
        if (h > img_h * 0.02) and (h < img_h * 0.8) and (0.2 < aspect_ratio < 3):
            digit_rects.append((x, y, w, h))
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 画绿框
        else:
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 1)  # 被过滤掉的画红框

    # 可视化调试：查看轮廓筛选结果
    plt.figure(figsize=(10, 5))
    plt.title("3. Detected Contours (Green=Kept, Red=Filtered)")
    plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # 5. 排序
    digit_rects.sort(key=lambda x: x[0])

    result_string = ""
    if len(digit_rects) == 0:
        print("未检测到任何数字，请检查二值化结果图。")
        return ""

    # 用于展示最终送入模型的图片
    plt.figure(figsize=(len(digit_rects), 2))

    for i, (x, y, w, h) in enumerate(digit_rects):
        # 提取 ROI (在清理后的二值图上提取)
        # 增加 padding，确保不切断笔画
        pad_h = int(h * 0.15)
        pad_w = int(w * 0.15)
        roi = thresh_cleaned[max(0, y - pad_h):min(img_h, y + h + pad_h),
              max(0, x - pad_w):min(img_w, x + w + pad_w)]

        # 标准化至 28x28 (保持长宽比)
        h_roi, w_roi = roi.shape
        # 确保 ROI 有效
        if h_roi == 0 or w_roi == 0: continue

        # 缩放到 20x20 区域内
        scale = 20.0 / max(h_roi, w_roi)
        new_w, new_h = int(w_roi * scale), int(h_roi * scale)
        # 确保 resize 尺寸有效
        if new_w <= 0 or new_h <= 0: new_w, new_h = 1, 1

        resized_roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((28, 28), dtype=np.uint8)
        start_x = (28 - new_w) // 2
        start_y = (28 - new_h) // 2
        canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized_roi

        # 6. 预测
        pil_img = Image.fromarray(canvas)
        tensor_img = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor_img)
            prediction = output.argmax(dim=1).item()
            result_string += str(prediction)

        # 可视化最终输入模型的小图
        plt.subplot(1, len(digit_rects), i + 1)
        plt.imshow(canvas, cmap='gray')
        plt.axis('off')
        plt.title(str(prediction))

    plt.show()
    return result_string


if __name__ == '__main__':
    # 请准备一张清晰的学号照片，背景尽量干净，光照均匀
    image_path = "test_id.jpg"
    # 如果没有图片，创建一个假的用于测试 (取消下面注释)
    # create_dummy_image(image_path)

    print(f"正在处理图片: {image_path} ...")
    predicted_id = predict_student_id_advanced(image_path)
    print(f"\nFinal Result --- 识别到的学号为: {predicted_id}")