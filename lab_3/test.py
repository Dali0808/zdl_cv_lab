import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ----------------------------
# 1. 定义你的模型结构
# ----------------------------
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        # Flatten data from (n, 1, 28, 28) to (n, 784)
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)  # flatten
        x = self.fc(x)
        return x


# ----------------------------
# 2. 图像预处理辅助函数
# ----------------------------
def preprocess_digit_image(img_roi):
    """
    将切分出来的单个数字图片转换为 28x28 的 MNIST 格式 tensor
    """
    # 1. 保持长宽比缩放
    h, w = img_roi.shape
    scale = 20.0 / max(h, w)  # 将最长边缩放到20像素 (留出边距)
    new_h, new_w = int(h * scale), int(w * scale)
    resized_img = cv2.resize(img_roi, (new_w, new_h))

    # 2. 创建 28x28 的黑色画布
    canvas = np.zeros((28, 28), dtype=np.uint8)

    # 3. 将缩放后的数字居中粘贴到画布上
    top = (28 - new_h) // 2
    left = (28 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized_img

    # 4. 转换为 Tensor 并归一化 (0-1)
    # 增加维度: (H, W) -> (1, H, W) -> (1, 1, H, W) batch size
    img_tensor = torch.from_numpy(canvas).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)

    return img_tensor


# ----------------------------
# 3. 主程序逻辑
# ----------------------------
def main():
    model_path = 'mnist_cnn.pth'
    image_path = 'test_id.jpg'

    # --- 加载模型 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("模型加载成功！")
    except FileNotFoundError:
        print(f"错误：找不到模型文件 {model_path}")
        return

    model.eval()  # 设置为评估模式

    # --- 读取并处理整个图片 ---
    # 读取灰度图
    src_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if src_img is None:
        print(f"错误：找不到图片文件 {image_path}")
        return

    # 二值化处理：
    # 将白纸黑字 (背景255, 字0) 转换为 黑底白字 (背景0, 字255)
    # THRESH_BINARY_INV 是关键，它会反转颜色
    _, thresh = cv2.threshold(src_img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 查找轮廓 (即每个数字的外框)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 过滤太小的噪点轮廓
    digit_rects = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 5 and h > 10:  # 简单的过滤条件，防止识别到噪点
            digit_rects.append((x, y, w, h))

    # **关键步骤**：按 X 坐标从左到右排序
    # 因为 findContours 找到的顺序是随机的，我们必须按位置排序才能得到正确的学号顺序
    digit_rects.sort(key=lambda x: x[0])

    print(f"检测到 {len(digit_rects)} 个数字。")

    result_string = ""

    # 可视化用列表
    debug_images = []

    # --- 逐个识别 ---
    with torch.no_grad():
        for i, (x, y, w, h) in enumerate(digit_rects):
            # 1. 切割出单个数字 (roi = Region of Interest)
            # 稍微留一点边框 padding
            pad = 2
            roi = thresh[max(0, y - pad):min(src_img.shape[0], y + h + pad),
                  max(0, x - pad):min(src_img.shape[1], x + w + pad)]

            # 2. 预处理成模型需要的格式 (1,1,28,28)
            input_tensor = preprocess_digit_image(roi).to(device)

            # 3. 模型推理
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()

            result_string += str(prediction)

            # 保存用于展示
            debug_images.append((roi, prediction))

    # --- 输出结果 ---
    print("-" * 30)
    print(f"最终识别结果 (学号): {result_string}")
    print("-" * 30)

    # (可选) 展示切割和识别效果
    plt.figure(figsize=(12, 2))
    for i, (img, pred) in enumerate(debug_images):
        plt.subplot(1, len(debug_images), i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(str(pred))
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()