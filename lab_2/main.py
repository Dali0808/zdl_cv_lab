import cv2
import numpy as np

def canny_edge_detector(image):
    """
    步骤1-3: 图像预处理与边缘检测
    将图像转为灰度 -> 高斯模糊降噪 -> Canny边缘检测
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


def region_of_interest(image):
    """
    步骤4: 定义感兴趣区域 (ROI)
    只保留图像下半部分的道路区域，屏蔽天空和周围环境的干扰
    """
    height = image.shape[0]
    width = image.shape[1]

    # 定义一个三角形或梯形区域，这里需要根据你的图片实际拍摄角度进行微调
    # 这里的坐标是：(左下角, 右下角, 中上点)
    polygons = np.array([
        [(0, height), (width, height), (int(width / 2), int(height * 0.55))]
    ])

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def display_lines(image, lines):
    """
    辅助函数: 在黑色背景上绘制检测到的直线
    """
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            # 用绿色绘制线条，线宽为10
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    return line_image


def average_slope_intercept(image, lines):
    """
    步骤6: 车道线优化 (拟合)
    将霍夫变换检测到的多段线段，根据斜率分为左车道和右车道，
    并计算平均斜率和截距，生成两条平滑的直线。
    """
    left_fit = []
    right_fit = []

    if lines is None:
        return None

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        # 拟合一次多项式 (直线 y = mx + b)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]

        # 根据斜率区分左右车道 (左车道斜率为负，右车道斜率为正)
        # 这里的 0.5 是一个阈值，过滤掉接近水平的干扰线
        if slope < -0.5:
            left_fit.append((slope, intercept))
        elif slope > 0.5:
            right_fit.append((slope, intercept))

    # 计算平均线
    left_fit_average = np.average(left_fit, axis=0) if left_fit else None
    right_fit_average = np.average(right_fit, axis=0) if right_fit else None

    averaged_lines = []
    if left_fit_average is not None:
        averaged_lines.append(make_coordinates(image, left_fit_average))
    if right_fit_average is not None:
        averaged_lines.append(make_coordinates(image, right_fit_average))

    return np.array(averaged_lines)


def make_coordinates(image, line_parameters):
    """
    辅助函数: 根据斜率和截距计算直线的起止坐标
    """
    slope, intercept = line_parameters
    y1 = image.shape[0]  # 线条从图片底部开始
    y2 = int(y1 * 0.6)  # 线条延伸到图片高度的 3/5 处
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


# --- 主程序执行部分 ---

# 1. 读取图片
image = cv2.imread('road1.jpeg')
if image is None:
    print("错误: 找不到图片，请检查文件名是否正确。")
else:
    lane_image = np.copy(image)

    # 2. 边缘检测
    canny_image = canny_edge_detector(lane_image)

    # 3. 提取感兴趣区域 (ROI)
    cropped_image = region_of_interest(canny_image)

    # 4. 霍夫变换检测直线
    # rho=2, theta=np.pi/180 (1度), threshold=100 (投票阈值)
    # minLineLength=40 (最小线长), maxLineGap=5 (最大断裂间隔)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)

    # 5. 优化拟合车道线 (得到单条平滑直线)
    averaged_lines = average_slope_intercept(lane_image, lines)

    # 6. 绘制结果
    line_image = display_lines(lane_image, averaged_lines)

    # 7. 将车道线叠加回原图
    combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)

    # 8. 显示结果
    cv2.imshow("Original", image)
    cv2.imshow("Canny Edges", canny_image)  # 显示边缘检测过程图，用于报告分析
    cv2.imshow("Result", combo_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存结果图片用于提交
    cv2.imwrite('result_lane_detection.jpg', combo_image)