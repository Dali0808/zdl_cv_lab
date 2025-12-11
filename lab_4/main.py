import cv2
from ultralytics import YOLO


def detect_campus_bikes(image_path):
    """
    检测流程主函数：输入图片 -> 特征提取 -> 定位与分类 -> 可视化输出
    """

    # ---------------------------------------------------------
    # 1. 加载模型 (对应：特征提取网络的准备)
    # ---------------------------------------------------------
    # 在实际实验中，应该加载你在 train.py 中训练好的权重
    # 路径通常在: runs/detect/campus_bike_model/weights/best.pt
    # 如果没训练完，直接用 'yolov8n.pt' 也可以，因为它本身就能识别 COCO 里的 bicycle
    model_path = 'yolov8n.pt'
    print(f"加载模型: {model_path} ...")
    model = YOLO(model_path)

    # ---------------------------------------------------------
    # 2. 执行推理 (对应：特征提取 -> 目标定位 -> 分类判断 的全过程)
    # ---------------------------------------------------------
    # model() 函数内部发生了什么？
    # a. 特征提取 (Backbone): 卷积层将图像像素转换为高维特征图。
    # b. 目标定位 (Localization): 预测边界框中心(x,y)和宽高(w,h)。
    # c. 分类判断 (Classification): 计算框内物体属于 "bicycle" 的概率。

    # classes=[1] 的意思是只检测 COCO 数据集中的第1类（bicycle/自行车）
    # 这样可以过滤掉行人、汽车等干扰，专注于共享单车
    results = model.predict(source=image_path, save=False, conf=0.25, classes=[1])

    result = results[0]  # 获取第一张图的结果
    img = cv2.imread(image_path)

    print(f"\n检测到的目标数量: {len(result.boxes)}")

    # ---------------------------------------------------------
    # 3. 解析结果与可视化 (理解定位与分类的输出)
    # ---------------------------------------------------------
    for box in result.boxes:
        # --- A. 获取定位信息 (Localization) ---
        # xyxy 格式: 左上角x, 左上角y, 右下角x, 右下角y
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        # --- B. 获取分类与置信度 (Classification) ---
        conf = float(box.conf[0])  # 置信度 (0.0 - 1.0)
        cls_id = int(box.cls[0])  # 类别ID
        label_name = result.names[cls_id]  # 类别名称，如 'bicycle'

        # --- C. 简单的逻辑判断 (区分品牌) ---
        # 注意：YOLOv8 默认只分出 "bicycle"。
        # 如果要区分 "哈啰" vs "美团"，需要在 train.py 中使用自定义数据集训练。
        # 这里模拟一个简单的逻辑：在实验中，可以假设特定颜色的车是哈啰。
        # (真实项目中需要训练细粒度分类器)
        display_text = f"{label_name}: {conf:.2f}"

        # 绘制边界框 (定位可视化)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 绘制标签 (分类可视化)
        cv2.putText(img, display_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        print(f"发现目标: {label_name} | 置信度: {conf:.2f} | 坐标: ({x1},{y1})")

    # 显示结果
    cv2.imshow('Campus Bike Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存结果图片
    output_path = "result_" + image_path
    cv2.imwrite(output_path, img)
    print(f"检测结果已保存至: {output_path}")


if __name__ == '__main__':

    img_path = 'campus_test.jpg'
