from ultralytics import YOLO

def train_model():
    """
    模型训练流程：特征提取网络的构建与权重更新
    """
    # 1. 加载模型
    # yolov8n.pt 是预训练模型，已经在大规模 COCO 数据集上训练过，具备了基础的特征提取能力
    # 这里对应：【特征提取】的初始化
    print("正在加载预训练模型...")
    model = YOLO('yolov8n.pt')

    # 2. 开始训练
    # data: 数据集配置文件路径。
    #       如果你有自己标注的校园单车数据，请创建一个 .yaml 文件指向它。
    #       为了演示，这里使用 'coco8.yaml' (会自动下载)，它包含了 bicycle 类别。
    # epochs: 训练轮数。
    # imgsz: 输入图像的大小。
    print("开始训练...")
    results = model.train(
        data='coco8.yaml',   # 建议替换为你自己标注的 'campus_bike.yaml'
        epochs=10,           # 实验演示设为10，实际训练建议 50-100
        imgsz=640,           # 图像尺寸缩放
        batch=16,            # 批次大小
        name='campus_bike_model' # 训练结果保存的文件夹名称
    )

    # 3. 评估模型 在验证集上测试模型的定位和分类性能
    metrics = model.val()
    print(f"训练完成，mAP50-95: {metrics.box.map}")

    path = model.export(format='onnx')
    print(f"模型已导出至: {path}")

if __name__ == '__main__':
    train_model()