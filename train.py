import sys
import getopt

import albumentations as A
import albumentations.pytorch
import torch
from torch.utils.data import DataLoader

from tools.utils import get_bound_boxes
from tools.mAP import mean_average_precision
from tools.fit import fit
from model.dataset import Dataset
from model.model import YOLOv1
from model.loss import YoloLoss

# 默认参数配置
data_train = './data_example/train'  # 训练数据目录
data_test = './data_example/test'  # 测试数据目录
data_label = './data_example/classes.names'  # 类别标签文件
extraction_weights_path = None  # 特征提取网络权重路径
yolo_weights_path = None  # YOLO完整模型权重路径
file_format = 'txt'  # 标注文件格式
width = 448  # 图像宽度
height = 448  # 图像高度
batch_size = 8  # 批量大小
epochs = 100  # 训练轮数
classes = 4  # 类别数量
learning_rate = 5e-5  # 学习率
weight_decay = 0.0005  # 权重衰减
convert_to_yolo = False  # 是否转换标注格式为YOLO格式
save = False  # 是否保存训练好的模型

# 可配置参数列表
config_list = ['data_train=',
               'data_test=',
               'data_label=',
               'extraction_weights_path=',
               'yolo_weights_path=',
               'file_format=',
               'width=',
               'height=',
               'batch_size=',
               'epochs=',
               'classes=',
               'learning_rate=',
               'weight_decay=',
               'convert_to_yolo',
               'save']

# 解析命令行参数
try:
    options, args = getopt.getopt(sys.argv[1:], '', config_list)
    for opt, arg in options:
        if opt in ['--data_train']:
            data_train = arg  # 设置训练数据路径
        if opt in ['--data_test']:
            data_test = arg  # 设置测试数据路径
        if opt in ['--data_label']:
            data_label = arg  # 设置类别标签文件路径
        if opt in ['--extraction_weights_path']:
            extraction_weights_path = arg  # 设置特征提取网络权重路径
        if opt in ['--yolo_weights_path']:
            yolo_weights_path = arg  # 设置YOLO模型权重路径
        if opt in ['--file_format']:
            file_format = arg  # 设置标注文件格式
        if opt in ['--width']:
            width = int(arg)  # 设置图像宽度
        if opt in ['--height']:
            height = int(arg)  # 设置图像高度
        if opt in ['--batch_size']:
            batch_size = int(arg)  # 设置批量大小
        if opt in ['--epochs']:
            epochs = int(arg)  # 设置训练轮数
        if opt in ['--classes']:
            classes = int(arg)  # 设置类别数量
        if opt in ['--learning_rate']:
            learning_rate = float(arg)  # 设置学习率
        if opt in ['--weight_decay']:
            weight_decay = float(arg)  # 设置权重衰减
        if opt in ['--convert_to_yolo']:
            convert_to_yolo = True  # 启用标注格式转换
        if opt in ['--save']:
            save = True  # 启用模型保存
except getopt.GetoptError:
    print("配置错误，请检查参数和值")
    sys.exit(-1)

# 训练数据增强变换
train_transform = A.Compose(
    [
        A.Resize(width, height),  # 调整图像大小
        A.HorizontalFlip(p=0.5),  # 50%概率水平翻转
        A.Normalize(),  # 归一化
        A.pytorch.ToTensorV2()  # 转换为张量
    ],
    bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])  # 边界框参数(YOLO格式)
)

# 测试数据变换(不包含数据增强)
test_transform = A.Compose(
    [
        A.Resize(width, height),
        A.Normalize(),
        A.pytorch.ToTensorV2()
    ],
    bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
)

# 创建训练数据集
train_dataset = Dataset(
    transforms=train_transform,
    data_dir=data_train,
    labels_dir=data_label,
    S=7,  # 网格大小
    C=classes,  # 类别数
    file_format=file_format,
    convert_to_yolo=convert_to_yolo  # 是否转换标注格式
)

# 创建验证数据集
val_dataset = Dataset(
    transforms=test_transform,
    data_dir=data_test,
    labels_dir=data_label,
    S=7,
    C=classes,
    file_format=file_format,
    convert_to_yolo=convert_to_yolo
)

# 数据集完整性检查
assert isinstance(train_dataset[0], dict)  # 检查样本是否为字典格式
assert len(train_dataset[0]) == 2  # 检查是否包含图像和目标
assert isinstance(train_dataset[0]['image'], torch.Tensor)  # 检查图像是否为张量
assert isinstance(train_dataset[0]['target'], torch.Tensor)  # 检查目标是否为张量
print('所有测试通过')

# 创建训练数据加载器
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True)  # 训练时打乱数据

# 创建验证数据加载器
val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=False)  # 验证时不打乱数据

# 设置设备(优先使用GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化YOLOv1模型
if extraction_weights_path is not None:
    # 加载预训练的特征提取网络权重
    model = YOLOv1(S=7, B=2, C=classes, device=device, extraction_weights_path=extraction_weights_path).to(device)
else:
    # 从头开始训练
    model = YOLOv1(S=7, B=2, C=classes, device=device).to(device)

# 加载完整的YOLO模型权重(如果提供)
if yolo_weights_path is not None:
    model.load_state_dict(torch.load(yolo_weights_path))

# 初始化损失函数、优化器和学习率调度器
Yolo_loss = YoloLoss(S=7, B=2, C=classes).to(device)  # YOLO自定义损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # Adam优化器
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)  # 指数衰减学习率

# 训练模型
fit(model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=Yolo_loss,
    epochs=epochs,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    device=device)

# 计算训练集mAP(平均精度均值)
pred_boxes, true_boxes = get_bound_boxes(train_dataloader, model, iou_threshold=0.5, threshold=0.15, device=device)
mAP = mean_average_precision(pred_boxes, true_boxes, classes=classes, iou_threshold=0.5)
print(f'训练集mAP: {mAP}\n')

# 计算验证集mAP
pred_boxes, true_boxes = get_bound_boxes(val_dataloader, model, iou_threshold=0.5, threshold=0.15, device=device)
mAP = mean_average_precision(pred_boxes, true_boxes, classes=classes, iou_threshold=0.5)
print(f'验证集mAP: {mAP}\n')

# 保存训练好的模型
if save:
    torch.save(model.state_dict(), './yolov1_' + str(epochs) + '.pt')