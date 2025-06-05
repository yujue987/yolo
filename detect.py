import sys
import getopt
import time

import cv2
import torch
import albumentations as A
import albumentations.pytorch
import random

from model.model import YOLOv1
from tools.utils import non_max_supression
from tools.utils import convert_to_yolo

# 默认参数配置
data_test = './310.jpg'  # 默认测试图像路径
data_label = './data/classes.names'  # 默认类别标签文件
iou_threshold = 0.5  # 非极大值抑制的IoU阈值
threshold = 0.15  # 置信度阈值
width = 448  # 输入图像宽度
height = 448  # 输入图像高度
classes = 4  # 默认类别数量
yolo_weights_path = './weights/yolov1.pt'  # 默认权重文件路径
show = False  # 默认不显示检测结果
image = True  # 默认处理图像（非视频）
out = None  # 默认不输出结果文件

# 可配置参数列表
config_list = ['data_test=',
               'data_label=',
               'classes=',
               'yolo_weights_path=',
               'show',
               'video',
               'output=']

# 解析命令行参数
try:
    options, args = getopt.getopt(sys.argv[1:], '', config_list)
    for opt, arg in options:
        if opt in ['--data_test']:
            data_test = arg  # 设置测试数据路径
        if opt in ['--data_label']:
            data_label = arg  # 设置类别标签文件路径
        if opt in ['--classes']:
            classes = int(arg)  # 设置类别数量
        if opt in ['--yolo_weights_path']:
            yolo_weights_path = arg  # 设置权重文件路径
        if opt in ['--show']:
            show = True  # 启用结果显示
        if opt in ['--video']:
            image = False  # 设置为视频处理模式
        if opt in ['--output']:
            out = arg  # 设置输出文件路径
except getopt.GetoptError:
    print("配置错误，请检查参数和值")
    sys.exit(-1)

# 定义图像预处理变换
transform = A.Compose(
    [
        A.Resize(width, height),  # 调整图像大小
        A.Normalize(),  # 归一化处理
        A.pytorch.ToTensorV2()  # 转换为PyTorch张量
    ])

# 设置设备（默认CPU）
device = torch.device('cpu')
print("使用CPU进行推理")

# 初始化YOLOv1模型
model = YOLOv1(S=7, B=2, C=classes, device=device).to(device)

# 加载预训练权重
if yolo_weights_path is not None:
    model.load_state_dict(torch.load(yolo_weights_path, map_location=device))

# 打开视频文件或图像
cap = cv2.VideoCapture(data_test)
# 获取原始视频/图像的宽高
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 如果指定了输出路径且是视频模式，初始化视频写入器
if out and not image:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4V编码
    out_video = cv2.VideoWriter(out, fourcc, 25.0, (width, height))

# FPS计算相关变量
frame_counter = 0
start = time.time()

# 主处理循环
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # 图像预处理
        transformed = transform(image=frame)
        transformed_image = transformed["image"]
        transformed_image = transformed_image.unsqueeze(dim=0)  # 增加batch维度
        transformed_image = transformed_image.to(device)

        # 模型推理
        model.eval()  # 设置为评估模式
        with torch.no_grad():  # 禁用梯度计算
            predictions = model(transformed_image)  # 获取预测结果

        # 处理预测框
        pred_boxes = []
        S = predictions.size(1)  # 网格大小(S=7)
        for y in range(0, S):
            for x in range(0, S):
                # 处理第一个预测框
                pred_box = torch.empty((5 + classes), dtype=torch.float32)
                pred_box[:5] = predictions[0, y, x, :5]  # 置信度+坐标
                pred_box[5:] = predictions[0, y, x, 10:]  # 类别概率
                pred_box[1:5] = convert_to_yolo(pred_box[1:5], x, y, S)  # 转换坐标格式
                pred_boxes.append(pred_box)

                # 处理第二个预测框
                pred_box = torch.empty((5 + classes), dtype=torch.float32)
                pred_box = predictions[0, y, x, 5:]  # 第二个框的数据
                pred_box[1:5] = convert_to_yolo(pred_box[1:5], x, y, S)
                pred_boxes.append(pred_box)

        # 应用非极大值抑制
        pred_boxes = non_max_supression(pred_boxes, iou_threshold, threshold)

        # 加载类别标签和颜色
        labels = [[str, tuple] for i in range(classes)]
        colors = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (100, 255, 40)]  # 默认颜色(BGR格式)
        with open(data_label, 'r') as f:
            for line in f:
                (val, key) = line.split()  # 解析标签文件
                labels[int(val)][0] = key  # 类别名称

                # 为每个类别分配颜色
                if int(val) < len(colors):
                    labels[int(val)][1] = colors[int(val)]
                else:
                    # 如果类别数多于预设颜色，随机生成颜色
                    labels[int(val)][1] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # 在图像上绘制检测结果
        height, width, _ = frame.shape
        for box in pred_boxes:
            conf = box[0].item()  # 置信度
            # 计算边界框坐标
            x1 = int(box[1] * width - box[3] * width / 2)
            y1 = int(box[2] * height - box[4] * height / 2)
            x2 = int(box[1] * width + box[3] * width / 2)
            y2 = int(box[2] * height + box[4] * height / 2)
            choose_class = torch.argmax(box[5:])  # 选择概率最高的类别

            # 绘制边界框
            line_thickness = 2
            text = labels[choose_class][0] + ' ' + str(round(conf, 2))  # 标签文本
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=labels[choose_class][1], thickness=line_thickness)

            # 绘制标签背景
            size, baseline = cv2.getTextSize(text, cv2.FONT_ITALIC, fontScale=0.5, thickness=1)
            text_w, text_h = size
            cv2.rectangle(frame, (x1, y1), (x1 + text_w + line_thickness, y1 + text_h + baseline),
                          color=labels[choose_class][1], thickness=-1)  # 填充矩形

            # 绘制标签文本
            cv2.putText(frame, text, (x1 + line_thickness, y1 + 2 * baseline + line_thickness), cv2.FONT_ITALIC,
                        fontScale=0.5, color=(0, 0, 0), thickness=1, lineType=9)

        # 显示结果
        if show:
            cv2.imshow('Detect', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):  # 按q键退出
                break

        # 保存结果
        if out:
            if image:
                cv2.imwrite(out, frame)  # 保存图像
            else:
                out_video.write(frame)  # 写入视频帧

        # 计算并显示FPS（视频模式）
        if not image:
            frame_counter += 1
            current_time = time.time() - start
            if current_time >= 1:
                print("FPS:", frame_counter)
                start = time.time()
                frame_counter = 0
    else:
        break  # 视频结束或读取失败

# 释放资源
if out is not None and not image:
    out_video.release()

if show:
    if image:
        cv2.waitKey(0)  # 图像模式等待按键
    cap.release()
    cv2.destroyWindow('Detect')  # 关闭显示窗口