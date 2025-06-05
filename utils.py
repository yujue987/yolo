import torch  # PyTorch深度学习框架，用于张量计算和模型训练
from tqdm import tqdm  # 进度条工具，用于可视化处理进度


def intersection_over_union(predicted_bbox, ground_truth_bbox) -> float:
    """
    计算两个边界框的交并比(IOU)

    参数:
        predicted_bbox: 预测框 [中心x, 中心y, 宽度w, 高度h]
        ground_truth_bbox: 真实框 [中心x, 中心y, 宽度w, 高度h]

    返回:
        IOU值(0到1之间的浮点数)
    """
    # 将中心坐标转换为左上和右下坐标(x1,y1,x2,y2格式)
    predicted_bbox_x1 = predicted_bbox[0] - predicted_bbox[2] / 2  # 预测框左上x
    predicted_bbox_y1 = predicted_bbox[1] - predicted_bbox[3] / 2  # 预测框左上y
    predicted_bbox_x2 = predicted_bbox[0] + predicted_bbox[2] / 2  # 预测框右下x
    predicted_bbox_y2 = predicted_bbox[1] + predicted_bbox[3] / 2  # 预测框右下y

    ground_truth_bbox_x1 = ground_truth_bbox[0] - ground_truth_bbox[2] / 2  # 真实框左上x
    ground_truth_bbox_y1 = ground_truth_bbox[1] - ground_truth_bbox[3] / 2  # 真实框左上y
    ground_truth_bbox_x2 = ground_truth_bbox[0] + ground_truth_bbox[2] / 2  # 真实框右下x
    ground_truth_bbox_y2 = ground_truth_bbox[1] + ground_truth_bbox[3] / 2  # 真实框右下y

    # 计算交集的坐标
    intersection_bbox = torch.tensor(
        [
            max(predicted_bbox_x1, ground_truth_bbox_x1),  # 交集左上x
            max(predicted_bbox_y1, ground_truth_bbox_y1),  # 交集左上y
            min(predicted_bbox_x2, ground_truth_bbox_x2),  # 交集右下x
            min(predicted_bbox_y2, ground_truth_bbox_y2),  # 交集右下y
        ]
    )

    # 计算交集面积(确保非负)
    intersection_area = max(intersection_bbox[2] - intersection_bbox[0], 0) * max(
        intersection_bbox[3] - intersection_bbox[1], 0
    )

    # 计算预测框面积
    area_predicted = (predicted_bbox_x2 - predicted_bbox_x1) * (predicted_bbox_y2 - predicted_bbox_y1)
    # 计算真实框面积
    area_gt = (ground_truth_bbox_x2 - ground_truth_bbox_x1) * (ground_truth_bbox_y2 - ground_truth_bbox_y1)

    # 计算并集面积
    union_area = area_predicted + area_gt - intersection_area

    # 计算IOU
    iou = intersection_area / union_area
    return iou


def non_max_supression(bboxes, iou_threshold, threshold):
    """
    非极大值抑制(NMS)算法

    参数:
        bboxes: 边界框列表，每个框格式为[置信度, x, y, w, h, class_scores...]
        iou_threshold: IOU阈值，用于判断是否重叠
        threshold: 置信度阈值，过滤低置信度预测

    返回:
        经过NMS处理后的边界框列表
    """
    # 1. 过滤低置信度框
    bboxes = [box for box in bboxes if box[0] >= threshold]
    # 2. 按置信度降序排序
    bboxes = sorted(bboxes, key=lambda x: x[0], reverse=True)

    non_max_bboxes = []  # 存储最终保留的框
    while bboxes:
        current_box = bboxes.pop(0)  # 取出当前最高置信度的框
        non_max_bboxes.append(current_box)  # 保留该框

        temp_bboxes = []  # 临时存储下一轮处理的框
        for box in bboxes:
            # 获取当前比较的两个框的类别
            class_box = torch.argmax(box[5:])
            class_current_box = torch.argmax(current_box[5:])

            # 如果IOU低于阈值或类别不同则保留
            if intersection_over_union(current_box[1:5], box[1:5]) < iou_threshold or class_box != class_current_box:
                temp_bboxes.append(box)

        bboxes = temp_bboxes  # 更新待处理框列表

    return non_max_bboxes


def convert_to_yolo(bbox, grid_cell_x, grid_cell_y, S):
    """
    将相对于网格单元的坐标转换为相对于整张图像的YOLO格式坐标

    参数:
        bbox: 边界框 [x, y, w, h]
        grid_cell_x: 网格单元x坐标
        grid_cell_y: 网格单元y坐标
        S: 网格尺寸

    返回:
        转换后的边界框
    """
    # 转换x,y坐标(从网格相对坐标到图像相对坐标)
    bbox[0] = (bbox[0] + grid_cell_x) / S  # x坐标转换
    bbox[1] = (bbox[1] + grid_cell_y) / S  # y坐标转换
    # w,h保持不变(已经是相对尺寸)
    return bbox


def get_bound_boxes(loader, model, iou_threshold=0.5, threshold=0.4, device='cpu'):
    """
    获取预测和真实的边界框(应用非极大值抑制)

    参数:
        loader: 数据加载器
        model: 训练好的模型
        iou_threshold: IOU阈值(默认0.5)
        threshold: 置信度阈值(默认0.4)
        device: 计算设备(cpu或gpu)

    返回:
        所有预测框和真实框的列表
    """
    # 验证loader类型
    assert isinstance(loader, torch.utils.data.dataloader.DataLoader), \
        "loader does not match the type of torch.utils.data.dataloader.DataLoader"

    model.eval()  # 设置模型为评估模式
    # 初始化存储变量
    predictions = None
    targets = None

    # 遍历数据加载器
    for i, batch in enumerate(tqdm(loader, desc=f'Prediction all bound boxes', leave=False)):
        images = batch['image'].to(device)  # 将图像移动到指定设备

        if i == 0:  # 第一次迭代初始化存储
            targets = batch['target'].to(device)
            with torch.no_grad():  # 禁用梯度计算
                predictions = model(images)
        else:  # 后续迭代拼接结果
            target = batch['target'].to(device)
            targets = torch.cat((targets, target))
            with torch.no_grad():
                predictions = torch.cat((predictions, model(images)))

    # 获取预测结果的尺寸信息
    size = predictions.size(0)  # 批次数
    S = predictions.size(1)  # 网格尺寸

    all_pred_boxes = []  # 存储所有预测框
    all_true_boxes = []  # 存储所有真实框

    # 处理每个图像
    for i in range(0, size):
        image_pred_boxes = []  # 当前图像的预测框
        image_true_boxes = []  # 当前图像的真实框

        # 遍历每个网格单元
        for y in range(0, S):
            for x in range(0, S):
                # 处理真实框(如果有目标)
                if targets[i, y, x, 0] == 1:  # 检查是否有目标
                    pred_box = targets[i, y, x]  # 获取真实框
                    pred_box[1:5] = convert_to_yolo(pred_box[1:5], x, y, S)  # 坐标转换
                    image_true_boxes.append(pred_box)  # 添加到真实框列表

                # 处理第一个anchor的预测框
                pred_box = torch.empty(targets.size(-1), dtype=torch.float32)
                pred_box[:5] = predictions[i, y, x, :5]  # 置信度和坐标
                pred_box[5:] = predictions[i, y, x, 10:]  # 类别概率
                pred_box[1:5] = convert_to_yolo(pred_box[1:5], x, y, S)  # 坐标转换
                image_pred_boxes.append(pred_box)  # 添加到预测框列表

                # 处理第二个anchor的预测框
                pred_box = torch.empty(targets.size(-1), dtype=torch.float32)
                pred_box = predictions[i, y, x, 5:]  # 获取第二个anchor的预测
                pred_box[1:5] = convert_to_yolo(pred_box[1:5], x, y, S)  # 坐标转换
                image_pred_boxes.append(pred_box)  # 添加到预测框列表

        # 对当前图像的预测框应用NMS
        image_pred_boxes = non_max_supression(image_pred_boxes, iou_threshold, threshold)
        all_pred_boxes.append(image_pred_boxes)  # 添加到总预测框列表
        all_true_boxes.append(image_true_boxes)  # 添加到总真实框列表

    return all_pred_boxes, all_true_boxes  # 返回所有预测和真实框