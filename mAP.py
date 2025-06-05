import torch  # 导入PyTorch深度学习框架
from tools.utils import intersection_over_union  # 从自定义工具导入IOU计算函数


def mean_average_precision(pred_boxes, true_boxes, classes, iou_threshold=0.5) -> float:
    """
    计算边界框的平均精度均值(mAP)

    参数:
        pred_boxes: 预测的边界框列表(每个图像一个列表)
        true_boxes: 真实的边界框列表(每个图像一个列表)
        classes: 类别数量
        iou_threshold: IOU判定阈值(默认0.5)

    返回:
        计算得到的mAP值(float)
    """
    average_precisions = []  # 存储每个类别的平均精度(AP)

    # 对每个类别单独计算AP
    for current_class in range(classes):
        FP = 0  # 假阳性(误检)
        FN = 0  # 假阴性(漏检)
        TP = 0  # 真阳性(正确检测)
        precisions = []  # 精度记录
        recalls = []  # 召回率记录

        # 初始化FN(统计该类别所有真实框数量)
        for i in range(len(true_boxes)):
            FN += len([box for box in true_boxes[i] if torch.argmax(box[5:]) == current_class])

        # 处理每个预测框
        for i in range(len(pred_boxes)):
            # 筛选当前类别的预测框和真实框
            pred_boxes_class = [box for box in pred_boxes[i] if torch.argmax(box[5:]) == current_class]
            true_boxes_class = [box for box in true_boxes[i] if torch.argmax(box[5:]) == current_class]

            # 对每个预测框计算最佳匹配
            for k in range(len(pred_boxes_class)):
                max_iou = 0  # 最大IOU值
                max_index = 0  # 最佳匹配的真实框索引

                # 寻找最佳匹配的真实框
                for j in range(len(true_boxes_class)):
                    iou = intersection_over_union(pred_boxes_class[k][1:5], true_boxes_class[j][1:5])
                    if iou > max_iou:
                        max_iou = iou
                        max_index = j

                # 根据IOU阈值判断是FP还是TP
                if max_iou < iou_threshold:
                    FP += 1  # 误检
                else:
                    TP += 1  # 正确检测
                    FN -= 1  # 减少漏检计数
                    true_boxes_class.pop(max_index)  # 移除已匹配的真实框

                # 记录当前精度和召回率
                precisions.append(TP / (TP + FP))
                recalls.append(TP / (TP + FN))

        # 计算当前类别的AP(使用梯形法计算PR曲线下面积)
        precisions = torch.tensor(precisions)
        recalls = torch.tensor(recalls)
        average_precisions.append(torch.trapezoid(precisions, recalls))

    # 返回所有类别AP的平均值(mAP)
    return sum(average_precisions) / len(average_precisions)