import torch
from torch import nn
from tools.utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=3):
        super().__init__()

        # 使用均方误差损失函数，设置reduction='sum'表示对所有元素求和后返回
        self.mse = nn.MSELoss(reduction='sum')

        # 初始化参数：
        # S: 网格划分的大小（默认7x7）
        # B: 每个网格预测的边界框数量（默认2个）
        # C: 类别数量（默认3类）
        self.S = S
        self.B = B
        self.C = C

        # 设置无目标框的损失权重（降低无目标框的损失影响）
        self.lambda_noobj = 0.5
        # 设置坐标损失的权重（增加坐标预测的重要性）
        self.lambda_coord = 5

    """
    target = [P, x, y, w, h, c1, c2, c3] 
    prediction = [P, x, y, w, h, P, x, y, w, h, c1, c2, c3]
    expected shape of predictions = S * S * (5 * B + C) * batch_size
    """

    def forward(self, predictions, target):
        # 获取批量大小
        batch_size = target.size(0)

        # 初始化各项损失值
        loss_xy = 0  # 边界框中心坐标(x,y)的损失
        loss_wh = 0  # 边界框宽高(w,h)的损失
        loss_obj = 0  # 有目标框的置信度损失
        loss_no_obj = 0  # 无目标框的置信度损失
        loss_class = 0  # 类别预测损失

        # 遍历每个样本
        for i in range(batch_size):
            # 遍历每个网格的行
            for y in range(self.S):
                # 遍历每个网格的列
                for x in range(self.S):
                    # 检查当前网格是否有目标（P=1表示有目标）
                    if target[i, y, x, 0] == 1:
                        # 获取当前网格的目标值
                        target_box = target[i, y, x]
                        # 获取当前网格的预测值
                        pred_box = predictions[i, y, x]

                        # 计算第一个预测框与真实框的IoU
                        iou1 = intersection_over_union(pred_box[1:5], target_box[1:5])
                        # 计算第二个预测框与真实框的IoU
                        iou2 = intersection_over_union(pred_box[6:10], target_box[1:5])

                        # 选择IoU较大的预测框作为负责预测的框
                        if iou1 > iou2:
                            iou = iou1
                            selected_box = pred_box[:5]  # 选择第一个预测框
                            selected_box = torch.cat((selected_box, pred_box[10:]))  # 拼接类别预测
                            unselected_box = pred_box[5:]  # 未选中的是第二个预测框
                        else:
                            iou = iou2
                            selected_box = pred_box[5:]  # 选择第二个预测框
                            unselected_box = pred_box[:5]  # 未选中的是第一个预测框
                            unselected_box = torch.cat((unselected_box, pred_box[10:]))  # 拼接类别预测

                        # 计算选中框的中心坐标(x,y)与真实框的MSE损失
                        loss_xy += self.mse(selected_box[1:3], target_box[1:3])

                        # 计算选中框的宽高(w,h)与真实框的MSE损失（使用平方根处理）
                        loss_wh += self.mse(
                            torch.sign(selected_box[3:5]) * torch.sqrt(torch.abs(selected_box[3:5])),
                            target_box[3:5].sqrt()
                        )

                        # 计算选中框的置信度损失（目标是IoU值）
                        loss_obj += (selected_box[0] - iou * 1) ** 2

                        # 计算未选中框的置信度损失（目标应为0）
                        loss_no_obj += (unselected_box[0] - 0) ** 2

                        # 计算类别预测的MSE损失
                        loss_class += self.mse(selected_box[5:], target_box[5:])

                    else:  # 当前网格没有目标
                        # 计算两个预测框的置信度损失（目标都为0）
                        loss_no_obj += torch.sum((predictions[i, y, x, [0, 5]] - 0) ** 2)

        # 加权求和所有损失项：
        # 坐标损失乘以权重 + 有目标置信度损失 + 无目标置信度损失乘以权重 + 类别损失
        loss = self.lambda_coord * (loss_xy + loss_wh) + loss_obj + self.lambda_noobj * loss_no_obj + loss_class

        # 返回平均损失（除以批量大小）
        return loss / batch_size