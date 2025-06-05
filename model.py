import numpy as np
from torch import nn
from tools.loadWeights import load_conv, load_conv_batch_norm


class Extraction(nn.Module):
    def __init__(self):
        super().__init__()

        # 定义卷积块序列（特征提取主干网络）
        self.conv_block = nn.Sequential(
            # 第一层卷积：3通道输入，64通道输出，7x7卷积核，步长2，padding3（保持尺寸计算）
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),  # 批归一化层
            nn.LeakyReLU(negative_slope=0.1),  # LeakyReLU激活函数（负斜率0.1）
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大池化层（下采样）

            # 第二层卷积块
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第三层卷积块（包含多个1x1和3x3卷积的组合）
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1),
            # 更多卷积层...

            # 最终输出1024通道的特征图
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1)
        )

        # 分类器部分（原设计用于ImageNet分类）
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1000, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.AvgPool2d(kernel_size=13, stride=13),  # 全局平均池化
            nn.Flatten()  # 展平输出
        )

    def forward(self, x):
        # 前向传播：先通过卷积块，再通过分类器
        x = self.conv_block(x)
        return self.classifier(x)

    def load_weights(self, weightfile):
        """
        从权重文件加载预训练权重
        - 模型目标权重大小应为23455400
        - 权重文件转换为numpy后应有23455400大小
        - 注意前4个索引是文件头信息

        参数: weightfile - 权重文件的二进制文件路径
        """
        with open(weightfile, 'rb') as fp:
            # 读取文件头（4个int32数值）
            header = np.fromfile(fp, count=4, dtype=np.int32)
            # 读取剩余权重数据（float32格式）
            buf = np.fromfile(fp, dtype=np.float32)
            start = 0  # 权重读取起始位置

            # 为卷积层加载权重
            for num_layer, layer in enumerate(self.conv_block):
                if start >= buf.size:
                    break
                # 如果是卷积层
                if isinstance(layer, nn.modules.conv.Conv2d):
                    conv_layer = self.conv_block[num_layer]
                    # 检查下一层是否是批归一化层
                    if num_layer + 1 != len(self.conv_block):
                        if isinstance(self.conv_block[num_layer + 1], nn.modules.BatchNorm2d):
                            batch_norm_layer = self.conv_block[num_layer + 1]
                            # 加载带BN的卷积权重
                            start = load_conv_batch_norm(buf, start, conv_layer, batch_norm_layer)
                    else:
                        # 加载普通卷积权重
                        start = load_conv(buf, start, conv_layer)

            # 为输出层加载权重
            conv_layer = self.classifier[0]
            start = load_conv(buf, start, conv_layer)

            # 检查是否所有权重都已加载
            if start == buf.size:
                print("Extraction权重文件加载成功")


class YOLOv1(nn.Module):
    """
    YOLOv1类实现原始YOLOv1模型架构
    包含Extraction类作为CNN主干网络
    """

    def __init__(self, S=7, B=2, C=3, device='cpu', extraction_weights_path=None):
        """
        参数:
        - S: 网格大小（默认7）
        - B: 每个网格预测的边界框数量（默认2）
        - C: 类别数量（默认3）
        - device: 模型运行的设备（默认cpu）
        - extraction_weights_path: Extraction主干网络的权重文件路径（默认None）
        """
        super(YOLOv1, self).__init__()

        self.S = S  # 网格尺寸
        self.B = B  # 每个网格的预测框数
        self.C = C  # 类别数

        # 初始化特征提取主干网络并移动到指定设备
        self.extraction = Extraction().to(device)

        # 如果提供了预训练权重，则加载
        if extraction_weights_path is not None:
            self.extraction.load_weights(extraction_weights_path)

        # 替换原分类器为YOLO特有的检测头
        self.extraction.classifier = nn.Sequential(
            # 检测头由多个3x3卷积组成
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),
            # 更多卷积层...
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1)
        )

        # 检测部分（全连接层）
        self.detection = nn.Sequential(
            nn.Flatten(),  # 展平特征图
            nn.Linear(in_features=self.S * self.S * 1024, out_features=4096),  # 全连接层
            nn.Dropout(p=0.5),  # Dropout防止过拟合
            nn.LeakyReLU(negative_slope=0.1),
            # 输出层：预测SxS网格x(5*B+C)个值（5=框坐标+置信度，B=框数，C=类别数）
            nn.Linear(in_features=4096, out_features=self.S * self.S * (5 * self.B + self.C))
        )

    def forward(self, x):
        # 前向传播：先通过特征提取网络，再通过检测头
        x = self.extraction(x)
        x = self.detection(x)
        # 调整输出形状为[batch_size, S, S, 5*B+C]
        batch_size = x.shape[0]
        return x.reshape(batch_size, self.S, self.S, 5 * self.B + self.C)