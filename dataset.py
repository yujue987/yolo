import os  # 操作系统接口模块，用于文件路径操作
import xmltodict  # XML解析库，用于解析标注文件
import torch  # PyTorch深度学习框架
import numpy as np  # 数值计算库
from PIL import Image  # 图像处理库
from torch.utils.data import Dataset  # PyTorch数据集基类
from typing import List  # 类型提示支持


class Dataset(Dataset):
    def __init__(self, data_dir, labels_dir, transforms, S=7, C=3, file_format='txt', convert_to_yolo=True):
        """
        目标检测数据集类

        参数:
            data_dir: 数据目录路径
            labels_dir: 标签文件路径
            transforms: 数据增强变换
            S: 网格尺寸(默认7)
            C: 类别数量(默认3)
            file_format: 标注文件格式(txt/xml，默认txt)
            convert_to_yolo: 是否转换为YOLO格式(默认True)
        """
        # 初始化类别标签映射字典
        self.class2tag = {}
        with open(labels_dir, 'r') as f:
            for line in f:
                (val, key) = line.split()  # 解析标签文件行
                self.class2tag[key] = val  # 存储类别到标签的映射

        # 初始化图像和标注文件路径列表
        self.image_paths = []
        self.box_paths = []
        # 遍历每个类别目录
        for tag in self.class2tag:
            for file in os.listdir(data_dir + '/' + tag):
                if file.endswith('.jpg'):  # 收集jpg图像文件
                    self.image_paths.append(data_dir + '/' + tag + '/' + file)
                if file.endswith('.' + file_format):  # 收集标注文件
                    self.box_paths.append(data_dir + '/' + tag + '/' + file)

        # 对文件路径排序以确保对应关系
        self.image_paths = sorted(self.image_paths)
        self.box_paths = sorted(self.box_paths)

        # 验证图像和标注文件数量匹配
        assert len(self.image_paths) == len(self.box_paths)

        # 存储初始化参数
        self.transforms = transforms  # 数据增强变换
        self.S = S  # 网格尺寸
        self.C = C  # 类别数量
        self.file_format = file_format  # 标注文件格式
        self.convert_to_yolo = convert_to_yolo  # YOLO格式转换标志

    def __getitem__(self, idx):
        """获取单个样本数据"""
        # 加载图像并转换为RGB格式
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))

        # 根据文件格式加载标注框和类别标签
        if self.file_format == 'xml':
            bboxes, class_labels = self.__get_boxes_from_xml(self.box_paths[idx])
        if self.file_format == 'txt':
            bboxes, class_labels = self.__get_boxes_from_txt(self.box_paths[idx])

        # 如果需要转换为YOLO格式
        if self.convert_to_yolo:
            for i, box in enumerate(bboxes):
                bboxes[i] = self.__convert_to_yolo_box_params(box, image.shape[1], image.shape[0])

        # 应用数据增强变换
        transformed = self.transforms(image=image, bboxes=bboxes, class_labels=class_labels)
        transformed_image = transformed['image']  # 变换后的图像
        transformed_bboxes = torch.tensor(transformed['bboxes'])  # 变换后的边界框
        transformed_class_labels = torch.tensor(transformed['class_labels'])  # 变换后的类别标签

        """
        创建目标矩阵
        每个网格单元 = [P, x, y, w, h, c1, c2, c3]
        网格尺寸 = S * S
        如果网格单元中有多个框，则选择最后一个框

        x, y值相对于网格单元计算
        """
        # 初始化目标张量(S×S×(5+C))
        target = torch.tensor([[0] * (5 + self.C)] * self.S * self.S, dtype=torch.float32)
        target = target.reshape((self.S, self.S, (5 + self.C)))

        # 填充目标矩阵
        for i, box in enumerate(transformed_bboxes):
            class_tensor = torch.zeros(self.C, dtype=torch.float32)  # 初始化类别one-hot向量
            class_tensor[transformed_class_labels[i]] = 1  # 设置对应类别为1

            # 计算边界框所属的网格单元
            x_cell = int(self.S * box[0])
            y_cell = int(self.S * box[1])

            # 填充网格单元数据
            target[y_cell, x_cell] = torch.cat((torch.tensor(
                [
                    1,  # 存在目标的置信度
                    self.S * box[0] - x_cell,  # 相对于网格单元的x中心
                    self.S * box[1] - y_cell,  # 相对于网格单元的y中心
                    box[2],  # 宽度
                    box[3]  # 高度
                ]
            ), class_tensor), dim=0)  # 拼接类别信息

        return {"image": transformed_image, "target": target}  # 返回样本字典

    def __len__(self):
        """返回数据集大小"""
        return len(self.image_paths)

    def __get_boxes_from_txt(self, txt_filename: str):
        """从txt文件解析边界框和类别标签"""
        boxes = []  # 存储边界框
        class_labels = []  # 存储类别标签

        with open(txt_filename) as f:
            for obj in f:
                param_list = list(map(float, obj.split()))  # 解析每行数据

                boxes.append(param_list[1:])  # 添加边界框坐标(x,y,w,h)
                class_labels.append(int(param_list[0]))  # 添加类别标签

        return boxes, class_labels

    def __get_boxes_from_xml(self, xml_filename: str):
        """从xml文件解析边界框和类别标签"""
        boxes = []  # 存储边界框
        class_labels = []  # 存储类别标签

        with open(xml_filename) as f:
            xml_content = xmltodict.parse(f.read())  # 解析XML文件
        xml_object = xml_content['annotation']['object']  # 获取对象列表

        # 处理单个或多个对象的情况
        if type(xml_object) is dict:
            xml_object = [xml_object]  # 单个对象转换为列表

        if type(xml_object) is list:
            for obj in xml_object:
                # 解析边界框坐标(xmin,ymin,xmax,ymax)
                boxe_list = list(map(float, [obj['bndbox']['xmin'], obj['bndbox']['ymin'],
                                             obj['bndbox']['xmax'], obj['bndbox']['ymax']]))
                boxes.append(boxe_list)  # 添加边界框
                class_labels.append(self.class2tag[obj['name']])  # 添加映射后的类别标签

        return boxes, class_labels

    def __convert_to_yolo_box_params(self, box_coordinates: List[int], im_w, im_h):
        """将边界框坐标转换为YOLO格式(中心坐标+宽高，相对值)"""
        ans = list()

        # 计算中心坐标(相对值)
        ans.append((box_coordinates[0] + box_coordinates[2]) / 2 / im_w)  # x_center
        ans.append((box_coordinates[1] + box_coordinates[3]) / 2 / im_h)  # y_center

        # 计算宽高(相对值)
        ans.append((box_coordinates[2] - box_coordinates[0]) / im_w)  # width
        ans.append((box_coordinates[3] - box_coordinates[1]) / im_h)  # height

        return ans