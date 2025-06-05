import torch


def load_conv_batch_norm(buf, start, conv_layer, batch_norm_layer):
    """
    加载卷积
    参数:层(带BatchNorm)的权重和偏置参数

        buf: 包含权重数据的numpy数组
        start: 当前读取位置的起始索引
        conv_layer: 要加载权重的卷积层
        batch_norm_layer: 要加载权重的BatchNorm层

    返回:
        start: 更新后的读取位置索引
    """
    # 加载BatchNorm层的偏置(bias)
    num_b = batch_norm_layer.bias.numel()  # 获取偏置参数的数量
    batch_norm_layer.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]))  # 从buf复制数据到层
    start += num_b  # 移动读取位置

    # 加载BatchNorm层的权重(weight)
    batch_norm_layer.weight.data.copy_(torch.from_numpy(buf[start:start + num_b]))
    start += num_b

    # 加载BatchNorm层的running_mean(运行均值)
    batch_norm_layer.running_mean.copy_(torch.from_numpy(buf[start:start + num_b]))
    start += num_b

    # 加载BatchNorm层的running_var(运行方差)
    batch_norm_layer.running_var.copy_(torch.from_numpy(buf[start:start + num_b]))
    start += num_b

    # 加载卷积层的权重
    num_w = conv_layer.weight.numel()  # 获取权重参数的数量
    conv_layer.weight.data.copy_(
        torch.from_numpy(buf[start:start + num_w]).reshape(conv_layer.weight.data.shape))
    start += num_w
    return start  # 返回新的读取位置


def load_conv(buf, start, conv_layer):
    """
    加载普通卷积层(不带BatchNorm)的权重和偏置参数

    参数:
        buf: 包含权重数据的numpy数组
        start: 当前读取位置的起始索引
        conv_layer: 要加载权重的卷积层

    返回:
        start: 更新后的读取位置索引
    """
    # 加载卷积层的偏置(bias)
    num_b = conv_layer.bias.numel()  # 获取偏置参数的数量
    conv_layer.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]))  # 复制数据
    start += num_b  # 移动读取位置

    # 加载卷积层的权重
    num_w = conv_layer.weight.numel()  # 获取权重参数的数量
    conv_layer.weight.data.copy_(
        torch.from_numpy(buf[start:start + num_w]).reshape(conv_layer.weight.data.shape))
    start += num_w
    return start  # 返回新的读取位置