import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


# 导入必要的库：
# - torch: PyTorch深度学习框架
# - matplotlib.pyplot: 绘图库，用于可视化训练过程
# - tqdm: 进度条库，用于显示训练进度


def evaluate(model, criterion, val_dataloader, device):
    """
    评估模型在验证集上的表现
    参数:
        model: 要评估的模型
        criterion: 损失函数
        val_dataloader: 验证集数据加载器
        device: 计算设备(cpu或gpu)
    返回:
        验证集上的平均损失
    """
    val_loss = 0  # 初始化验证损失为0
    model.eval()  # 将模型设置为评估模式(关闭dropout等训练专用层)

    # 使用tqdm包装验证数据加载器，显示进度条
    for batch in tqdm(val_dataloader, desc=f'Evaluation', leave=False):
        images, targets = batch['image'], batch['target']  # 从批次中获取图像和目标
        images, targets = images.to(device), targets.to(device)  # 将数据移动到指定设备

        with torch.no_grad():  # 禁用梯度计算，节省内存和计算资源
            predictions = model(images)  # 模型前向传播
            loss = criterion(predictions, targets)  # 计算损失
            val_loss += loss.item()  # 累加损失值

    return val_loss / len(val_dataloader)  # 返回平均损失


def fit(model, optimizer, scheduler, criterion, epochs, train_dataloader, val_dataloader, device='cpu'):
    """
    训练模型并绘制损失曲线

    参数:
        model: 要训练的模型
        optimizer: 优化器
        scheduler: 学习率调度器
        criterion: 损失函数
        epochs: 训练轮数
        train_dataloader: 训练集数据加载器
        val_dataloader: 验证集数据加载器
        device: 计算设备(默认为cpu)
    """
    train_loss_log = []  # 记录训练损失的列表
    val_loss_log = []  # 记录验证损失的列表

    # 创建绘图窗口
    fig = plt.figure(figsize=(11, 7))  # 设置图形大小
    fig_number = fig.number  # 获取图形编号，用于后续检查图形是否存在

    # 开始训练循环
    for epoch in range(epochs):
        model.train()  # 将模型设置为训练模式
        train_loss = 0  # 初始化训练损失为0

        # 使用tqdm包装训练数据加载器，显示进度条
        for batch in tqdm(train_dataloader, desc=f"Training, epoch {epoch}", leave=False):
            images, targets = batch['image'], batch['target']  # 从批次中获取图像和目标
            images, targets = images.to(device), targets.to(device)  # 将数据移动到指定设备

            predictions = model(images)  # 模型前向传播
            loss = criterion(predictions, targets)  # 计算损失
            train_loss += loss.item()  # 累加损失值
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新模型参数
            optimizer.zero_grad()  # 清空梯度

        scheduler.step()  # 更新学习率

        train_loss /= len(train_dataloader)  # 计算平均训练损失
        train_loss_log.append(train_loss)  # 记录训练损失

        # 在验证集上评估模型
        val_loss = evaluate(model, criterion, val_dataloader, device)
        val_loss_log.append(val_loss)  # 记录验证损失

        # 检查绘图窗口是否仍然存在，如果不存在则重新创建
        if not plt.fignum_exists(num=fig_number):
            fig = plt.figure(figsize=(11, 7))
            fig_number = fig.number

        # 打印当前epoch的训练信息
        print(f"epoch: {epoch}")
        print(f"train loss: {train_loss}")
        print(f"val loss: {val_loss}")

        # 绘制损失曲线
        line_train, = plt.plot(list(range(0, epoch + 1)), train_loss_log, color='blue')  # 训练损失曲线(蓝色)
        line_val, = plt.plot(list(range(0, epoch + 1)), val_loss_log, color='orange')  # 验证损失曲线(橙色)

        # 设置图表属性
        plt.title("Loss")  # 图表标题
        plt.xlabel("epoch")  # x轴标签
        plt.ylabel("loss")  # y轴标签
        plt.title("Train steps")  # 图表标题(重复设置)
        plt.legend((line_train, line_val), ['train loss', 'validation loss'])  # 图例

        plt.draw()  # 重绘图形
        plt.pause(0.001)  # 短暂暂停，确保图形更新

        # 保存损失曲线图像
        fig.savefig('loss.png', bbox_inches='tight')