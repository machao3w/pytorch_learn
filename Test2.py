import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
# 预定义的模型
from torch import nn

# 线性回归简洁实现
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


def load_array(data_arrays, batch_size, is_train=True):
    """ 构造一个PyTorch 数据迭代器 """
    dataset = data.TensorDataset(*data_arrays)
    # 每次从里面随机挑选batch_size大小的样本 shuffle是否随机打乱
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)
# 转换成迭代器
next(iter(data_iter))

# Linear 线性模型
# Sequential 网络容器
net = nn.Sequential(nn.Linear(2, 1))

# normal 使用正态分布替换weight 均值为0 方差为1
net[0].weight.data.normal_(0, 0.01)
# 偏差
net[0].bias.data.fill_(0)

# 均方误差
loss = nn.MSELoss()

trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epochs = 3
for epoch in range(num_epochs):
    for x, y in data_iter:
        l = loss(net(x), y)
        # 先把梯度清零
        trainer.zero_grad()
        l.backward()
        # 模型更新
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
