# softmax回归简单实现
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)

# 交叉熵损失函数
loss = nn.CrossEntropyLoss()

# 学习率为0.1的小批量随机梯度下降算法
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 10
# 训练模型
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
