# 多层感知机
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs, num_outputs, num_hiddens = 784, 10, 256
w1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True))
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
w2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True))
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [w1, b1, w2, b2]


def relu(x):
    # 形状一样 但是元素为0
    a = torch.zeros_like(x)
    return torch.max(x, a)


def net(x):
    x = x.reshape((-1, num_inputs))
    # 矩阵乘法 @ 等价与 torch.matmul(x,w1)
    h = relu(x @ w1 + b1)
    return (h @ w2 + b2)


loss = nn.CrossEntropyLoss()

num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)

if __name__ == '__main__':
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
    # plt.show()
