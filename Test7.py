# 多层感知机简单实现
import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))


def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0)


net.apply(init_weight)

batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

if __name__ == '__main__':
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
