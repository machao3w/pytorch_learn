# 卷积神经网络
import torch
from torch import nn
from d2l import torch as d2l

device = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu'
)


def corr2d(X, K):
    """ 计算二维互相关运算 """
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1), device=device)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]], device=device)
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]], device=device)
Y = corr2d(X, K)
print(Y)


# print(X.device)

# 二维卷积层
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


# 检测图像中不同颜色的边缘

X = torch.ones((6, 8), device=device)
X[:, 2:6] = 0
print(X)

K = torch.tensor([[1.0, -1.0]], device=device)
Y = corr2d(X, K)
print(Y)

# 转置 X.t()


# 学习卷积核
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False, device=device)
# 前两个维度 为 通道维 批量大小维
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape(1, 1, 6, 7)

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # 梯度下降 学习率0.03
    conv2d.weight.data[:] -= 3e-2 * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'batch {i + 1},loss {l.sum():.3f}')

conv2d.weight.data.reshape((1, 2))
