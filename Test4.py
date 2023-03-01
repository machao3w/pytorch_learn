# softmax回归 从零开始实现
import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 原图片28*28 拉成向量为784
num_inputs = 784
# 有10个类别 输出维度
num_outputs = 10

w = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
# 偏移量
b = torch.zeros(num_outputs, requires_grad=True)


def softmax(x):
    x_exp = torch.exp(x)
    partition = x_exp.sum(1, keepdim=True)
    return x_exp / partition  # 广播机制


def net(x):
    # x reshape 成 256 * 784的矩阵
    return softmax(torch.matmul(x.reshape((-1, w.shape[0])), w) + b)


y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# y_hat中 第零个样本的第零个个元素 和第一个样本的第二个元素
print(y_hat[[0, 1], y])
print(y_hat[range(len(y_hat)), y])


# 交叉熵损失函数
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])


print(cross_entropy(y_hat, y))
