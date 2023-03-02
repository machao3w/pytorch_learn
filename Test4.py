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


def accuracy(y_hat, y):
    """ 计算预测正确的数量 """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # argmax
        y_hat = y_hat.argmax(dim=1)
        # 数据类型比较 bool的tensor
    cmp = y_hat.type(y.dtype) == y
    # 返回所有预测正确的样本数
    return float(cmp.type(y.dtype).sum())


print(accuracy(y_hat, y) / len(y))


class Accumulator:
    """ 累加器 在n个变量上累加 """

    def __init__(self, n):
        self.data = [0, 0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0, 0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def evaluate_accuracy(net, data_iter):
    """ 计算在指定数据集上模型的精度 """
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    for x, y in data_iter:
        metric.add(accuracy(net(x), y), y.numel())
    return metric[0] / metric[1] # 分类正确的样本数 总样本数
