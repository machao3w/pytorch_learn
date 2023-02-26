import torch
import random

# 线性回顾从零实现
# from d2l import torch as d2l


def synthetic_data(w, b, num_examples):
    # 均值为0 方差为1 1000行 2列
    x = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(x, w) + b
    # 噪音
    e = torch.normal(0, 0.01, y.shape)
    y = y + e
    # -1 表示无此列
    return x, y.reshape((-1, 1))


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    # 每次随机取十个
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)]
        )
        # 返回一个迭代器
        yield features[batch_indices], labels[batch_indices]


batch_size = 10
for x, y in data_iter(batch_size, features, labels):
    print(x, '\n', y)
    break
##
# 均值为0 方差为0.01 长度为 2 列表1 需要计算梯度
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
print(w)
# 偏差 size 为1 所有值为0
b = torch.zeros(1, requires_grad=True)


# 预测线性回归模型
def linreg(x, w, b):
    return torch.matmul(x, w) + b


# 损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# 小批量随机梯度下降
# lr 学习率
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


lr = 0.03  # 学习率
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for x, y in data_iter(batch_size, features, labels):
        l = loss(net(x, w, b), y)
        l.sum().backward()
        # 更新 线性模型
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
