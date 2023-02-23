import torch
import random
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
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)]
        )
        yield features[batch_indices], labels[batch_indices]


batch_size = 10
for x, y in data_iter(batch_size, features, labels):
    print(x,'\n',y)
    break
##