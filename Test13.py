# 多输入多通道
import torch
from d2l import torch as d2l

device = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu'
)


# 多输入通道
def corr2d_multi_in(X, K):
    # zip对[1:]维度做遍历
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))


X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]], device=device)

# (1 × 1 + 2 × 2 + 4 × 3 + 5 × 4) + (0 × 0 + 1 × 1 + 3 × 2 + 4 × 3) = 56
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]], device=device)

corr2d_multi_in(X, K)


# 多个通道输出

def corr2d_multi_in_out(X, K):
    # 迭代“K”的第0个维度，每次都对输⼊“X”执⾏互相关运算。
    # 最后将所有结果都叠加在⼀起
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)


# 在第0个维度 堆起来
K = torch.stack((K, K + 1, K + 2), 0)
print(K)
print(K.shape)

print(corr2d_multi_in_out(X, K))


# 1*1卷积
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape(c_o, c_i)
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))


X = torch.normal(0, 1, (3, 3, 3))
K = torch.normal(0, 1, (2, 3, 1, 1))

Y1 = corr2d_multi_in_out_1x1(X, K)
print(Y1)
Y2 = corr2d_multi_in_out(X, K)
print(Y2)

assert float(torch.abs(Y1 - Y2).sum()) < 1e-6
