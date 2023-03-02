import torch

x = torch.randn(2, 4)
print(x)

y0 = torch.argmax(x, dim=1)
print(y0)
y0_ = x.argmax(axis=1)
y0__ = x.argmax(dim=1)
print(y0_)
print(y0__)

# 测试发现上述三种方法等价

# 维度按照 randn的输入决定 比如输入2个维度 第一个维度张量为2 第二个维度张量为4

# dim=0 理解为 消除第一维度， 保留两个长度为4的向量，依次比较两个向量中 每个元素最大的那个 并返回索引
# 返回的tensor的size = 消除dim之后的size 索引范围为[0,dim的size-1]



