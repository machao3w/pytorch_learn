# LeNet
import torch
from torch import nn
from d2l import torch as d2l

device = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu'
)


class Reshape(nn.Module):
    def forward(self, x):
        # 批量数不变 通道数遍为1
        return x.view(-1, 1, 28, 28)


net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=(5, 5), padding=2, device=device),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=(5, 5), device=device), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120, device=device), nn.Sigmoid(),
    nn.Linear(120, 84, device=device), nn.Sigmoid(),
    nn.Linear(84, 10, device=device)
)

X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32, device=device)

for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)
    print(X.device)
