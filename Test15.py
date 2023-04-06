# AlexNet
import torch
from torch import nn
from d2l import torch as d2l

device = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu'
)

net = nn.Sequential(
    nn.Conv2d(1, 96, kernel_size=(11, 11), stride=(4, 4), padding=1, device=device),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),

    nn.Conv2d(96, 256, kernel_size=(5, 5), padding=2, device=device),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),

    nn.Conv2d(256, 384, kernel_size=(3, 3), padding=1, device=device),
    nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=(3, 3), padding=1, device=device),
    nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1, device=device),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),

    nn.Linear(6400, 4096, device=device),
    nn.ReLU(),
    # 暂退法 减轻过拟合
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096, device=device),
    nn.ReLU(),
    nn.Dropout(p=0.5),

    nn.Linear(4096, 10, device=device)

)

# X = torch.randn(1, 1, 224, 224, device=device)
# for layer in net:
#     X = layer(X)
#     print(layer.__class__.__name__, 'output shape:\t', X.shape)

batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

lr, num_epochs = 0.01, 10

if __name__ == '__main__':
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

