# softmax回归

import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()
# ToTensor实例 可以将图像数据从PIL类型变换成32位浮点数格式
trans = transforms.ToTensor

mnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=trans, download=True)

def get_fashion_mnist_labels(labels):
    """ 返回Fashion-MNIST数据集的文本标签 """
    text_labels = [
        't-shirt','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle boot'
    ]
