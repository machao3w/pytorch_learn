# softmax回归

import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()
# ToTensor实例 可以将图像数据从PIL类型变换成32位浮点数格式
trans = transforms.ToTensor()

mnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=trans, download=True)


def get_fashion_mnist_labels(labels):
    """ 返回Fashion-MNIST数据集的文本标签 """
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """ Plot a listof images """
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


x, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
# show_images(x.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))

batch_size = 256


def get_dataloader_workers():
    return 4


if __name__ == '__main__':
    # 由于使用了多线程 所以要放入main函数里运行
    train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers())
    timer = d2l.Timer()
    for x, y in train_iter:
        # print(x)
        # print(y)
        continue
    print(f'{timer.stop():.2f} sec')


def load_data_fashion_mnist(batch_size, resize=None):
    """ 下载数据集，加载到内存"""
    trans = [transforms.ToTensor()]
    if resize:
        # 将图片变大
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=trans, download=True)
    return (
        data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()),
        data.DataLoader(mnist_test, batch_size, shuffle=True, num_workers=get_dataloader_workers())
    )
