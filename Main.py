import numpy as np
from torch.distributions import one_hot_categorical
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn import Module, Linear, Conv2d, ReLU, MaxPool2d
from torch.nn.functional import l1_loss, mse_loss
import torch.optim as optim

import matplotlib.pyplot as plt

batch_size = 32

train_loader = DataLoader(datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])), batch_size=batch_size, shuffle=True)

test_loader = DataLoader(datasets.MNIST('../data', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=batch_size, shuffle=True)


def to_categorical(tensor:Tensor, num_classes=None):
    newtensor = []
    for t in tensor:
        newtensor.append(np.eye(num_classes)[t])
    return Tensor(newtensor)


def convToLinCalc(input_size:tuple, layers) -> np.array:
    size = np.array(input_size)
    for layer in layers:
        size = (size + 2*np.array(layer.padding) - np.array(layer.dilation) * (np.array(layer.kernel_size) - 1) - 1) / np.array(layer.stride) + 1
    return size


class Net(Module):

    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(1, 16, 5)
        self.conv2 = Conv2d(16, 64, 5)
        self.pool = MaxPool2d(2)

        out_size = convToLinCalc((28, 28), [self.conv1, self.pool, self.conv2, self.pool])

        self.feature_size = int(np.prod(out_size) * 64)

        self.l1 = Linear(self.feature_size, 10)
        self.relu = ReLU()

    def forward(self, input, show=False):
        out = input
        if show:
            out.to('cpu')
            a = out.detach()
            print(out.shape)
            plt.imshow(a[0].view(28, 28), cmap='gray')
            plt.show()
            out.to('cuda')
        out = self.pool(self.relu(self.conv1(out.view(-1, 1, 28, 28))))
        if show:
            out.to('cpu')
            a = out[0][0]
            for i in out[0][1:]:
                a += i
            plt.imshow(a.detach().view(12, 12), cmap='gray')
            plt.show()
            out.to('cuda')
        out = self.pool(self.relu(self.conv2(out)))
        if show:
            out.to('cpu')
            a = out[0][0]
            for i in out[0][1:]:
                a += i
            plt.imshow(a.detach().view(4, 4), cmap='gray')
            plt.show()
            out.to('cuda')
        out = self.relu(self.l1(out.view(-1, self.feature_size)))
        return out


net = Net()
optimizer = optim.SGD(net.parameters(), 0.01)

for i in train_loader:
    data, label = i

    optimizer.zero_grad()

    output = net(data)
    target = Tensor(to_categorical(label, 10))

    # print(output)
    # print(target)

    loss = mse_loss(output, target)
    loss.backward()
    optimizer.step()
    #print(loss)


for i in test_loader:
    data, label = i

    output = np.argmax(net(data, False).detach(), 1)

    print(output)
    print(label)
    print(output - label)
    print()

    plt.imshow(data[0].view(28, 28), cmap='gray')
    plt.show()
