import numpy as np
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


def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


class Net(Module):

    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(1, 16, 5)
        self.conv2 = Conv2d(16, 64, 5)
        self.pool = MaxPool2d(2)

        self.feature_size = 4 * 4 * 64

        self.l1 = Linear(self.feature_size, 10)
        self.relu = ReLU(False)

    def forward(self, input):
        out = self.pool(self.relu(self.conv1(input.view(-1, 1, 28, 28))))
        out = self.pool(self.relu(self.conv2(out)))
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

    output = np.argmax(net(data[0]).detach())

    print(output)

    plt.imshow(data[0].view(28, 28), cmap='gray')
    plt.show()
