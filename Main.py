from math import floor

import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from torch import Tensor
from torch.nn import Module, Linear, Conv2d, ReLU, MaxPool2d
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

batch_size = 50

train_loader = DataLoader(datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])), batch_size=batch_size, shuffle=True)

test_loader = DataLoader(datasets.MNIST('../data', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])), batch_size=batch_size, shuffle=True)


def to_categorical(tensor:Tensor, num_classes=None, device='cuda'):
    newtensor = []
    for t in tensor:
        newtensor.append(np.eye(num_classes)[t])
    return Tensor(newtensor).to(device)


def convToLinCalc(input_size:tuple, layers) -> np.array:
    size = np.array(input_size)
    for layer in layers:
        size[1:] = (size[1:] + 2*np.array(layer.padding) - np.array(layer.dilation) * (np.array(layer.kernel_size) - 1) - 1) / np.array(layer.stride) + 1
        if type(layer) == Conv2d:
            size[0] = layer.out_channels
    return size


class Net(Module):

    def __init__(self, first_conv_size=(4, 4), second_conv_size=(8,8)):
        super().__init__()
        self.cs1 = first_conv_size
        self.cs2 = second_conv_size
        self.conv1 = Conv2d(1, self.cs1[0] * self.cs1[1], 5)
        self.conv2 = Conv2d(self.cs1[0] * self.cs1[1], self.cs2[0] * self.cs2[1], 5)
        self.pool = MaxPool2d(2)

        out_size = convToLinCalc((1, 28, 28), [self.conv1, self.pool, self.conv2, self.pool])
        print(out_size)

        self.feature_size = int(np.prod(out_size))

        self.l1 = Linear(self.feature_size, 10)
        self.relu = ReLU()

    def forward(self, input, show=False):
        out = input
        if show:
            a = out.detach()
            a.to('cpu')
            plt.imshow(a[0].view(28, 28), cmap='gray')
            plt.show()
        out = self.pool(self.relu(self.conv1(out.view(-1, 1, 28, 28))))
        if show:
            a = out[0].detach()
            a.to('cpu')
            f, axarr = plt.subplots(*self.cs1)
            for i, tensor in enumerate(a):
                axarr[floor(i/self.cs1[1]), i%self.cs1[1]].imshow(tensor.view(12, 12), cmap='gray')
            plt.show()
        out = self.pool(self.relu(self.conv2(out)))
        if show:
            a = out[0].detach()
            a.to('cpu')
            f, axarr = plt.subplots(*self.cs2)
            for i, tensor in enumerate(a):
                axarr[floor(i/self.cs2[1]), i%self.cs2[1]].imshow(tensor.view(4, 4), cmap='gray')
            plt.show()
        out = self.relu(self.l1(out.view(-1, self.feature_size)))
        return out


net = Net((4, 5), (9, 9)).to('cuda')
optimizer = optim.Adam(net.parameters(), 0.0001)

scheduler = StepLR(optimizer, step_size=1, gamma=0.7)


def train(model, device, train_loader, optimizer, epoch, log_interval=200):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = mse_loss(output, to_categorical(target, 10))
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch+1} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({round(100 * batch_idx / len(train_loader))}%)]\tLoss: {loss.item()}')


for epoch in range(5):
    train(net, 'cuda', train_loader, optimizer, epoch, 100)
    scheduler.step()
    print()

running_count = 0
running_correct_count = 0
for i in test_loader:
    data, label = i

    output = np.argmax(net(data.to('cuda'), False).to('cpu').detach(), 1)

    correct_count = 0
    for t in output - label:
        if t == 0:
            correct_count += 1
    running_correct_count += correct_count
    running_count += 1
    print(round(100 * correct_count/test_loader.batch_size, 2))
    print(round(100 * running_correct_count/(running_count*test_loader.batch_size), 2))
    print()
