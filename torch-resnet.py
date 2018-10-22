import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])

trainData = torchvision.datasets.CIFAR10(root="./data",
                                         train=True,
                                         transform=transform,
                                         download=True)

testData = torchvision.datasets.CIFAR10(root="./data",
                                        train=False,
                                        transform=transforms.ToTensor())

trainLoader = torch.utils.data.DataLoader(dataset=trainData,
                                          batch_size=256,
                                          shuffle=True)

testLoader = torch.utils.data.DataLoader(dataset=testData,
                                         batch_size=256,
                                         shuffle=False)

def conv3x3(num_in, num_out, stride):
    return nn.Conv2d(num_in, num_out, kernel_size=3,
                     stride=stride, padding=1, bias=False)

def align(num_in, num_out, stride):
    if num_in != num_out or stride > 1:
        return nn.Sequential(conv3x3(num_in, num_out, stride),
                             nn.BatchNorm2d(num_out))
    else:
        return lambda x: x

class ResBlock(nn.Module):
    def __init__(self, num_in, num_out, stride):
        super(ResBlock, self).__init__()
        self.align = align(num_in, num_out, stride)
        self.conv1 = conv3x3(num_in, num_out, stride)
        self.bn1 = nn.BatchNorm2d(num_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(num_out, num_out, 1)
        self.bn2 = nn.BatchNorm2d(num_out)

    def forward(self, x):
        o = self.conv1(x)
        o = self.bn1(o)
        o = self.relu(o)
        o = self.conv2(o)
        o = self.bn2(o)
        o = o + self.align(x)
        o = self.relu(o)
        return o

def buildResBlocks(num_in, num_out, stride, num_blocks):
    blocks = [ResBlock(num_in, num_out, stride)]
    for _ in range(1, num_blocks):
        blocks.append(ResBlock(num_out, num_out, 1))
    return nn.Sequential(*blocks)

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.blocks0 = nn.Sequential(conv3x3(3, 16, 1), nn.BatchNorm2d(16),
                                     nn.ReLU(inplace=True))
        self.blocks1 = buildResBlocks(16, 16, 1, 2)
        self.blocks2 = buildResBlocks(16, 32, 2, 2)
        self.blocks3 = buildResBlocks(32, 64, 2, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        n = x.shape[0]
        o = self.blocks0(x)
        o = self.blocks1(o)
        o = self.blocks2(o)
        o = self.blocks3(o)
        o = self.avgpool(o)
        o = self.fc(o.reshape(n, -1))
        return o

model = ResNet(10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train():
    for epoch in range(1, 11):
        for i, (x, y) in enumerate(trainLoader):
            (x, y) = x.to(device), y.to(device)
            o = model(x)
            loss = F.cross_entropy(o, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                 print("Epoch: {}\tLoss: {}".format(epoch, loss.item()))

tic = time.time()
train()
print(time.time()-tic, "s")


n, N = 0, 0
with torch.no_grad():
    for (x, y) in testLoader:
        (x, y) = x.to(device), y.to(device)
        o = model(x)
        _, ŷ = torch.max(o, 1)
        N += y.size(0)
        n += torch.sum(ŷ == y).item()
    print("Accuracy: {}".format(n/N))
