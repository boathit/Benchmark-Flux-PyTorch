import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

trainData = torchvision.datasets.MNIST(root='./data',
                                       train=True,
                                       transform=transforms.ToTensor(),
                                       download=True)

testData = torchvision.datasets.MNIST(root='./data',
                                      train=False,
                                      transform=transforms.ToTensor())

trainLoader = torch.utils.data.DataLoader(dataset=trainData,
                                          batch_size=256,
                                          shuffle=True)

testLoader = torch.utils.data.DataLoader(dataset=testData,
                                         batch_size=256,
                                         shuffle=False)





class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 20, kernel_size=5),
                                    nn.MaxPool2d(2),
                                    nn.ReLU(),
                                    nn.Conv2d(20, 50, kernel_size=5),
                                    nn.MaxPool2d(2),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(800, 500),
                                    nn.ReLU(),
                                    nn.Linear(500, 10),
                                    nn.LogSoftmax(dim=1))
    def forward(self, x):
        h = self.layer1(x)
        o = self.layer2(h.reshape(-1, 800))
        return o

model = ConvNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train():
    for epoch in range(1, 11):
        for i, (x, y) in enumerate(trainLoader):
            (x, y) = x.to(device), y.to(device)
            o = model(x)
            loss = F.nll_loss(o, y)

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
