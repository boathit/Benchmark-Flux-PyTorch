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



model = nn.Sequential(nn.Linear(784, 400),
                      nn.ReLU(),
                      nn.Linear(400, 10),
                      nn.LogSoftmax(dim=1)).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train():
    for epoch in range(1, 11):
        for i, (x, y) in enumerate(trainLoader):
            (x, y) = x.to(device), y.to(device)
            x = x.reshape(-1, 784)
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
        x = x.reshape(-1, 784)
        o = model(x)
        _, ŷ = torch.max(o, 1)
        N += y.size(0)
        n += torch.sum(ŷ == y).item()
    print("Accuracy: {}".format(n/N))
