import torch 
import torch.nn as nn
import torchvision.datasets as dsets
from torch.utils.data import RandomSampler, DataLoader, Subset
import torchvision.transforms as transforms
import numpy as np
import time
import matplotlib.pyplot as plt


# Hyper Parameters
batch_size = 128
learning_rate = 0.001


transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor()
])


train_dataset = dsets.MNIST(root='./data/',
                               train=True, 
                               download=True,
                               transform=transform)
num_train_samples = 128
sample_ds = Subset(train_dataset, np.arange(num_train_samples))

test_dataset = dsets.MNIST(root='./data/',
                              train=False, 
                              download=True,
                              transform=transform)


# Data Loader (Input Pipeline)
train_loader = DataLoader(dataset=sample_ds,
                                           batch_size=batch_size, 
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=10000, 
                                          shuffle=False)


train_loader_iter = iter(train_loader)
data = next(train_loader_iter)
features, labels = data

test_loader_iter = iter(test_loader)
data_test = next(test_loader_iter)
features_test, labels_test = data_test

ber = np.random.binomial(size=128, n=1, p= 0.5)
ber_test = np.random.binomial(size=len(test_dataset), n=1, p= 0.5)

for i in range(len(labels)):
    labels[i] = ber[i]
    
for i in range(len(labels_test)):
    labels_test[i] = ber_test[i]
    
import torch.optim as optim
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
    
net = Net()
loss_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.005)

loss_arr = []
for epoch in range(20): 
    i = 0
    for data in train_loader:  
        X = data[i]
        i += 1
        y = labels
        net.zero_grad()  
        output = net(X.view(-1,784))  
        loss = loss_criterion(output, y)  
        loss.backward()  
        loss_arr.append(loss.detach().numpy())
        optimizer.step()   
        
loss_arr_test = []
for epoch in range(20): 
    i = 0
    for data in test_loader:  
        X_test = data[i]
        i += 1
        y_test = labels_test
        net.zero_grad()  
        output_test = net(X_test.view(-1,784))  
        loss_test = loss_criterion(output_test, y_test)  
        loss_arr_test.append(loss_test.detach().numpy())
        
plt.figure(figsize=(8,6))
plt.plot(range(20), loss_arr, c='blue')
plt.xlabel('epochs')
plt.ylabel('loss')

plt.plot(range(20), loss_arr_test, c='red')
plt.xlabel('epochs')
plt.ylabel('loss')

plt.xlim([0, 20])

plt.legend(['train', 'test'])

plt.show()