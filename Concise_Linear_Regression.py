import numpy as np
import torch
import torch.utils.data as Data
from torch import nn

def synthetic_data(w,b, num_examples):
    """generate (label)y = X(features)w + b + noise"""
    X = torch.normal(0,1, (num_examples, len(w)))
    Y = torch.matmul(X,w) + b
    Y += torch.normal(0, 0.01, Y.shape)
    return X,  Y.reshape((-1,1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

def load_array(data_arrays, batch_size, is_train=True):
    dataset = Data.TensorDataset(*data_arrays)
    return Data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10

# data_iter = load_array((features, labels), batch_size)
# next(iter(data_iter))

"""nn = neural network"""
net = nn.Sequential(nn.Linear(2,1))

net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

loss = nn.MSELoss()

trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in load_array((features, labels), batch_size):
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')