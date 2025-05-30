import matplotlib.pyplot as plt
# %matplotlib inline

import random
import torch

def synthetic_data(w,b, num_examples):
    """generate (label)y = X(features)w + b + noise"""
    X = torch.normal(0,1, (num_examples, len(w)))
    Y = torch.matmul(X,w) + b
    Y += torch.normal(0, 0.01, Y.shape)
    return X,  Y.reshape((-1,1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 100)

# print('features(x):', features[0], '\nlabel(Y):', labels)

# my_fig = plt.figure()
# plt.scatter(features[:,0].detach().numpy(), labels.detach().numpy(), 1)
# plt.show()

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i+batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

batch_size = 10

# for X, Y in data_iter(batch_size, features, labels):
#     print(X, "\n", Y)
#     break


"""initialize model parameters"""
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

def linreg(X, w, b):
    """linear regression model"""
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    """squared loss"""
    return (y_hat - y.reshape(y_hat.shape))**2 / 2

def sgd(params, lr, batch_size):
    """minibatch stochastic gradient descent"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

lr = 0.05
num_epochs = 5
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, Y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), Y) #compute l with respect to [w, b]
        l.sum().backward() #compute gradient on l with respect to [w, b]
        sgd([w, b], lr, batch_size) #update parameters using their gradient
    with torch.no_grad(): #train loss
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch+1}, loss {float(train_l.mean()):f}')
        