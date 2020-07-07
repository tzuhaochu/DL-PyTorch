#矢量计算
import torch
from time import time

a = torch.ones(1000)
b = torch.ones(1000)

#逐位相加的时间
start = time()
c = torch.zeros(1000)
for i in range(1000):
    c[i] = a[i] + b[i]
print(time() - start)

#矢量相加的时间
start = time()
d = a + b
print(time() - start)

#复习广播机制
a = torch.ones(3)
b = 10
print(a + b)


#线性回归的实现
import torch
from IPython import display
from IPython.display import set_matplotlib_formats
import matplotlib.pyplot as plt
import numpy as np
import random

#生成数据
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs,
                       dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                       dtype=torch.float32)

print(features[0], labels[0])

#plt.rcParams['figure.figsize'] = (3.5, 2.5)
#plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
#plt.show()

#读取数据
def data_iter(bathc_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)       #对list重新排序
    for i in range(0, num_examples, bathc_size):
        j = torch.LongTensor(indices[i: min(i + bathc_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)

batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, y)
    break

#初始化权重和bias
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)

#设置允许autograd
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


def linreg(X, w, b):
    return torch.mm(X, w) + b  #矩阵相乘，非位乘


def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size


lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

#epoch为迭代周期
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()   #批量损失非标量，需要转为标量
        l.backward()
        sgd([w, b], lr, batch_size)

        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))

print(true_w, '\n', w)
print(true_b, '\n', b)