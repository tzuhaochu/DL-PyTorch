import torch

#requires_grad=True可以追踪在该tensor上的所有操作
x = torch.ones(2, 2, requires_grad=True)    #直接创建的称为叶子节点
print(x)
print(x.grad_fn)    #grab_fn记录创建该tensor的function

y = x + 2
print(y)
print(y.grad_fn)

#判断是否为叶子节点
print(x.is_leaf, y.is_leaf)

z = y * y * 3
out = z.mean()
print(z, out)

a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

#out是标量，无需指定求导变量
out.backward()
print(out)
print(x.grad)

#grad在BP过程中是累加的，需要清0
out2 = x.sum()
out2.backward()
print(x.grad)

out3 = x.sum()
x.grad.data.zero_()
out3.backward()
print(x.grad)

x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
y = 2 * x
z = y.view(2, 2)
print(z)

v = torch.tensor([[1.0, 0.1],[0.01, 0.001]], dtype=torch.float)
print(z * v)
z.backward(v)
print(x.grad)

#这边没有grad为false是不会回传的
x = torch.tensor(1.0, requires_grad=True)
y1 = x ** 2
with torch.no_grad():
    y2 = x ** 3
y3 = y1 + y2

print(x.requires_grad)
print(y1, y1.requires_grad)
print(y2, y2.requires_grad)
print(y3, y3.requires_grad)

y3.backward()
print(x.grad)


#对tensor.data操作不会被记录在autograd中，但是还是会改变tensor的值
x = torch.ones(1, requires_grad=True)
print(x.data)
print(x.data.requires_grad)

y = 2 * x
x.data *= 100

y.backward()
print(x)
print(x.grad)