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
print(x.grad)