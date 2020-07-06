import torch

#empty为创建未初始化tensor，不是空tensor
x = torch.empty(5, 3)
print(x)

#rand为创建随机初始化的tensor
x = torch.rand(5, 3)
print(x)

#可以在创建的时候指定tensor类型
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

#可以直接根据数组创建tensor
x = torch.tensor([5.5, 3])
print(x)

x = x.new_ones(5, 3, dtype=torch.float64)
print(x)
#根据现有tensor创建
x = torch.rand_like(x, dtype=torch.float)
print(x)

#获取tensor的属性,返回的是个tuple
print(x.size())
print(x.shape)

#pytorch的加法操作
y = torch.rand(5, 3)
print(x + y)    #直接加
print(torch.add(x, y))   #通过库函数加

result = torch.empty(5, 3)
torch.add(x, y, out=result)   #通过库函数加并且指定输出流
print(result)

y.add_(x)   #在y上面自加
print(y)

#对tensor进行索引,切片共享内存
y = x[0, :]
y += 1
print(y)
print(x[0, :])


#改变tensor的形状，view还是共享data
y = x.view(15)
z = x.view(-1, 5)    #这边-1应该是不确定具体维数，自动计算得出
print(x.size(), y.size(), z.size())

x += 1
print(x)
print(y)

#进行深拷贝的方法clone，然后改变形状。。。。clone会将计算梯度传回源tensor
#有提供reshape的方法改变形状，但不能保证返回拷贝
x_cp = x.clone().view(15)
x -= 1
print(x)
print(x_cp)

#将一阶tensor转换为数
x = torch.randn(1)
print(x)
print(x.item())


#针对不一样形状的tensor会使用广播机制，复制成一样形状进行操作
x = torch.arange(1, 3).view(1, 2)
print(x)
y = torch.arange(1, 4).view(3, 1)
print(y)
print(x + y)

#可以根据id判断两个实例是否相同，view共享data，但是id不一致（有其他属性）
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y = y + x
print(id(y) == id_before)
#切片索引共享内存
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y[:] = y + x
print(id(y) == id_before)
#指定输出流到y，还是一样的内存
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
torch.add(x, y, out=y)
print(id(y) == id_before)

#tensor与numpy数组相互转化,numpy()和from_numpy()
a = torch.ones(5)
b = a.numpy()
print(a, b)

a += 1
print(a, b)
b += 1
print(a, b)

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
print(a, b)
a += 1
print(a, b)
b += 1
print(a, b)

#不共享内存的方法
c = torch.tensor(a)
a += 1
print(a, c)

#将tensor在cpu和gpu之间移动
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))