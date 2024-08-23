import torch
import numpy as np

''' 
torch.tensor()
   1.直接创建
   2.从其他Tensor创建
   3.和numpy的互相转换
'''
# x = torch.tensor([1, 2, 3])
# x1 = torch.randn(8, 5, 2)
# x2 = torch.ones_like(x1)
# x3 = torch.zeros_like(x1)
# data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# x4 = torch.from_numpy(data)

''' torch.sum '''
# dim就是被压缩的维度
# a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# b = torch.sum(a, dim=1)                                        # [6,15,24]       -> [3,]
# c = torch.sum(a, dim=0)                                        # [12,15,18]      -> [3,]
# d = torch.sum(a, dim=1, keepdim=True)                          # [[6],[15],[24]] -> [3,1]
# e = torch.sum(a, dim=0, keepdim=True)                          # [[12,15,18]]    -> [1,3]
# f = torch.sum(a, dim=(0, 1))     # 当 dim 是 list的时候          # [45]
# g = torch.sum(a)                 # 不添加维度默认全部维度进行整合

# # 可以举一些实际的例子，比如加上批次这样 # #
"""
torch中对tensor变形的函数
    torch.view() 相当于reshape
    torch.reshape()
    torch.stack()
    # 下面的函数是转置的功能,对tensor进行转置
    torch.transpose()    主要是对简单的tensor进行转置，交换两个维度
    torch.permute()      对复杂的tensor进行转置，交换多个维度
"""
''' torch.view() '''
# a = torch.arange(4 * 5 * 6).view(4, 5, 6)
#
# # 一个简单的例子
# x1 = torch.arange(0, 16)
# x2 = x1.view(2, 8)
# x3 = x1.view(8, 2)
# x4 = x1.view(4, 4)
#
# # 自动适应大小，这个比较常用
# # 二维的，三维的
# x7 = x1.view(2 * 4, -1)    # -1表示自动适应
# x5 = x1.view(2, 2, -1)
# x6 = x1.view(8, -1, 2)


''' torch.transpose() '''
# a = torch.arange(1, 17)
# x = torch.reshape(a, (8, 2, 1))
# x1 = x.transpose(0, 1)

''' torch.permute() '''
# a = torch.arange(1, 17)
# x = a.view(1, 8, 2)               # [1,8,2]
# x2 = torch.permute(x, (2, 0, 1))  # [2,1,8]

"""  
对两个以上的tensor进行拼接的函数
   tensor.stack()
   tensor.cat()
"""
''' torch.stack() '''

# T1 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # [3,3]
# T2 = torch.tensor([[10, 20, 30], [40, 50, 60], [70, 80, 90]])  # [3,3]
# x0 = torch.stack([T1, T2], dim=0)
# x1 = torch.stack([T1, T2], dim=1)
# x2 = torch.stack([T1, T2], dim=2)
# print(x0, x0.shape)
# print(x1, x1.shape)
# print(x2, x2.shape)

''' torch.cat() '''

x1 = torch.tensor([[11, 21, 31], [21, 31, 41]], dtype=torch.int)  # [2,3]
x2 = torch.tensor([[12, 22, 32], [22, 32, 42]], dtype=torch.int)  # [2,3]
x3 = torch.cat([x1, x2], dim=0)
x4 = torch.cat([x1, x2], dim=1)

print(x3)
print(x4)
