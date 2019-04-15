# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

# 对data求梯度, 用于反向传播
data = Variable(torch.FloatTensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]), requires_grad=True)
print('data',data)
# 多分类标签 one-hot格式
label = Variable(torch.zeros((3, 3)))
label[0, 2] = 1
label[1, 1] = 1
label[2, 0] = 1
print(label)

data2 = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
data2 = torch.Tensor(np.log(data2))
# for batch loss = mean( -sum(Pj*logSj) )
# for one : loss = -sum(Pj*logSj)
print(F.softmax(data, dim=1))
print(F.log_softmax(data, dim=1))

print(-torch.sum(label * torch.log(F.softmax(data, dim=1)), dim=1))
loss = torch.mean(-torch.sum(label * torch.log(F.softmax(data, dim=1)), dim=1))  # 对行维度求和

loss.backward()
print(loss, data.grad)

print(np.log10(0.09))
