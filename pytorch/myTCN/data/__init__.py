import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
#a = np.arange(14).reshape(2,7)
# # length_of_sequences = map(len, a)
# #
# # for x in np.arange(len(a)):
# #     print(length_of_sequences.__next__())
#
# print(a.shape)
# print(len(a))

# weight = torch.Tensor([1,2,1,1,10])
# loss_fn = torch.nn.CrossEntropyLoss(reduce=False, size_average=False, weight=weight)
input = Variable(torch.randn(3, 5)) # (batch_size, C)
# target = Variable(torch.LongTensor(3).random_(5))
# loss = loss_fn(input, target)
# print(input); print(target); print(loss)


# (in_channels, out_channels, kernel_size, stride=1,padding=0, dilation=1, groups=1, bias=True):
m1 = nn.Conv1d(16, 33, 3, padding=1, dilation=1)
m2 = nn.Conv1d(16, 33, 3, dilation=1)
input = torch.randn(20, 16, 50)
print(input.transpose(2, 1).contiguous().view(-1, 16).shape)
print(input.shape)
# output1 = m1(input)
# output2 = m2(input)
# print()
# print(output1.shape)
# print(output2.shape)
# print(input.unsqueeze(0).shape)
#
# x = np.random.randn(100,100)
# y = np.random.randn(100,100)
# print(np.std(x))
# print((x+y).shape)
a = np.arange(12).reshape(3,2,2)
print(a)
print('-----')
print(a[:,:, :-1])
print('-----')
print(a[:, :,1:])