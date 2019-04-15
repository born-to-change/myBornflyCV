import torch
from torch.nn import init
from torch.autograd import Variable
t1 = torch.FloatTensor([1., 2.])
v1 = Variable(t1)
t2 = torch.FloatTensor([2., 3.])
v2 = Variable(t2)
v3 = v1 + v2
print('v3:', v3)
v3_detached = v3.detach()
v3_detached.data.add_(t1) # 修改了 v3_detached Variable中 tensor 的值
print(v3, v3_detached)    # v3 中tensor 的值也会改变
