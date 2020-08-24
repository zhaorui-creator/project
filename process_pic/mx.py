import torch
import torch.nn.functional as F
#a = torch.tensor([[1,5,62,54], [2,6,2,6], [2,65,2,6]])
a =torch.rand((1,12,100,100))
avg_pool = F.avg_pool2d( a, a.size(2), stride=a.size(2) )
print(avg_pool)
print(avg_pool.shape)
#b = torch.max(a,1)[0]
#print(b)
#c = b[0]
#print(c)
