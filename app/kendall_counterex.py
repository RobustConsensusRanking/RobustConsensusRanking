import torch
from torch import argsort

from depthfunction.metrics.ranking_metrics import kendall_tau

metric = kendall_tau

s_123 = torch.Tensor([0,1,2]).long()
s_132 = torch.Tensor([0,2,1]).long()
s_213 = torch.Tensor([1,0,2]).long()
s_231 = torch.Tensor([1,2,0]).long()
s_321 = torch.Tensor([2,1,0]).long()
s_312 = torch.Tensor([2,0,1]).long()

a_123 = 0.45
a_231 = 0.45
a_321 = 0.1

def expected_kendall(sigma):
    return a_123 * metric(s_123, sigma) + a_231 * metric(s_231, sigma) + a_321 * metric(s_321, sigma)

print("123:", 60 * expected_kendall(s_123))
print("132:", 60 * expected_kendall(s_132))
print("213:", 60 * expected_kendall(s_213))
print("231:", 60 * expected_kendall(s_231))
print("321:", 60 * expected_kendall(s_321))
print("312:", 60 * expected_kendall(s_312))

n=3
r = torch.Tensor([1.,2.,3.])
g = 1. - torch.matmul((1./(n-r+1)), torch.triu(torch.ones([3,3])))
print('g: ', g)
g_0 = -(expected_kendall(s_123) * g[argsort(s_123)] \
      + expected_kendall(s_132) * g[argsort(s_132)] \
      + expected_kendall(s_213) * g[argsort(s_213)] \
      + expected_kendall(s_231) * g[argsort(s_231)] \
      + expected_kendall(s_312) * g[argsort(s_312)] \
      + expected_kendall(s_321) * g[argsort(s_321)])

print('g_0: ', g_0)
print((argsort(g_0, descending=True)+1.).long())
