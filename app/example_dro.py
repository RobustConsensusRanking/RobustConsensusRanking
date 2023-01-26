import numpy as np
import torch
from scipy.optimize import linprog
from torch import argsort

from depthfunction.metrics.ranking_metrics import kendall_tau

m = kendall_tau

s_123 = torch.Tensor([0,1,2]).long()
s_132 = torch.Tensor([0,2,1]).long()
s_213 = torch.Tensor([1,0,2]).long()
s_231 = torch.Tensor([1,2,0]).long()
s_321 = torch.Tensor([2,1,0]).long()
s_312 = torch.Tensor([2,0,1]).long()

permuts = torch.row_stack([s_123, s_132, s_213, s_231, s_321, s_312])

L = torch.zeros((6,6))
for i in range(6):
    for j in range(6):
        L[i,j] = 3*m(permuts[i], permuts[j])
print(f"loss matrix = {L}")

def dro_value(p, eps, loss):
    c = np.concatenate([loss, np.zeros_like(loss)])
    A_eq = np.concatenate([np.ones_like(p), np.zeros_like(p)])[np.newaxis, :]
    b_eq = np.array([1.])
    I = np.eye(len(p))
    A_ub = np.concatenate([np.concatenate([-I, I, np.zeros((1,len(p)))]), np.concatenate([-I, -I, np.ones((1,len(p)))])], axis=1)
    b_ub = np.concatenate([-p, p, [eps]])

    res = linprog(-c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
    return np.round(-res.fun, 4)

p = torch.Tensor([
    0.551,    # 123
    0.0,    # 132 --
    0.449,    # 213 ++
    0.0,    # 231 ++
    0.0,    # 321 ++ --
    0.0     # 312 --
])
print(f"1=2>3 group proba={p[2]+p[3]+p[4]} // 1>2=3 group proba={p[1]+p[4]+p[5]}")

print("empirical loss", torch.matmul(p, L))

eps = 0.1
print("dro strict order", [dro_value(p, eps=eps, loss=L[:,i]) for i in range(6)])

pairs = [(0,1), (0,2), (1,5), (2, 3), (3,4), (4,5)]
print("dro weak order", [dro_value(p, eps=eps, loss=(L[:,i]+L[:,j])/2) for i,j in pairs])

print("full random", dro_value(p, eps=eps, loss=np.average(L, axis=-1)))
