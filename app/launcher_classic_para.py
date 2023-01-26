import numpy as np
import os
from typing import NamedTuple
import copy
import torch
import itertools
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal, Normal
from torch.optim.lr_scheduler import LambdaLR, ConstantLR, CyclicLR
from torch.autograd import Variable
from scipy.optimize import root_scalar, shgo
import networkx as nx
from scipy import optimize
from math import factorial
from pyspark.sql import functions as ps_functions

import argparse

device = "cpu"
default_tensor_type = torch.FloatTensor

# Arguments
class Config(NamedTuple):
    n : int = 4
    exp_types: str = "unimodal"
    seed_vals: str = "938"
    ms: str = "0.1"
    delta_list: str = "1"
    epochs_list: str = "10000"
    dist_type_sym_list: str = "False"
    norm_type_list: str = "1"
    ratio_list: str = "1.0"
    nb_simu_training_list: str = "25"


def str2bool(value):
    if value in [True, "True", "true"]:
        return True
    else:
        return False


def get_config():
    parser = argparse.ArgumentParser(description="Parser")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--exp_types", type=str, default="unimodal")
    parser.add_argument("--seed_vals", type=str, default="938")
    parser.add_argument("--ms", type=str, default="0.1")
    parser.add_argument("--delta_list", type=str, default="1")
    parser.add_argument("--epochs_list", type=str, default="10000")
    parser.add_argument("--dist_type_sym_list", type=str, default="False")
    parser.add_argument("--norm_type_list", type=str, default="1")
    parser.add_argument("--ratio_list", type=str, default="1.0")
    parser.add_argument("--nb_simu_training_list", type=str, default="25")

    args, _ = parser.parse_known_args()

    args.exp_types = list(map(str, str(args.exp_types).split(";")))
    args.seed_vals = list(map(int, str(args.seed_vals).split(";")))
    args.ms = list(map(float, str(args.ms).split(";")))
    args.delta_list = list(map(int, str(args.delta_list).split(";")))
    args.epochs_list = list(map(int, str(args.epochs_list).split(";")))
    args.dist_type_sym_list = list(map(str2bool, str(args.dist_type_sym_list).split(";")))
    args.norm_type_list = list(map(str, str(args.norm_type_list).split(";")))
    args.ratio_list = list(map(float, str(args.ratio_list).split(";")))
    args.nb_simu_training_list = list(map(int, str(args.nb_simu_training_list).split(";")))

    max_len = np.max((len(args.exp_types),len(args.seed_vals),len(args.ms),len(args.delta_list),len(args.epochs_list),len(args.dist_type_sym_list),len(args.norm_type_list),len(args.ratio_list),len(args.nb_simu_training_list)))
    args.exp_types = args.exp_types*max_len
    args.seed_vals = args.seed_vals* max_len
    args.delta_list = args.delta_list* max_len
    args.epochs_list = args.epochs_list* max_len
    args.dist_type_sym_list = args.dist_type_sym_list* max_len
    args.norm_type_list = args.norm_type_list* max_len
    args.ratio_list = args.ratio_list* max_len
    args.nb_simu_training_list = args.nb_simu_training_list * max_len

    return args.__dict__ #Config(**args.__dict__), args.__dict__


# Generating ranks - basic parameters
n_items = 4
all_ranks = list(itertools.permutations(list(np.arange(n_items))))
all_ranks = [np.array(elem) for elem in all_ranks]
torch_all_ranks = torch.from_numpy(np.asarray(all_ranks))


def pairwise_marginals_cached(p, all_pairwises):
    return np.tensordot(p, all_pairwises, axes=(0,0))

def pairwise_preference(sigma):
    inv = np.argsort(sigma)[:,np.newaxis]
    return (inv < inv.T).astype(float)

def pairwise_matrix(p_torch, torch_all_ranks, n=4):
    all_rank = torch_all_ranks.detach().numpy()
    all_pairwises = np.array([pairwise_preference(sigma) for sigma in all_rank])
    if torch.is_tensor(p_torch):
        p = p_torch.squeeze().detach().numpy()
    else:
        p = p_torch.squeeze()
    M = pairwise_marginals_cached(p, all_pairwises)
    return torch.tensor(M)

def expected_kendall(P1, P2):
    P1 = P1 - torch.diag(torch.diag(P1))
    P2 = P2 - torch.diag(torch.diag(P2))
    return torch.norm(P1 * (1-P2), 1)

def symmetrized_hausdorff_on_kendall(P1, P2):
    return torch.norm(torch.triu(P1,1) - torch.triu(P2,1), 1)

def disymmetrized_hausdorff_on_kendall(P1, P2):
    # If P2 is included in P1
    if (P1[P1 != 1/2] == P2[P1 != 1/2]).all():
        return torch.tensor(0.0)
    # If P1 is included in P2
    elif (P1[P2 != 1/2] == P2[P2 != 1/2]).all():
        return 2*torch.norm(torch.triu(P1,1) - torch.triu(P2,1), 1)
    else:
        idxs = torch.argwhere(P1 == 1/2)
        v = torch.sum(P2 == 1/2)/2
        idxs2 = torch.argwhere(P2 == 1/2)
        P1[[idxs[:, 0], idxs[:, 1]]] = 0
        P2[[idxs[:, 0], idxs[:, 1]]] = 0
        P1[[idxs2[:, 0], idxs2[:, 1]]] = 0
        P2[[idxs2[:, 0], idxs2[:, 1]]] = 0
        return torch.norm(torch.triu(P1,1) - torch.triu(P2,1), 1) + v


def smooth_pg_2(m, k_x):
    r"""
    Smooth a metric $m$ by convolution with a kernel $k_x$ centered in $x$.
    $$\tilde{m}(y,x) = \int m(y,u) k_x(u) du$$
    As the gradient is not available in closed-form, the loss build is a policy gradient loss, that leads to a noisy but
    unbiased estimate of the gradient of $\tilde{m}$.
    $$g_x(y,u) = \log(k_x(u)) f(y,u) ~~~~\text{for}~~~ u\sim k_x$$
    :param m: function to be smoothed
    :param k_x: function $x\mapsto k$ where $k$ is a kernel centered in $x$
    :return: $g_x(y,u)$ for $u\sim k_x$
    """

    def smoothed_m_pg(y, x):
        k = k_x(x)
        batch_size = torch.Size()
        if len(y.size()) > 1:
            batch_size = y.size()[:-1]
        u = k.sample(batch_size).to(device).type(default_tensor_type)
        loss = k.log_prob(u) * m(y, u)
        return loss

    return smoothed_m_pg

def monte_carlo_phi(p, q, dist_Tp_Tq, std_dev_=0.00001, nb_simu=50):
    kernel_conv = lambda _s: MultivariateNormal(_s, std_dev_*torch.eye(_s.size()[-1]))
    rhos = list()
    for i in range(nb_simu):
        q2 = kernel_conv(q.float()).sample(torch.Size())
        rho = dist_Tp_Tq(p, q2.unsqueeze(0)).detach().numpy().item()
        rhos.append(rho)
    rho_final = np.mean(rhos)
    return torch.tensor(rho_final)

class NewPhi(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    def __init__(self):
        self.name = "New Phi class"

    @staticmethod
    def forward(ctx, input_q2, input_backward, p_torch, dist_Tp_Tq, std_dev_, nb_simu):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input_backward)
        monte_carlo_val = monte_carlo_phi(p_torch, input_q2, dist_Tp_Tq, std_dev_=std_dev_, nb_simu=nb_simu)
        return monte_carlo_val

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        back_,  = ctx.saved_tensors
        return back_ * grad_output, None, None, None, None, None


def torch_optim_attack_relaxed(dist_Tp_Tq, p_torch, reg, epochs, std_dev=0.01, lr=0.01):
    r"""
    :param method: $\tau_K(T(p), T(q))$
    :param reg: $\lambda$
    :param epochs: limit on the number of optimization iterations
    """
    p_torch2 = p_torch.detach().clone()
    p_torch2.requires_grad = False
    softmax = torch.nn.Softmax(dim=0)
    softplus = torch.nn.Softplus(beta=1, threshold=20)
    kernel_conv = lambda _s: MultivariateNormal(_s, std_dev * torch.eye(_s.size()[-1]))  # $x\mapsto k_x(.)$
    smoothed_dist_Tp_Tq = lambda p, q: smooth_pg_2(dist_Tp_Tq, kernel_conv)(p, q)  # $\phi(.,.)$

    s_ = p_torch[0, :].detach().clone().to(device).type(default_tensor_type)
    s_.requires_grad = True
    optimizer = torch.optim.SGD([s_], lr=0.01, momentum=0.9)
    scheduler = CyclicLR(optimizer, 0.01, 1.0, step_size_down=50, cycle_momentum=False)

    for epoch in range(epochs):
        optimizer.zero_grad()
        q = softmax(s_)  # projection $\sum = 1$

        # Decrease the approximation error over time
        kernel_conv = lambda _s: MultivariateNormal(_s, std_dev * (10 / (10 + epoch)) * torch.eye(
            _s.size()[-1]))  # $x\mapsto k_x(.)$
        smoothed_dist_Tp_Tq = lambda p, q: smooth_pg_2(dist_Tp_Tq, kernel_conv)(p, q)  # $\phi(.,.)$

        loss = -smoothed_dist_Tp_Tq(p_torch2, q) + reg * (
            torch.norm(p_torch2 - q, 1))  # Lagrangian relaxation of the objective
        loss.backward()
        optimizer.step()
        scheduler.step()

    return q


def subsample_fct(vect, time=10):
    l = len(vect)
    res = vect[0:l:time]
    return res


def custom_scheduler(my_optimizer, lr, grad_data):
    if (torch.abs(grad_data) > 20).any():
        my_optimizer.param_groups[0]['lr'] = lr / 100
    elif (torch.abs(grad_data) > 1.5).any():
        my_optimizer.param_groups[0]['lr'] = lr * torch.log(torch.mean(torch.abs(grad_data))) / 10
    elif (torch.abs(grad_data) > 0.5).any():
        my_optimizer.param_groups[0]['lr'] = lr * torch.mean(torch.abs(grad_data)) / 1
    else:
        my_optimizer.param_groups[0]['lr'] = 10 * lr
    return my_optimizer


def plot_end_training(p_torch, dist_Tp_Tq, norms_, losses, mean_qs_, phis_, mean_phi2_, lambdas_, mean_lambdas_,
                      grad_data, freq_phi_, norm_type):
    f, ax = plt.subplots(2, 2, figsize=(15, 8))
    ax[0, 0].plot(norms_)
    # ax[0,0].set_ylim([-0.05, 2.05])
    ax[0, 0].set_title(f"Norm diff btw p and q")

    ax[0, 1].plot(phis_)
    ax[0, 1].set_ylim([-0.05, 6.05])
    ax[0, 1].set_title(f"Phi")

    ax[1, 0].plot(lambdas_)
    # ax[1,0].set_ylim([-0.05, 8.05])
    ax[1, 0].set_title(f"Lambda")

    ax[1, 1].plot(losses)
    # ax[1,1].set_ylim([-10.05, 10.05])
    ax[1, 1].set_title(f"Loss")
    plt.show()

    time_fct = 100
    x_axis_val = subsample_fct(np.linspace(0, len(mean_lambdas_), len(mean_lambdas_)), time=time_fct)
    f, ax = plt.subplots(2, 2, figsize=(15, 8))

    mean_qs_subsample = subsample_fct(mean_qs_, time_fct)
    mean_qs1_ = [torch.norm(p_torch - torch.tensor(mean_q_), int(norm_type)) for mean_q_ in
                 mean_qs_subsample]  # [mean_q_[0] for mean_q_ in mean_qs_]
    ax[0, 0].plot(x_axis_val, mean_qs1_)
    # ax[0,0].set_ylim([-0.05, 1.05])
    ax[0, 0].set_title(f"Norm of the difference between p and q")

    Phi_ = NewPhi.apply
    mean_phis_ = [Phi_(torch.tensor(mean_q_), grad_data, p_torch, dist_Tp_Tq, 1e-7, 100) for mean_q_ in
                  mean_qs_subsample]
    ax[0, 1].plot(x_axis_val, mean_phis_, label="from mean q")
    ax[0, 1].plot(x_axis_val, subsample_fct(mean_phi2_, time_fct), label="from phis")
    ax[0, 1].legend()
    ax[0, 1].set_ylim([-0.05, 6.05])
    ax[0, 1].set_title(f"Phi")

    ax[1, 0].plot(x_axis_val, subsample_fct(mean_lambdas_, time_fct))
    ax[1, 0].set_title(f"Lambda")

    ax[1, 1].plot(x_axis_val, subsample_fct(losses, time_fct))
    ax[1, 1].set_title(f"Loss")
    plt.plot()



def approximate_breakdown_function(delta, dist_Tp_Tq, p_torch, epochs=150000, std_dev=0.01, lr=0.01, maxiter=10,
                                   max_reg=10., eval_strat="real", norm_type="1", ratio_norm_=1.0, nb_simu_training=25):
    softmax = torch.nn.Softmax(dim=0)
    softplus = torch.nn.Softplus(beta=1, threshold=20)
    # q = torch_optim_attack_relaxed(dist_Tp_Tq, p_torch, epochs=20, std_dev=std_dev, reg=10, lr=0.01)
    # q2 = q.detach().clone()
    q2 = p_torch.detach().clone().squeeze(0)  # q.detach().clone() #
    q2.requires_grad = True
    s_ = q2[:].detach().clone().to(device).type(default_tensor_type)
    s_.requires_grad = True

    lambda_ = torch.tensor((10.0,), requires_grad=True)
    qs_ = list()
    qs_total_ = list()
    lambdas_ = list()
    norms_ = list()

    mean_qs_ = list()
    mean_phi2_ = list()
    mean_lambdas_ = list()

    freq_phi_ = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

    phis_ = list()
    losses = list()

    lr_list = list()

    optimizer_q2 = torch.optim.SGD([s_], lr=lr, weight_decay=0.01, momentum=0.1)
    optimizer_lambda = torch.optim.SGD([lambda_], lr=lr, weight_decay=0.01, momentum=0.1, maximize=True)

    scheduler_q2 = torch.optim.lr_scheduler.LinearLR(optimizer_q2, start_factor=1, end_factor=1 / 20, total_iters=30000)
    scheduler_lambda = torch.optim.lr_scheduler.LinearLR(optimizer_lambda, start_factor=1, end_factor=1 / 20,
                                                         total_iters=30000)

    for epoch in range(epochs):
        try:
            std_dev_training = 1e-5
            kernel_conv = lambda _s: MultivariateNormal(_s, std_dev_training * torch.eye(_s.size()[-1]))
            q2 = softplus(s_) / torch.sum(softplus(s_))
            res = smooth_pg_2(dist_Tp_Tq, kernel_conv)(p_torch, q2)
            res.backward(retain_graph=True)
            grad_data = s_.grad.data.detach().clone()
            Phi_ = NewPhi.apply
            phi_ = Phi_(q2, grad_data, p_torch, dist_Tp_Tq, std_dev_training, nb_simu_training)

            # Norm: L1, L2, or something between the two
            norm_ = ratio_norm_ * torch.norm(p_torch - torch.unsqueeze(q2, 0), int(norm_type)) + (
                        1.0 - ratio_norm_) * torch.norm(p_torch - torch.unsqueeze(q2, 0),
                                                        2 * int(norm_type) - 3 * (int(norm_type) - 1))

            optimizer_q2.zero_grad()
            optimizer_lambda.zero_grad()

            # LOSS
            loss = norm_ + lambda_ * delta - lambda_ * phi_

            loss.backward()
            optimizer_q2.step()
            optimizer_lambda.step()
            scheduler_q2.step()
            scheduler_lambda.step()

            with torch.no_grad():
                lambda_[:] = lambda_.clamp(min=0.0, max=None)

            losses.append(loss.detach().item())
            qs_total_.append(q2.detach().numpy())
            norms_.append(norm_.item())

            phis_.append(phi_.item())
            lambdas_.append(lambda_.detach().item())

            if phi_.item() in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
                freq_phi_[np.round(phi_.item())] += 1
            lr_list.append(optimizer_q2.param_groups[0]['lr'])

            if epoch == 0:
                mean_qs_.append(qs_total_[-1])
                mean_phi2_.append(phis_[-1])
                mean_lambdas_.append(lambdas_[-1])
            else:
                mean_qs_.append((epoch + 1) / (epoch + 2) * mean_qs_[-1] + 1 / (epoch + 2) * qs_total_[
                    -1])  # mean_qs_.append(np.mean(qs_total_[0:], axis=0))
                mean_lambdas_.append((epoch + 1) / (epoch + 2) * mean_lambdas_[-1] + 1 / (epoch + 2) * lambdas_[
                    -1])  # mean_etas_.append(np.mean(etas_[0:]))
                mean_phi2_.append((epoch + 1) / (epoch + 2) * mean_phi2_[-1] + 1 / (epoch + 2) * phis_[-1])
        except:
            break

    return norms_, losses, s_, qs_total_, mean_qs_, phis_, mean_phi2_, lambdas_, mean_lambdas_, grad_data, freq_phi_  # losses, qs_, lambdas_, s_, mean_qs_, mean_lambdas_ #epsilons_, etas_, alphas_, s_, mean_qs_, mean_epsilons_, mean_etas_, mean_alphas_



def proba_plackett_luce(w, all_ranks, n_items=4):
    list_proba = list()

    for sigma in all_ranks:
        val_ = list()
        for r in range(n_items):
            num_ = w[sigma[r]]
            denom_ = 0
            for s in range(r, n_items):
                v_ = w[sigma[s]]
                denom_ += v_
            val_.append(num_/denom_)
        proba_ = np.prod(val_)
        list_proba.append(proba_)
    return np.matrix(list_proba)


def _maxpair(P, threshold=0.):
    s = torch.sum(1 * (P > 0.5) + 1 / 2 * (P == 0.5), axis=1)
    sigma = torch.argsort(-s)
    M = P[np.ix_(sigma, sigma)]

    idxs = torch.argwhere(M > 1 / 2 + threshold)
    for idx in idxs:
        M[:idx[0] + 1, idx[1]] = 1
        M[idx[1], :idx[0] + 1] = 0

    m = torch.max(torch.abs(M - 0.5) * (M != 0.5) * (torch.abs(M - 0.5) <= threshold))
    while m > 0:
        i, j = torch.argwhere(np.abs(M - 0.5) == m)[0]
        if i <= j:
            idx_tomerge1, idx_tomerge2 = i, j + 1
        elif i > j:
            idx_tomerge1, idx_tomerge2 = j, i + 1
        M = torch_merge_idx(M, torch.arange(idx_tomerge1, idx_tomerge2))
        m = torch.max(np.abs(M - 0.5) * (M != 0.5) * (torch.abs(M - 0.5) <= threshold))
    return M[np.ix_(torch.argsort(sigma), torch.argsort(sigma))]


def maxpair(p, torch_all_ranks, n=4, threshold=0.):
    P = pairwise_matrix(p, torch_all_ranks=torch_all_ranks, n=n)
    return _maxpair(P, threshold=threshold)


# @torch.jit.script #@njit
def torch_merge_idx(M, idx):
    # i,j \notin idx -> P_ij = M_ij
    # i \in idx, j \notin idx -> P_ij = \max_{k\in idx} M_kj
    # i \notin idx, j \in idx -> P_ij = \max_{k\in idx} M_ik
    # i,j \in idx -> P_ij = 0.5
    P = M
    for i in torch.arange(M.shape[0]):
        m = torch.max(M[i, idx])
        for j in idx:
            P[i,j] = m
    for j in torch.arange(M.shape[0]):
        m = torch.max(M[idx, j])
        for i in idx:
            P[i,j] = m
    for i in idx:
        for j in idx:
            P[i,j] = 0.5
    PTRIU = torch.triu(P, 0)
    P = PTRIU + torch.tril(1 - PTRIU.T, -1)
    return P

#@torch.jit.script #@njit
def merge(p, torch_all_ranks, threshold=0, n=4):
    P = pairwise_matrix(p, torch_all_ranks=torch_all_ranks, n=n)
    cont = True
    while cont:
        P_mod = torch.abs(torch.triu(P,1)-torch.tensor(0.5))
        m = torch.min(torch.abs(torch.triu(P,1)-torch.tensor(0.5)))
        if m == 0.0 or m > threshold:
            cont = False
        else:
            idxs = torch.argwhere(P_mod == m)
            idxs = idxs.reshape(torch.prod(torch.tensor(idxs.shape)))
            P = torch_merge_idx(P, idxs)
    return P


def erm(p, torch_all_ranks, n=4):
    P = pairwise_matrix(p, torch_all_ranks=torch_all_ranks, n=n)
    return torch.round(P)


def kendall_tau(sigma1, sigma2):
    n = sigma1.size()[-1]
    sigma1_inv = torch.argsort(sigma1, dim=-1)
    sigma2_inv = torch.argsort(sigma2, dim=-1)
    sigma1_pairs = (sigma1_inv.unsqueeze(dim=-1) > sigma1_inv.unsqueeze(dim=-2)).float()
    sigma2_pairs = (sigma2_inv.unsqueeze(dim=-1) > sigma2_inv.unsqueeze(dim=-2)).float()
    return torch.sum(torch.abs(sigma1_pairs - sigma2_pairs), dim=[-2, -1]).double() / 2  # /(n*(n-1))


def kendall_matrix(torch_all_ranks):
    K = torch.zeros(len(torch_all_ranks), len(torch_all_ranks))
    for i, rank1 in enumerate(torch_all_ranks):
        for j, rank2 in enumerate(torch_all_ranks):
            K[i, j] = kendall_tau(rank1, rank2)
    return K.double()


def get_all_buckets(torch_all_ranks, n=4):
    list_bucket = list()

    for rank1 in torch_all_ranks:
        for rank2 in torch_all_ranks:
            if kendall_tau(rank1, rank2) == 1.0:
                list_bucket.append([rank1, rank2])

    for i in np.arange(n):
        temp_ranks = list()
        for rank in torch_all_ranks:
            if rank[3] == i:
                temp_ranks.append(rank)
        list_bucket.append(temp_ranks)

    list_bucket.append([torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 3, 2]), torch.tensor([1, 0, 2, 3]),
                        torch.tensor([1, 0, 3, 2])])
    list_bucket.append([torch.tensor([0, 2, 1, 3]), torch.tensor([0, 2, 3, 1]), torch.tensor([2, 0, 1, 3]),
                        torch.tensor([2, 0, 3, 1])])
    list_bucket.append([torch.tensor([0, 3, 1, 2]), torch.tensor([0, 3, 2, 1]), torch.tensor([3, 0, 1, 2]),
                        torch.tensor([3, 0, 2, 1])])
    list_bucket.append([torch.tensor([1, 2, 0, 3]), torch.tensor([1, 2, 3, 0]), torch.tensor([2, 1, 0, 3]),
                        torch.tensor([2, 1, 3, 0])])
    list_bucket.append([torch.tensor([1, 3, 0, 2]), torch.tensor([1, 3, 2, 0]), torch.tensor([3, 1, 0, 2]),
                        torch.tensor([3, 1, 2, 0])])
    list_bucket.append([torch.tensor([2, 3, 0, 1]), torch.tensor([2, 3, 1, 0]), torch.tensor([3, 2, 0, 1]),
                        torch.tensor([3, 2, 1, 0])])

    list_bucket.append(torch_all_ranks)
    for rank in torch_all_ranks:
        list_bucket.append(list([rank]))

    return list_bucket


def bucket_distrib(torch_all_ranks, list_bucket):
    l_ = list()
    div = len(list_bucket)
    for a_ in torch_all_ranks:
        count = 0
        for b_ in list_bucket:
            count += 1
            if (b_ == a_).all():
                l_.append(1.0 / div)
                count += 1
                continue
        if count == div:
            l_.append(0)
    return torch.tensor(l_).double()


def depth_metric_optim(p, K, list_bucket, torch_all_ranks, norm="l1", printer=False):
    n_ranks_to_test = len(list_bucket)
    val = torch.inf
    for i, bucket in enumerate(list_bucket):
        q = bucket_distrib(torch_all_ranks, bucket)
        if norm == "l1":
            val_ = torch.norm(torch.matmul(p.double(), K) - torch.matmul(q.double(), K), 1)
        elif norm == "l2":
            val_ = torch.norm(torch.matmul(p.double(), K) - torch.matmul(q.double(), K), 2)
        if val_ <= val:
            best_distrib = q
            val = val_
    return best_distrib, val


def depth(p, torch_all_ranks, norm="l1", printer=False):
    list_bucket = get_all_buckets(torch_all_ranks, n=4)
    K = kendall_matrix(torch_all_ranks)
    q, val = depth_metric_optim(p, K, list_bucket, torch_all_ranks, norm=norm, printer=printer)
    Q = pairwise_matrix(q.unsqueeze(0), torch_all_ranks=torch_all_ranks, n=4)
    return Q


def wasserstein_dist(p, q, K, n=4):
    nsize = factorial(n)
    K_bis = K.view(nsize * nsize)

    for i in range(nsize):
        A_ = [0] * nsize * i + [1] * nsize + [0] * nsize * (nsize - i - 1)
        if i == 0:
            A = torch.tensor(A_).reshape(1, nsize * nsize)
        else:
            A = torch.cat((A, torch.tensor(A_).reshape(1, nsize * nsize)))
    for i in range(nsize):
        A_ = ([0] * i + [1] + [0] * (nsize - i - 1)) * nsize
        A = torch.cat((A, torch.tensor(A_).reshape(1, nsize * nsize)))

    b = torch.cat((p, q), 1)

    optim_val = optimize.linprog(K_bis, A_eq=A, b_eq=b, bounds=(0, 1))
    return optim_val.fun


def wasserstein_(myp, torch_all_ranks, n=4, printer=False):
    list_bucket = get_all_buckets(torch_all_ranks, n=4)
    K = kendall_matrix(torch_all_ranks)
    val = torch.inf
    for i, bucket in enumerate(list_bucket):
        q = bucket_distrib(torch_all_ranks, bucket)
        if len(q.shape) < 2:
            q = q.unsqueeze(0)
        val_ = wasserstein_dist(myp, q, K, n=n)
        if val_ <= val:
            best_distrib = q
            val = val_
    return best_distrib, val


def wasserstein(myp, torch_all_ranks, n=4):
    q, val = wasserstein_(myp, torch_all_ranks, n=n, printer=False)
    Q = pairwise_matrix(q, torch_all_ranks=torch_all_ranks, n=n)
    return Q








#### LAUNCHER FUNCTIONS
def get_automatic_thresholds(w):
    p = proba_plackett_luce(w, all_ranks)
    p_torch = torch.from_numpy(p)
    torch_all_ranks = torch.from_numpy(np.asarray(all_ranks))
    P = pairwise_matrix(p_torch, torch_all_ranks, n=4)

    m = torch.flatten(torch.triu(P, 1))
    m = torch.sort(m[m != 0.0])[0]
    minval = (m - 0.5)[0]
    maxval = (m - 0.5)[-1]
    thresholds_ = [0]
    for val in list(np.linspace(minval - 0.0005, maxval + 0.0005, 21)):
        thresholds_.append(val)
    return thresholds_


def torch_dist(dist, p_torch1, p_torch2, torch_all_ranks, threshold, dist_type_sym):
    if dist == "erm":
        R1 = erm(p_torch1, torch_all_ranks, n=4)
        R2 = erm(p_torch2, torch_all_ranks, n=4)
    elif dist == "maxpair":
        R1 = maxpair(p_torch1, torch_all_ranks, n=4, threshold=threshold)
        R2 = maxpair(p_torch2, torch_all_ranks, n=4, threshold=threshold)
    elif dist == "merge":
        R1 = merge(p_torch1, torch_all_ranks, threshold=threshold, n=4)
        R2 = merge(p_torch2, torch_all_ranks, threshold=threshold, n=4)
    elif dist == "depth":
        R1 = depth(p_torch1, torch_all_ranks, norm="l1", printer=False)
        R2 = depth(p_torch2, torch_all_ranks, norm="l1", printer=False)
    elif dist == "wasserstein":
        R1 = wasserstein(p_torch1, torch_all_ranks, n=4)
        R2 = wasserstein(p_torch2, torch_all_ranks, n=4)
    if dist_type_sym:
        return symmetrized_hausdorff_on_kendall(R1, R2)
    else:
        return disymmetrized_hausdorff_on_kendall(R1, R2)


def launch_exp(dist, p_torch, w, delta, thresholds_, epochs, save=True, dist_type_sym=True, norm_type="1",
               ratio_norm_=1.0, nb_simu_training=25, exp_type="unimodal"):
    torch_all_ranks = torch.from_numpy(np.asarray(all_ranks))
    P = pairwise_matrix(p_torch, torch_all_ranks, n=4)

    eps_list1 = []
    eps_list2 = []
    alt_eps_list1 = []
    alt_eps_list2 = []
    perf_list = []
    if dist in ["erm", "depth", "wasserstein"]:
        thresholds = [0.0]
    elif dist in ["maxpair", "merge"]:
        thresholds = thresholds_

    dict_training = dict()

    for i_, threshold in enumerate(thresholds):
        if threshold >= 0.4:
            div_ = 10
        else:
            div_ = 1
        dist_Tp_Tq = lambda _p, _q: torch_dist(dist, _p, _q, torch_all_ranks, threshold=threshold,
                                               dist_type_sym=dist_type_sym)

        norms, losses, s_, qs_, mean_qs, phis, mean_phi2, lambdas, mean_lambdas, grad_data, freq_phi = approximate_breakdown_function(
            delta - 1e-6, dist_Tp_Tq, p_torch, epochs=epochs, std_dev=0.001, lr=0.025 / div_, maxiter=21,
            eval_strat="smoothed", norm_type=norm_type, ratio_norm_=ratio_norm_, nb_simu_training=nb_simu_training)
        #plot_end_training(p_torch, dist_Tp_Tq, norms, losses, mean_qs, phis, mean_phi2, lambdas, mean_lambdas,
        #                  grad_data, freq_phi, norm_type)
        dict_res_training = {"norms": norms, "losses": losses, "s_": s_, "qs_": qs_, "mean_qs": mean_qs, "phis": phis,
                             "mean_phi2": mean_phi2, "lambdas": lambdas, "mean_lambdas": mean_lambdas,
                             "grad_data": grad_data, "freq_phi": freq_phi}
        dict_training[threshold] = dict_res_training

        # q1 is the mean of the last q found
        qlist_ = qs_[epochs - 1000:]
        q1 = np.mean(qlist_, axis=0)

        # q2 is the overall mean
        q2 = mean_qs[-1]

        eps_list1.append(torch.norm(p_torch - torch.tensor(q2).unsqueeze(0), 1))
        eps_list2.append(torch.norm(p_torch - torch.tensor(q2).unsqueeze(0), 2))

        alt_eps_list1.append(torch.norm(p_torch - torch.tensor(q1).unsqueeze(0), 1))
        alt_eps_list2.append(torch.norm(p_torch - torch.tensor(q1).unsqueeze(0), 2))

        if dist == "erm":
            Ptilde = erm(p_torch, torch_all_ranks, n=4)
        elif dist == "maxpair":
            Ptilde = maxpair(p_torch, torch_all_ranks, n=4, threshold=threshold)
        elif dist == "merge":
            Ptilde = merge(p_torch, torch_all_ranks, threshold=threshold, n=4)
        elif dist == "depth":
            Ptilde = depth(p_torch, torch_all_ranks, norm="l1", printer=False)
        elif dist == "wasserstein":
            Ptilde = wasserstein(p_torch, torch_all_ranks, n=4)
        exp_kendall = expected_kendall(Ptilde, P).detach().item()
        perf_list.append(exp_kendall)


    filename = f"perf_robustness_dist={dist}_w={w}_delta={delta}_epochs={epochs}_dist_type_sym={dist_type_sym}_norm_L{norm_type}_ratio_{ratio_norm_}.pt"
    final_val_dict = {"thresholds": thresholds, "perf_list": perf_list, "eps_list1": eps_list1,
                          "eps_list2": eps_list2, "alt_eps_list1": alt_eps_list1, "alt_eps_list2": alt_eps_list1,
                          "p_torch": p_torch}
    res_dict = {"training_dict": dict_training, "final_val_dict": final_val_dict}

    return res_dict


def check_SST(P):
    P = P.numpy()
    cycles = list()
    P2 = np.nan_to_num((P - 1 / 2).clip(0))
    P2
    G = nx.from_numpy_matrix(P2, create_using=nx.MultiDiGraph())
    for cycle in nx.simple_cycles(G):
        cycles.append(cycle)
    if len(cycles) < 1:
        res = True
    else:
        res = False
    return res


def get_multimodal(ws, ratios, all_ranks):
    torch_all_ranks = torch.from_numpy(np.asarray(all_ranks))
    p_torch = torch.zeros((1, 24))
    for (w, ratio) in zip(ws, ratios):
        p_ = proba_plackett_luce(w, all_ranks)
        p_torch_ = torch.from_numpy(p_)
        p_torch += ratio * p_torch_
    P = pairwise_matrix(p_torch, torch_all_ranks, n=4)
    is_sst = check_SST(P)

    return p_torch, P


def main_exp_launcher(config):
    config = json.loads(config)[0]
    res_dict_general = dict()

    for seed_val, exp_type, delta, epochs, dist_type_sym, norm_type, ratio_norm_, nb_simu_training in zip(config["seed_vals"], config["exp_types"], config["delta_list"], config["epochs_list"], config["dist_type_sym_list"], config["norm_type_list"], config["ratio_list"], config["nb_simu_training_list"]):
        print("WE ARE IN THIS LOOP")
        torch.manual_seed(seed_val)
        mykey = f"{exp_type}_seed={seed_val}_epochs={epochs}_delta={delta}_sym_dist={dist_type_sym}"
        res_dict_general[mykey] = dict()

        for m in config["ms"]:
            print("WE ARE IN THIS OTHER LOOP")
            if exp_type == "unimodal":
                logits = torch.sort(torch.randn(4), descending=True)[0].numpy()
                w = np.exp(logits) ** m
                p = proba_plackett_luce(w, all_ranks)
                p_torch = torch.from_numpy(p)
            elif exp_type == "multimodal_sst":
                m1 = 1
                logits1 = torch.sort(torch.randn(4), descending=True)[0].numpy()
                w1 = np.exp(logits1) ** m
                m2 = 1
                logits2 = torch.randn(4).numpy()
                w2 = np.exp(logits2) ** m
                p_torch, P = get_multimodal(ws=[w1, w2], ratios=[0.5, 0.5], all_ranks=all_ranks)
                w = f"multimodal_sst_m1={m1}_m2={m2}"

            # Other parameters
            thresholds_ = np.linspace(0., 0.5, 21)

            # ERM
            dist = "erm"
            save = False
            res_dict_erm = launch_exp(
                dist, p_torch, w, delta, thresholds_, epochs, dist_type_sym=dist_type_sym, norm_type=norm_type,
                ratio_norm_=ratio_norm_, nb_simu_training=nb_simu_training, save=save, exp_type=exp_type)

            # Maxpair
            #dist = "maxpair"
            #res_dict_maxpair = launch_exp(
            #    dist, p_torch, w, delta, thresholds_, epochs, dist_type_sym=dist_type_sym, norm_type=norm_type,
            #    ratio_norm_=ratio_norm_, nb_simu_training=nb_simu_training, save=save, exp_type=exp_type)

            # Merge
            #dist = "merge"
            #res_dict_merge = launch_exp(
            #    dist, p_torch, w, delta, thresholds_, epochs, dist_type_sym=dist_type_sym, norm_type=norm_type,
            #    ratio_norm_=ratio_norm_, nb_simu_training=nb_simu_training, save=save, exp_type=exp_type)

            #dict_for_m = {"erm":res_dict_erm, "maxpair":res_dict_maxpair, "merge":res_dict_merge}
            dict_for_m = {"erm":res_dict_erm}
            res_dict_general[mykey][m] = dict_for_m

    return res_dict_general



if __name__ == "__main__":
    from pyspark.sql import SparkSession
    from pyspark.sql.types import StructType, StructField, IntegerType, ArrayType, StringType, FloatType, BooleanType
    from pyspark.sql.functions import pandas_udf
    import base64
    import gzip
    import json

    schema = StructType([StructField("n", IntegerType()),
                         StructField("exp_types", ArrayType(StringType())),
                         StructField("seed_vals", ArrayType(IntegerType())),
                         StructField("ms", ArrayType(FloatType())),
                         StructField("delta_list", ArrayType(IntegerType())),
                         StructField("epochs_list", ArrayType(IntegerType())),
                         StructField("dist_type_sym_list", ArrayType(BooleanType())),
                         StructField("norm_type_list", ArrayType(StringType())),
                         StructField("ratio_list", ArrayType(FloatType())),
                         StructField("nb_simu_training_list", ArrayType(IntegerType())),
                         ])
    #my_config, args_ = get_config()
    my_config = get_config()
    ml_udf = ps_functions.udf(main_exp_launcher)
    print(my_config)
    #my_final_dict = main_exp_launcher(my_config)

    spark = SparkSession.builder.master("local[1]").getOrCreate()
    df = spark.createDataFrame([my_config], schema=schema)
    df.show()
    print(ps_functions.struct(df.columns))

    #df = df.withColumn("config", ps_functions.to_json(ps_functions.to_json(ps_functions.struct(df.columns))))
    df = df.agg(ps_functions.to_json(ps_functions.collect_list(ps_functions.struct(ps_functions.col('n'),
                                                                                    ps_functions.col('exp_types'),
                                                                                    ps_functions.col('seed_vals'),
                                                                                    ps_functions.col('ms'),
                                                                                    ps_functions.col('delta_list'),
                                                                                    ps_functions.col('epochs_list'),
                                                                                    ps_functions.col('dist_type_sym_list'),
                                                                                    ps_functions.col('norm_type_list'),
                                                                                    ps_functions.col('ratio_list'),
                                                                                    ps_functions.col('nb_simu_training_list')))).alias('descr'))
    #df.show()

    @pandas_udf(StringType())
    def jsn(lst):
        return lst.apply(lambda lst: base64.b64encode(gzip.compress(json.dumps(lst).encode('utf-8'))).decode("utf-8"))

    #df = df.withColumn('descr2', jsn(ps_functions.col('descr')))
    df.show()

    #print(json.load(ps_functions.col("descr")))
    res = ml_udf(ps_functions.col("descr")) #ml_udf(json.load(ps_functions.col("descr")))
    print(res)

    print(res)
    df = df.withColumn("result", ml_udf(ps_functions.col("descr")))
    df.show()
    #pdf = df.toPandas()
    #pdf.to_csv("my_path.csv")

