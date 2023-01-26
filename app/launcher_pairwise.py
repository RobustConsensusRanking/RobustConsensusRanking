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
#from numba import jit

import argparse
from utils._utils import get_logger

logger = get_logger("Classic Launcher")

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
    '''
    data = json({"n" : 4, "exp-types": Array("unimodal")},{"n" : 4, "exp-types": Array("unimodal")})
    for exp in data : 
        n = exp("n")
    

    '''
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

    args.exp_types = np.array(list(map(str, str(args.exp_types).split(";"))))
    args.seed_vals = np.array(list(map(int, str(args.seed_vals).split(";"))))
    args.ms = np.array(list(map(float, str(args.ms).split(";"))))
    args.delta_list = np.array(list(map(int, str(args.delta_list).split(";"))))
    args.epochs_list = np.array(list(map(int, str(args.epochs_list).split(";"))))
    args.dist_type_sym_list = np.array(list(map(str2bool, str(args.dist_type_sym_list).split(";"))))
    args.norm_type_list = np.array(list(map(str, str(args.norm_type_list).split(";"))))
    args.ratio_list = np.array(list(map(float, str(args.ratio_list).split(";"))))
    args.nb_simu_training_list = np.array(list(map(int, str(args.nb_simu_training_list).split(";"))))

    max_len = np.max((args.exp_types.shape[0],args.seed_vals.shape[0],args.ms.shape[0],args.delta_list.shape[0],args.epochs_list.shape[0],args.dist_type_sym_list.shape[0],args.norm_type_list.shape[0],args.ratio_list.shape[0],args.nb_simu_training_list.shape[0]))
    args.exp_types = np.repeat(args.exp_types, int(max_len/args.exp_types.shape[0]))
    args.seed_vals = np.repeat(args.seed_vals, int(max_len/args.seed_vals.shape[0]))
    args.ms = np.repeat(args.ms, int(max_len/args.ms.shape[0]))
    args.delta_list = np.repeat(args.delta_list, int(max_len/args.delta_list.shape[0]))
    args.epochs_list = np.repeat(args.epochs_list, int(max_len/args.epochs_list.shape[0]))
    args.dist_type_sym_list = np.repeat(args.dist_type_sym_list, int(max_len/args.dist_type_sym_list.shape[0]))
    args.norm_type_list = np.repeat(args.norm_type_list, int(max_len/args.norm_type_list.shape[0]))
    args.ratio_list = np.repeat(args.ratio_list, int(max_len/args.ratio_list.shape[0]))
    args.nb_simu_training_list = np.repeat(args.nb_simu_training_list , int(max_len/args.nb_simu_training_list.shape[0]))

    return Config(**args.__dict__), args.__dict__


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

def return_pairwise_mat(M):
    for i in range(M.shape[0]):
        M[i,i] = torch.tensor(0.0)
        for j in range(i+1, M.shape[0]):
            if M[i,j] > torch.tensor(1.0):
                M[i,j] = torch.tensor(1.0)
            if M[i,j] < torch.tensor(0.0):
                M[i,j] = torch.tensor(0.0)
            M[j,i] = 1.0-M[i,j]
    return M

def expected_kendall(P1, P2):
    P1 = P1 - torch.diag(torch.diag(P1))
    P2 = P2 - torch.diag(torch.diag(P2))
    return torch.norm(P1 * (1-P2), 1)

def symmetrized_hausdorff_on_kendall(P1, P2):
    if len(P1.shape) < 2:
        P1 = P1.reshape( (int(np.sqrt(P1.shape[0])), int(np.sqrt(P1.shape[0]))) )
    if len(P2.shape) < 2:
        P2 = P2.reshape( (int(np.sqrt(P2.shape[0])), int(np.sqrt(P2.shape[0]))) )
    return torch.norm(torch.triu(P1,1) - torch.triu(P2,1), 1)

def disymmetrized_hausdorff_on_kendall(P1, P2):
    if len(P1.shape) < 2:
        P1 = P1.reshape( (int(np.sqrt(P1.shape[0])), int(np.sqrt(P1.shape[0]))) )
    if len(P2.shape) < 2:
        P2 = P2.reshape( (int(np.sqrt(P2.shape[0])), int(np.sqrt(P2.shape[0]))) )
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
        if len(y.size()) > 2:
            batch_size = y.size()[:-1]
        u = k.sample(batch_size).to(device).type(default_tensor_type)
        loss = k.log_prob(u) * m(y, u)
        return loss

    return smoothed_m_pg

def monte_carlo_phi(P, Q, dist_Tp_Tq, std_dev_=0.00001, nb_simu=50):
    kernel_conv = lambda _S: MultivariateNormal(_S.reshape(-1), std_dev_*torch.eye(_S.reshape(-1).shape[0])) #MultivariateNormal(_s, std_dev_*torch.eye(_s.size()[-1]))
    rhos = list()
    for i in range(nb_simu):
        Q2 = kernel_conv(Q.float()).sample(torch.Size()).reshape((P.shape[0], P.shape[1]))
        Q2 = return_pairwise_mat(Q2)
        rho = dist_Tp_Tq(P, Q2).detach().numpy().item()
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
    def forward(ctx, input_S, input_backward, P, dist_Tp_Tq, std_dev_, nb_simu):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        with torch.no_grad():
            softsign = torch.nn.Softsign()
            ctx.save_for_backward(input_backward)
            if len(input_S.shape) < 2:
                input_Q2 = torch.zeros((P.shape[0],P.shape[0]))
                input_Q2[torch.triu(torch.ones(P.shape[0], P.shape[0]), 1) == 1] = (softsign(input_S) + 1.0)/2.0
                input_Q2[torch.tril(torch.ones(P.shape[0], P.shape[0]), -1) == 1] = 1.0 - (softsign(input_S) + 1.0)/2.0
            else:
                input_Q2 = input_S
            monte_carlo_val = monte_carlo_phi(P, input_Q2, dist_Tp_Tq, std_dev_=std_dev_, nb_simu=nb_simu)
            return monte_carlo_val

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        back_,  = ctx.saved_tensors
        #print(f"back size = {back_.shape} ({back_}) and grad = {grad_output.shape} ({grad_output})")
        #print(f"size grad phi = {(back_ * grad_output).shape}")
        return (back_ * grad_output), None, None, None, None, None


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


def plot_end_training(P, dist_Tp_Tq, norms_, losses, mean_qs_, phis_, mean_phi2_, lambdas_, mean_lambdas_, grad_data,
                      freq_phi_, norm_type):
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

    #print("\n \n")
    time_fct = 100
    x_axis_val = subsample_fct(np.linspace(0, len(mean_lambdas_), len(mean_lambdas_)), time=time_fct)
    f, ax = plt.subplots(2, 2, figsize=(15, 8))

    mean_qs_subsample = subsample_fct(mean_qs_, time_fct)
    mean_qs1_ = [torch.norm(P - torch.tensor(mean_q_), int(norm_type)) for mean_q_ in
                 mean_qs_subsample]  # [mean_q_[0] for mean_q_ in mean_qs_]
    ax[0, 0].plot(x_axis_val, mean_qs1_)
    # ax[0,0].set_ylim([-0.05, 1.05])
    ax[0, 0].set_title(f"Norm of the difference between p and q")

    Phi_ = NewPhi.apply
    mean_phis_ = [Phi_(torch.tensor(mean_q_), grad_data, P, dist_Tp_Tq, 1e-7, 100) for mean_q_ in mean_qs_subsample]
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

    #print(f"Phi freq: {freq_phi_}")


def approximate_breakdown_function(delta, dist_Tp_Tq, P, epochs=150000, std_dev=0.01, lr=0.01, maxiter=10, max_reg=10.,
                                   eval_strat="real", norm_type="1", ratio_norm_=1.0, nb_simu_training=25):
    softmax = torch.nn.Softmax(dim=0)
    softplus = torch.nn.Softplus(beta=1, threshold=20)
    softsign = torch.nn.Softsign()
    S_ = P[torch.triu(torch.ones(P.shape[0], P.shape[0]), 1) == 1].reshape(-1).detach().clone().to(device).type(
        default_tensor_type)
    S_.requires_grad = True
    Q2 = torch.zeros((P.shape[0], P.shape[0]))
    Q2[torch.triu(torch.ones(P.shape[0], P.shape[0]), 1) == 1] = S_
    Q2[torch.tril(torch.ones(P.shape[0], P.shape[0]), -1) == 1] = 1.0 - S_

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

    optimizer_q2 = torch.optim.SGD([S_], lr=lr, weight_decay=0.0, momentum=0.1)
    optimizer_lambda = torch.optim.SGD([lambda_], lr=lr, weight_decay=0.0, momentum=0.1, maximize=True)

    scheduler_q2 = torch.optim.lr_scheduler.LinearLR(optimizer_q2, start_factor=1, end_factor=1 / 20, total_iters=10000)
    scheduler_lambda = torch.optim.lr_scheduler.LinearLR(optimizer_lambda, start_factor=1, end_factor=1 / 20,
                                                         total_iters=10000)
    for epoch in range(epochs):
        # try:
        std_dev_training = 1e-7
        std_dev_training_grad = 0.1
        std_dev_training = 1e-1 / np.sqrt(epoch + 1)
        kernel_conv = lambda _S: MultivariateNormal(_S, std_dev_training_grad * torch.eye(
            _S.size()[0]))  # MultivariateNormal(_S, std_dev_training*torch.eye(_s.size()[-1]))

        # with torch.no_grad():
        #    Q2[:,:] = (softsign(Q2[:,:]) + 1.0)/2.0

        res = smooth_pg_2(dist_Tp_Tq, kernel_conv)(P, Q2.reshape(-1))
        res.backward(retain_graph=True)
        grad_data = S_.grad.data.detach().clone()
        Phi_ = NewPhi.apply
        phi_ = Phi_(S_, grad_data, P, dist_Tp_Tq, std_dev_training, nb_simu_training)

        # Norm: L1, L2, or something between the two
        norm_ = ratio_norm_ * torch.norm(P - Q2, int(norm_type)) + (1.0 - ratio_norm_) * torch.norm(P - Q2, 2 * int(
            norm_type) - 3 * (int(norm_type) - 1))

        # LOSS
        loss = norm_ + lambda_ * delta - lambda_ * phi_

        optimizer_q2.zero_grad()
        optimizer_lambda.zero_grad()

        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_value_(S_, clip_value=5.0)
        optimizer_q2.step()
        optimizer_lambda.step()
        scheduler_q2.step()
        scheduler_lambda.step()

        with torch.no_grad():
            lambda_[:] = lambda_.clamp(min=0.0, max=None)
            if phi_ >= delta:
                lambda_[:] = 0.0
            # S_[:] = (softsign(S_) + 1.0)/2.0 #torch.sigmoid(S_) #S_.clamp(min=0.0, max=1.0)
            Q2[torch.triu(torch.ones(P.shape[0], P.shape[0]), 1) == 1] = (softsign(S_) + 1.0) / 2.0
            Q2[torch.tril(torch.ones(P.shape[0], P.shape[0]), -1) == 1] = 1.0 - (softsign(S_) + 1.0) / 2.0

        if epoch % (epochs / 10) == 0:
            logger.info(f"epoch = {epoch}")
            #print(
            #    f"EPOCH {epoch}: \n \t Q2={Q2} \n \t S_={S_} \n \t grad = {-S_.grad.data} \n \t lr = {optimizer_q2.param_groups[0]['lr']} \n \t phi = {phi_} and std_dev = {std_dev_training} and norm diff = {norm_} \n \n \t lambda_={lambda_} and grad = {lambda_.grad.data} \n \t lr = {optimizer_lambda.param_groups[0]['lr']}")
        losses.append(loss.detach().item())
        qs_total_.append(Q2.detach().numpy())
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

        # except:
        #    print(f"Something went wrong")
        #    print(f"EPOCH {epoch}: \n \t Q2={Q2} \n \t S_={S_} and grad = {-S_.grad.data} \n \t lr = {optimizer_q2.param_groups[0]['lr']} \n \t phi = {phi_} and std_dev = {1.0/(1+epoch)} \n \n \t lambda_={lambda_} and grad = {lambda_.grad.data} \n \t lr = {optimizer_lambda.param_groups[0]['lr']}")
        #    break

    return norms_, losses, S_, qs_total_, mean_qs_, phis_, mean_phi2_, lambdas_, mean_lambdas_, grad_data, freq_phi_  # losses, qs_, lambdas_, s_, mean_qs_, mean_lambdas_ #epsilons_, etas_, alphas_, s_, mean_qs_, mean_epsilons_, mean_etas_, mean_alphas_

#@jit
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


#@torch.jit.script
def _maxpair(P, threshold=0.):
    if len(P.shape) < 2:
        P = P.reshape((int(np.sqrt(P.shape[0])), int(np.sqrt(P.shape[0]))))
    s = torch.sum(1 * (P > 0.5) + 1 / 2 * (P == 0.5), axis=1)
    sigma = torch.argsort(-s)
    M = P[np.ix_(sigma, sigma)]

    idxs = torch.argwhere(M > 1 / 2 + threshold)
    for idx in idxs:
        M[:idx[0] + 1, idx[1]] = 1
        M[idx[1], :idx[0] + 1] = 0

    m = torch.max(torch.abs(M - 0.5) * (M != 0.5) * (torch.abs(M - 0.5) <= threshold))
    while m > 0:
        i, j = torch.argwhere(np.abs(M - 0.5) == m)[0, 0], torch.argwhere(np.abs(M - 0.5) == m)[0, 1]
        if i <= j:
            idx_tomerge1, idx_tomerge2 = i, j + 1
        elif i > j:
            idx_tomerge1, idx_tomerge2 = j, i + 1
        M = torch_merge_idx(M, torch.arange(idx_tomerge1, idx_tomerge2))
        m = torch.max(np.abs(M - 0.5) * (M != 0.5) * (torch.abs(M - 0.5) <= threshold))
    return M[np.ix_(torch.argsort(sigma), torch.argsort(sigma))]


def maxpair(P, torch_all_ranks, threshold=0.):
    # P = pairwise_matrix(p, torch_all_ranks=torch_all_ranks, n=n)
    return _maxpair(P, threshold=threshold)



@torch.jit.script #@njit
def torch_merge_idx(M, idx):
    # i,j \notin idx -> P_ij = M_ij
    # i \in idx, j \notin idx -> P_ij = \max_{k\in idx} M_kj
    # i \notin idx, j \in idx -> P_ij = \max_{k\in idx} M_ik
    # i,j \in idx -> P_ij = 0.5
    if len(M.shape) < 2:
        M = M.reshape( (int(torch.sqrt(M.shape[0])), int(torch.sqrt(M.shape[0]))) )
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

#@torch.jit.script
def merge(P, torch_all_ranks, threshold=0):
    #P = pairwise_matrix(p, torch_all_ranks=torch_all_ranks, n=n)
    cont = True
    while cont:
        if len(P.shape) < 2:
            P = P.reshape(int(np.sqrt(P.shape[0])), int(np.sqrt(P.shape[0])))
        P_mod = torch.abs(torch.triu(P,1)-torch.tensor(0.5))
        m = torch.min(torch.abs(torch.triu(P,1)-torch.tensor(0.5)))
        if m == 0.0 or m > threshold:
            cont = False
        else:
            idxs = torch.argwhere(P_mod == m)
            idxs = idxs.reshape(torch.prod(torch.tensor(idxs.shape)))
            P = torch_merge_idx(P, idxs)
    return P


def erm(P, torch_all_ranks):
    #P = pairwise_matrix(p, torch_all_ranks=torch_all_ranks, n=n)
    if len(P.shape) < 2:
        P = P.reshape( (int(np.sqrt(P.shape[0])), int(np.sqrt(P.shape[0]))) )
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
        #if printer:
        #    print(f"val: {val_} for {bucket}")
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
def get_automatic_thresholds(w, n_items=4):
    p = proba_plackett_luce(w, all_ranks, n_items=n_items)
    p_torch = torch.from_numpy(p)
    torch_all_ranks = torch.from_numpy(np.asarray(all_ranks))
    P = pairwise_matrix(p_torch, torch_all_ranks, n=4)
    #print(f"Pairwise mat: \n {P}")

    m = torch.flatten(torch.triu(P, 1))
    m = torch.sort(m[m != 0.0])[0]
    # m = torch.cat((torch.tensor([0.5]), m), dim=-1)
    # l = [ ((m[i+1] + m[i])/2).numpy() - 0.5 for i in range(0, m.shape[0]-1)]
    # thresholds_ = np.unique(l)
    minval = (m - 0.5)[0]
    maxval = (m - 0.5)[-1]
    thresholds_ = [0]
    for val in list(np.linspace(minval - 0.0005, maxval + 0.0005, 21)):
        thresholds_.append(val)
    #print(f"thresholds = {thresholds_}")
    return thresholds_


def torch_dist(dist, P1, P2, torch_all_ranks, threshold, dist_type_sym):
    if dist == "erm":
        R1 = erm(P1, torch_all_ranks)
        R2 = erm(P2, torch_all_ranks)
    elif dist == "maxpair":
        R1 = maxpair(P1, torch_all_ranks, threshold=threshold)
        R2 = maxpair(P2, torch_all_ranks, threshold=threshold)
    elif dist == "merge":
        R1 = merge(P1, torch_all_ranks, threshold=threshold)
        R2 = merge(P2, torch_all_ranks, threshold=threshold)
    elif dist == "depth":
        R1 = depth(P1, torch_all_ranks, norm="l1", printer=False)
        R2 = depth(P2, torch_all_ranks, norm="l1", printer=False)
    elif dist == "wasserstein":
        R1 = wasserstein(P1, torch_all_ranks, n=4)
        R2 = wasserstein(P2, torch_all_ranks, n=4)
    if dist_type_sym:
        return symmetrized_hausdorff_on_kendall(R1, R2)
    else:
        return disymmetrized_hausdorff_on_kendall(R1, R2)


def launch_exp(all_ranks, dist, p_torch, w, delta, thresholds_, epochs, save=True, dist_type_sym=True, norm_type="1",
               ratio_norm_=1.0, nb_simu_training=25, exp_type="unimodal", n_items=4):

    torch_all_ranks = torch.from_numpy(np.asarray(all_ranks))
    P = pairwise_matrix(p_torch, torch_all_ranks, n=n_items)
    logger.info(f"Original proba p = {p_torch} and pairwise = {P}")

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
        logger.info(f"\n \t EXP THRESHOLD nb {i_} = {threshold} \n \n \n")
        dist_Tp_Tq = lambda _P, _Q: torch_dist(dist, _P, _Q, torch_all_ranks, threshold=threshold,
                                               dist_type_sym=dist_type_sym)

        norms, losses, s_, qs_, mean_qs, phis, mean_phi2, lambdas, mean_lambdas, grad_data, freq_phi = approximate_breakdown_function(
            delta - 1e-6, dist_Tp_Tq, P, epochs=epochs, std_dev=0.001, lr=0.5 / div_, maxiter=21, eval_strat="smoothed",
            norm_type=norm_type, ratio_norm_=ratio_norm_, nb_simu_training=nb_simu_training)
        #plot_end_training(P, dist_Tp_Tq, norms, losses, mean_qs, phis, mean_phi2, lambdas, mean_lambdas, grad_data,
        #                  freq_phi, norm_type)
        dict_res_training = {"norms": norms, "losses": losses, "s_": s_, "qs_": qs_, "mean_qs": mean_qs, "phis": phis,
                             "mean_phi2": mean_phi2, "lambdas": lambdas, "mean_lambdas": mean_lambdas,
                             "grad_data": grad_data, "freq_phi": freq_phi}
        dict_training[threshold] = dict_res_training

        # q1 is the mean of the last q found
        qlist_ = qs_[epochs - 10000:]
        q1 = np.mean(qlist_, axis=0)

        # q2 is the overall mean
        q2 = mean_qs[-1]
        #print(
        #    f"\n Found attack distrib q = {q2} (erm = {erm(torch.tensor(q2), torch_all_ranks)}) OR \nq = {q1} (erm = {erm(torch.tensor(q1), torch_all_ranks)})")  # and corresponding phi = {erm(q2, torch_all_ranks)} \n")

        eps_list1.append(torch.norm(P - torch.tensor(q2), 1))
        eps_list2.append(torch.norm(P - torch.tensor(q2), 2))

        alt_eps_list1.append(torch.norm(P - torch.tensor(q1).unsqueeze(0), 1))
        alt_eps_list2.append(torch.norm(P - torch.tensor(q1).unsqueeze(0), 2))

        logger.info(f"L1-norm = {torch.norm(P - torch.tensor(q2), 1)} or {torch.norm(P - torch.tensor(q1).unsqueeze(0), 1)}")


        if dist == "erm":
            Ptilde = erm(P, torch_all_ranks)
        elif dist == "maxpair":
            Ptilde = maxpair(P, torch_all_ranks, threshold=threshold)
        elif dist == "merge":
            Ptilde = merge(P, torch_all_ranks, threshold=threshold)
        elif dist == "depth":
            Ptilde = depth(P, torch_all_ranks, norm="l1", printer=False)
        elif dist == "wasserstein":
            Ptilde = wasserstein(P, torch_all_ranks, n=4)
        exp_kendall = expected_kendall(Ptilde, P).detach().item()
        perf_list.append(exp_kendall)

    if save:
        directory = f"{os.getcwd()}/perf_robustness_profile/pairwise_res/"+exp_type+"/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = f"perf_robustness_dist={dist}_w={w}_delta={delta}_epochs={epochs}_dist_type_sym={dist_type_sym}_norm_L{norm_type}_ratio_{ratio_norm_}.pt"
        final_val_dict = {"thresholds": thresholds, "perf_list": perf_list, "eps_list1": eps_list1,
                          "eps_list2": eps_list2, "alt_eps_list1": alt_eps_list1, "alt_eps_list2": alt_eps_list1,
                          "p_torch": p_torch}

        torch.save({"training_dict": dict_training, "final_val_dict": final_val_dict}, directory + filename)
        logger.info(f"File saved at {directory}\n{filename}")

    return perf_list, eps_list1, eps_list2, alt_eps_list1, alt_eps_list2, dict_training


def check_SST(P):
    P = P.numpy()
    cycles = list()
    P2 = np.nan_to_num((P - 1 / 2).clip(0))
    P2
    G = nx.from_numpy_matrix(P2, create_using=nx.MultiDiGraph())
    for cycle in nx.simple_cycles(G):
        cycles.append(cycle)
    if len(cycles) < 1:
        #print("The distribution is SST")
        res = True
    else:
        #print("The distribution is -NOT- SST")
        res = False
    return res


def get_multimodal(ws, ratios, all_ranks, n_items=4):
    torch_all_ranks = torch.from_numpy(np.asarray(all_ranks))
    p_torch = torch.zeros((1, 24))
    #print(p_torch, p_torch.shape)
    for (w, ratio) in zip(ws, ratios):
        p_ = proba_plackett_luce(w, all_ranks, n_items)
        p_torch_ = torch.from_numpy(p_)
        #print(p_torch_, p_torch_.shape)
        p_torch += ratio * p_torch_
    P = pairwise_matrix(p_torch, torch_all_ranks, n=4)
    is_sst = check_SST(P)

    return p_torch, P


def main_exp_launcher(config):

    # Generating ranks - basic parameters
    n_items = config.n
    all_ranks = list(itertools.permutations(list(np.arange(n_items))))
    all_ranks = [np.array(elem) for elem in all_ranks]
    torch_all_ranks = torch.from_numpy(np.asarray(all_ranks))

    for seed_val, exp_type, delta, epochs, dist_type_sym, norm_type, ratio_norm_, nb_simu_training, m in zip(config.seed_vals, config.exp_types, config.delta_list, config.epochs_list, config.dist_type_sym_list, config.norm_type_list, config.ratio_list, config.nb_simu_training_list, config.ms):
        torch.manual_seed(seed_val)
        logger.info(f"\t \t SEED EXP {seed_val}")
        logger.info(f"\t \t \t M VAL = {m}")

        if exp_type == "unimodal":
            logits = torch.sort(torch.randn(n_items), descending=True)[0].numpy()
            w = np.exp(logits) ** m
            p = proba_plackett_luce(w, all_ranks, n_items=n_items)
            p_torch = torch.from_numpy(p)
            P = pairwise_matrix(p_torch, torch_all_ranks, n=n_items)
        elif exp_type == "multimodal_sst":
            logits1 = torch.sort(torch.randn(n_items), descending=True)[0].numpy()
            w1 = np.exp(logits1) ** m
            logits2 = torch.randn(n_items).numpy()
            w2 = np.exp(logits2) ** m
            p_torch, P = get_multimodal(ws=[w1, w2], ratios=[0.5, 0.5], all_ranks=all_ranks, n_items=n_items)
            w = f"multimodal_sst_m={m}_seed={seed_val}_n={n_items}"
        elif exp_type == "two_untied":
            import math

            p_torch = torch.zeros((1,math.factorial(n_items)))

            logits = torch.sort(torch.randn(n_items), descending=True)[0].numpy()
            w = np.exp(logits)
            p = proba_plackett_luce(w, all_ranks, n_items=n_items)

            mixture_val = 1
            gap_mode = 0.01

            if n_items == 8:
                p_torch[0,0] += (1/6 + 2/6*gap_mode)
                p_torch[0,1] += (1/6 + 1/6*gap_mode)
                p_torch[0,2] += (1/6 + 0/6*gap_mode)
                p_torch[0,3] += (1/6 - 0/6*gap_mode)
                p_torch[0,4] += (1/6 - 1/6*gap_mode)
                p_torch[0,5] += (1/6 - 2/6*gap_mode)
            elif n_items == 4:
                p_torch[0,0] += (0.5+gap_mode)
                p_torch[0,1] += (0.5-gap_mode)

            p_torch = mixture_val*p_torch + (1-mixture_val)*p
            P = pairwise_matrix(p_torch, torch_all_ranks, n=n_items)

            w = f"two_untied_mix={mixture_val}_gap={gap_mode}_seed={seed_val}_n={n_items}"

        # Other parameters
        #thresholds_ = np.linspace(0., 0.5, 11)
        a = P[P>0.5].reshape(-1)#torch.triu(P, 1).reshape(-1)
        a = a[a!=0.0]
        a = a-0.5+1e-4
        a = torch.sort(a)[0][:6]
        shape_val = a.shape[0]-1

        for i, a_ in enumerate(a):
            if i == 0:
                inter = torch.tensor(np.linspace(0, a_, 3)[1:])
                a = torch.cat((inter, a))
            elif int(i) >= shape_val:
                inter = torch.tensor(np.linspace(a_, a_+0.15, 5)[1:])
                #a = torch.cat((inter, a))
            else:
                if int(i) < 3:
                    inter = torch.tensor(np.linspace(a[i-1], a_, 3)[1:])
                else:
                    inter = torch.tensor(np.linspace(a[i-1], a_, 5)[1:])
                a = torch.cat((inter, a))
        a = torch.unique(a)
        a = a[a<0.5]
        thresholds_ = torch.sort(a)[0]

        logger.info(f"thresholds_ = {thresholds_} len = {thresholds_.shape}")
        save = True

        # ERM
        logger.info(f"ERM with delta={delta}")
        dist = "erm"
        perf_list_erm, eps_list_erm1, eps_list_erm2, alt_eps_list_erm1, alt_eps_list_erm2, dict_res_training_erm = launch_exp(
            all_ranks, dist, p_torch, w, delta, thresholds_, epochs, dist_type_sym=dist_type_sym, norm_type=norm_type,
            ratio_norm_=ratio_norm_, nb_simu_training=nb_simu_training, save=save, exp_type=exp_type, n_items=n_items)

        # Maxpair
        logger.info(f"Maxpair with delta={delta}")
        dist = "maxpair"
        perf_list_maxpair, eps_list_maxpair1, eps_list_maxpair2, alt_eps_list_maxpair1, alt_eps_list_maxpair2, dict_res_training_maxpair = launch_exp(
            all_ranks, dist, p_torch, w, delta, thresholds_, epochs, dist_type_sym=dist_type_sym, norm_type=norm_type,
            ratio_norm_=ratio_norm_, nb_simu_training=nb_simu_training, save=save, exp_type=exp_type, n_items=n_items)

        # Merge
        #logger.info(f"Merge")
        #dist = "merge"
        #perf_list_merge, eps_list_merge1, eps_list_merge2, alt_eps_list_merge1, alt_eps_list_merge2, dict_res_training_merge = launch_exp(
        #    all_ranks, dist, p_torch, w, delta, thresholds_, epochs, dist_type_sym=dist_type_sym, norm_type=norm_type,
        #    ratio_norm_=ratio_norm_, nb_simu_training=nb_simu_training, save=save, exp_type=exp_type, n_items=n_items)


if __name__ == "__main__":
    my_config, args_ = get_config()
    logger.info(f"my_config = {my_config}")
    main_exp_launcher(my_config)