import numpy as np
import os
from typing import NamedTuple
import torch
import itertools
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal, Normal

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
    thresholds: str = "0.05"
    gap_mode: str = "0.1"
    mixture_val: str = "1"


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
    parser.add_argument("--thresholds", type=str, default="0.05")
    parser.add_argument("--gap_mode", type=str, default="0.1")
    parser.add_argument("--mixture_val", type=str, default="1")

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
    args.thresholds = np.array(list(map(float, str(args.thresholds).split(";"))))
    args.gap_mode = np.array(list(map(float, str(args.gap_mode).split(";"))))
    args.mixture_val = np.array(list(map(float, str(args.mixture_val).split(";"))))

    max_len = np.max((args.thresholds.shape[0],
        args.exp_types.shape[0],
        args.seed_vals.shape[0],
        args.ms.shape[0],
        args.delta_list.shape[0],
        args.epochs_list.shape[0],
        args.dist_type_sym_list.shape[0],
        args.norm_type_list.shape[0],
        args.ratio_list.shape[0],
        args.nb_simu_training_list.shape[0]))
    args.exp_types = np.repeat(args.exp_types, int(max_len/args.exp_types.shape[0]))
    args.seed_vals = np.repeat(args.seed_vals, int(max_len/args.seed_vals.shape[0]))
    args.ms = np.repeat(args.ms, int(max_len/args.ms.shape[0]))
    args.delta_list = np.repeat(args.delta_list, int(max_len/args.delta_list.shape[0]))
    args.epochs_list = np.repeat(args.epochs_list, int(max_len/args.epochs_list.shape[0]))
    args.dist_type_sym_list = np.repeat(args.dist_type_sym_list, int(max_len/args.dist_type_sym_list.shape[0]))
    args.norm_type_list = np.repeat(args.norm_type_list, int(max_len/args.norm_type_list.shape[0]))
    args.ratio_list = np.repeat(args.ratio_list, int(max_len/args.ratio_list.shape[0]))
    args.nb_simu_training_list = np.repeat(args.nb_simu_training_list , int(max_len/args.nb_simu_training_list.shape[0]))
    args.thresholds = np.repeat(args.thresholds, int(max_len / args.thresholds.shape[0]))
    args.gap_mode = np.repeat(args.gap_mode, int(max_len / args.gap_mode.shape[0]))
    args.mixture_val = np.repeat(args.mixture_val, int(max_len / args.mixture_val.shape[0]))

    return Config(**args.__dict__), args.__dict__


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

def return_pairwise_mat(M, stat=False):
    for i in range(M.shape[0]):
        M[i,i] = torch.tensor(0.0)
        for j in range(i+1, M.shape[0]):
            if M[i,j] > torch.tensor(1.0):
                M[i,j] = torch.tensor(1.0)
            if M[i,j] < torch.tensor(0.0):
                M[i,j] = torch.tensor(0.0)

            if stat:
                if M[i,j] > torch.tensor(0.5):
                    M[i,j] = torch.tensor(1.0)
                if M[i,j] < torch.tensor(0.5):
                    M[i,j] = torch.tensor(0.0)
            M[j,i] = 1.0-M[i,j]
    return M

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
        softplus = torch.nn.Softplus(beta=1, threshold=20)
        k = k_x(x)
        batch_size = torch.Size()
        if len(y.size()) > 1:
            batch_size = y.size()[:-1]
        #u = k.sample(batch_size).to(device).type(default_tensor_type)
        z = k.sample(batch_size).to(device).type(default_tensor_type)
        u = softplus(z) / torch.sum(softplus(z)) #torch.exp(z) / torch.sum(torch.exp(z))
        loss = k.log_prob(z) * m(y, u)
        return loss

    return smoothed_m_pg

def monte_carlo_phi(p, q, dist_Tp_Tq, std_dev_=0.00001, nb_simu=50, s=None):
    kernel_conv = lambda _s: MultivariateNormal(_s, std_dev_*torch.eye(_s.size()[-1]))
    softplus = torch.nn.Softplus(beta=1, threshold=20)
    rhos = list()
    for i in range(nb_simu):
        #q2 = kernel_conv(q.float()).sample(torch.Size())
        if s is not None:
            logits = kernel_conv(s).sample(torch.Size())
        else:
            logits = kernel_conv(torch.log(q.float())).sample(torch.Size())
        q2 = softplus(logits) / torch.sum(softplus(logits)) #torch.exp(logits) / torch.sum(torch.exp(logits))
        rho = dist_Tp_Tq(p, q2.unsqueeze(0)).detach().numpy().item()
        rhos.append(rho)
    rho_final = np.mean(rhos)
    #logger.info(f"rhos = {rhos} and final = {rho_final}")
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
    def forward(ctx, input_q2, input_backward, p_torch, dist_Tp_Tq, std_dev_, nb_simu, s):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input_backward)
        monte_carlo_val = monte_carlo_phi(p_torch, input_q2, dist_Tp_Tq, std_dev_=std_dev_, nb_simu=nb_simu, s=s)
        return monte_carlo_val

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        back_,  = ctx.saved_tensors
        return back_ * grad_output, None, None, None, None, None, None


def subsample_fct(vect, time=10):
    l = len(vect)
    res = vect[0:l:time]
    return res

def plot_end_training(p_torch, dist_Tp_Tq, norms_, losses, mean_qs_, phis_, mean_phi2_, lambdas_, mean_lambdas_,
                      grad_data, norm_type):
    f, ax = plt.subplots(2, 2, figsize=(15, 8))
    ax[0, 0].plot(norms_)
    ax[0, 0].set_title(f"Norm diff btw p and q")

    ax[0, 1].plot(phis_)
    ax[0, 1].set_ylim([-0.05, 6.05])
    ax[0, 1].set_title(f"Phi")

    ax[1, 0].plot(lambdas_)
    ax[1, 0].set_title(f"Lambda")

    ax[1, 1].plot(losses)
    ax[1, 1].set_title(f"Loss")
    plt.show()

    time_fct = 100
    x_axis_val = subsample_fct(np.linspace(0, len(mean_lambdas_), len(mean_lambdas_)), time=time_fct)
    f, ax = plt.subplots(2, 2, figsize=(15, 8))

    mean_qs_subsample = subsample_fct(mean_qs_, time_fct)
    mean_qs1_ = [torch.norm(p_torch - torch.tensor(mean_q_), int(norm_type)) for mean_q_ in
                 mean_qs_subsample]
    ax[0, 0].plot(x_axis_val, mean_qs1_)
    ax[0, 0].set_title(f"Norm of the difference between p and q")

    Phi_ = NewPhi.apply
    mean_phis_ = [Phi_(torch.tensor(mean_q_), grad_data, p_torch, dist_Tp_Tq, 1e-7, 100, None) for mean_q_ in
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


def approximate_breakdown_function(delta, dist_Tp_Tq, p_torch, epochs=20000, lr=0.01,  norm_type="1", nb_simu_training=25):
    softplus = torch.nn.Softplus(beta=1, threshold=20)
    q2 = p_torch.detach().clone().squeeze(0)
    q2.requires_grad = True
    s_ = (-1)*torch.rand(q2.shape).to(device).type(default_tensor_type)+1.0
    logger.info(f"Init val = {s_}")
    s_.requires_grad = True

    lambda_ = torch.tensor((1.0,), requires_grad=True)
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

    optimizer_q2 = torch.optim.SGD([s_], lr=0.5*lr, weight_decay=0.0, momentum=0.1)
    optimizer_lambda = torch.optim.SGD([lambda_], lr=10*lr, weight_decay=0.0, momentum=0.1, maximize=True)

    scheduler_q2 = torch.optim.lr_scheduler.LinearLR(optimizer_q2, start_factor=1, end_factor=1 / 50, total_iters=30000)
    scheduler_lambda = torch.optim.lr_scheduler.LinearLR(optimizer_lambda, start_factor=1, end_factor=1 / 50,
                                                         total_iters=30000)

    for epoch in range(epochs):
        std_dev_training = 1e-3 #1e-2
        std_dev_training_grad = 1e-1 #1e-0
        kernel_conv = lambda _s: MultivariateNormal(_s, std_dev_training_grad * torch.eye(_s.size()[-1]))
        q2 = softplus(s_) / torch.sum(softplus(s_))
        res = smooth_pg_2(dist_Tp_Tq, kernel_conv)(p_torch, s_)
        res.backward(retain_graph=True)
        grad_data = s_.grad.data.detach().clone()
        Phi_ = NewPhi.apply
        phi_ = Phi_(q2, grad_data, p_torch, dist_Tp_Tq, std_dev_training, nb_simu_training, s_)

        # Norm: L1, L2, or something between the two
        norm_ = torch.norm(p_torch - torch.unsqueeze(q2, 0), int(norm_type))
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
            if len(phis_) > 50 and np.max(delta - phis_[-50:]) < 0:
                lambda_ *= 0.0
            lambda_[:] = lambda_.clamp(min=0.0, max=None)

        if int(epoch % (epochs / 10)) == 0:
            logger.info(f"Epoch = {epoch}")
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
                -1])
            mean_lambdas_.append((epoch + 1) / (epoch + 2) * mean_lambdas_[-1] + 1 / (epoch + 2) * lambdas_[
                -1])
            mean_phi2_.append((epoch + 1) / (epoch + 2) * mean_phi2_[-1] + 1 / (epoch + 2) * phis_[-1])

    return norms_, losses, s_, qs_total_, mean_qs_, phis_, mean_phi2_, lambdas_, mean_lambdas_, grad_data, freq_phi_


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
    M = M[np.ix_(torch.argsort(sigma), torch.argsort(sigma))]
    #M = return_pairwise_mat(M)
    return M #M[np.ix_(torch.argsort(sigma), torch.argsort(sigma))]


def maxpair(p, torch_all_ranks, n=4, threshold=0.):
    P = pairwise_matrix(p, torch_all_ranks=torch_all_ranks, n=n)
    return _maxpair(P, threshold=threshold)


@torch.jit.script
def torch_merge_idx(M, idx):
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

def merge(p, torch_all_ranks, threshold=0, n=4):
    P = pairwise_matrix(p, torch_all_ranks=torch_all_ranks, n=n)
    cont = True
    while cont:
        P_mod = torch.abs(torch.triu(P,1)-torch.tensor(0.5))
        m = torch.min(P_mod[P_mod > 0.0])
        if m > threshold:
            cont = False
        else:
            idxs = torch.argwhere(P_mod == m)
            idxs = idxs.reshape(torch.prod(torch.tensor(idxs.shape)))
            P = torch_merge_idx(P, idxs)
        if m == 0.5:
            cont = False
    P = return_pairwise_mat(P, stat=True)
    return P


def erm(p, torch_all_ranks, n=4):
    P = pairwise_matrix(p, torch_all_ranks=torch_all_ranks, n=n)
    return torch.round(P)


def torch_dist(dist, p_torch1, p_torch2, torch_all_ranks, threshold, dist_type_sym, n_items=4):
    if dist == "erm":
        R1 = erm(p_torch1, torch_all_ranks, n=n_items)
        R2 = erm(p_torch2, torch_all_ranks, n=n_items)
    elif dist == "maxpair":
        R1 = maxpair(p_torch1, torch_all_ranks, n=n_items, threshold=threshold)
        R2 = maxpair(p_torch2, torch_all_ranks, n=n_items, threshold=threshold)
    elif dist == "merge":
        R1 = merge(p_torch1, torch_all_ranks, threshold=threshold, n=n_items)
        R2 = merge(p_torch2, torch_all_ranks, threshold=threshold, n=n_items)
    if dist_type_sym:
        return symmetrized_hausdorff_on_kendall(R1, R2)
    else:
        return disymmetrized_hausdorff_on_kendall(R1, R2)


def launch_exp(all_ranks, dist, p_torch, w, delta, thresholds_, epochs, save=True, dist_type_sym=True, norm_type="1",
               nb_simu_training=25, exp_type="unimodal", n_items=4):
    my_lr_training = 0.5

    torch_all_ranks = torch.from_numpy(np.asarray(all_ranks))
    P = pairwise_matrix(p_torch, torch_all_ranks, n=n_items)
    
    eps_list1 = []
    eps_list2 = []
    alt_eps_list1 = []
    alt_eps_list2 = []
    perf_list = []
    if dist in ["erm"]:
        thresholds = [0.0]
    elif dist in ["maxpair", "merge"]:
        thresholds = thresholds_

    dict_training = dict()

    for i_, threshold in enumerate(thresholds):
        logger.info(f"\t EXP THRESHOLD nb {i_} = {threshold} \n")

        if dist == "erm":
            stat_p = erm(p_torch, torch_all_ranks)
        elif dist == "maxpair":
            stat_p = maxpair(p_torch, torch_all_ranks, n=n_items, threshold=threshold)
        elif dist == "merge":
            stat_p = merge(p_torch, torch_all_ranks, threshold=threshold, n=n_items)
        logger.info(f"Original proba p = {p_torch} \n and pairwise = {pairwise_matrix(p_torch, torch_all_ranks)} \n and stat origin ={stat_p}")

        dist_Tp_Tq = lambda _p, _q: torch_dist(dist, _p, _q, torch_all_ranks, threshold=threshold,
                                               dist_type_sym=dist_type_sym, n_items=n_items)

        norms, losses, s_, qs_, mean_qs, phis, mean_phi2, lambdas, mean_lambdas, grad_data, freq_phi = approximate_breakdown_function(
            delta - 1e-6, dist_Tp_Tq, p_torch, epochs=epochs,  lr=my_lr_training, norm_type=norm_type, nb_simu_training=nb_simu_training)
        dict_res_training = {"norms": norms, "losses": losses, "s_": s_, "qs_": qs_, "mean_qs": mean_qs, "phis": phis,
                             "mean_phi2": mean_phi2, "lambdas": lambdas, "mean_lambdas": mean_lambdas,
                             "grad_data": grad_data, "freq_phi": freq_phi}
        dict_training[threshold] = dict_res_training

        # q1 is the mean of the last q found
        qlist_ = qs_[epochs - 5000:]
        q1 = np.mean(qlist_, axis=0)

        # q2 is the overall mean
        q2 = mean_qs[-1]
        Q2 = pairwise_matrix(q2, torch_all_ranks, n=n_items)
        if dist == "erm":
            stat_q2 = erm(torch.tensor(q2).unsqueeze(0), torch_all_ranks)
        elif dist == "maxpair":
            stat_q2 = maxpair(torch.tensor(q2).unsqueeze(0), torch_all_ranks, n=n_items, threshold=threshold)
        elif dist == "merge":
            stat_q2 = merge(torch.tensor(q2).unsqueeze(0), torch_all_ranks, threshold=threshold, n=n_items)
        logger.info(f"Res proba = {q2}\nRes Pairwise = {Q2}\nres stat = {stat_q2}")

        eps_list1.append(torch.norm(p_torch - torch.tensor(q2).unsqueeze(0), 1))
        eps_list2.append(torch.norm(p_torch - torch.tensor(q2).unsqueeze(0), 2))

        alt_eps_list1.append(torch.norm(p_torch - torch.tensor(q1).unsqueeze(0), 1))
        alt_eps_list2.append(torch.norm(p_torch - torch.tensor(q1).unsqueeze(0), 2))

        logger.info(f"L1-norm = {torch.norm(p_torch - torch.tensor(q2).unsqueeze(0), 1)} or {torch.norm(p_torch - torch.tensor(q1).unsqueeze(0), 1)}")

        if dist == "erm":
            Ptilde = erm(p_torch, torch_all_ranks, n=n_items)
        elif dist == "maxpair":
            Ptilde = maxpair(p_torch, torch_all_ranks, n=n_items, threshold=threshold)
        elif dist == "merge":
            Ptilde = merge(p_torch, torch_all_ranks, threshold=threshold, n=n_items)
        logger.info(f"Statistic res = {Ptilde}")
        exp_kendall = expected_kendall(Ptilde, P).detach().item()
        perf_list.append(exp_kendall)

    if save:
        directory = f"{os.getcwd()}/perf_robustness_profile/"+exp_type+"/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = f"perf_robustness_dist={dist}_w={w}_delta={delta}_epochs={epochs}_dist_type_sym={dist_type_sym}_norm_L{norm_type}.pt"
        final_val_dict = {"thresholds": thresholds, "perf_list": perf_list, "eps_list1": eps_list1,
                          "eps_list2": eps_list2, "alt_eps_list1": alt_eps_list1, "alt_eps_list2": alt_eps_list1,
                          "p_torch": p_torch}

        torch.save({"training_dict": dict_training, "final_val_dict": final_val_dict}, directory + filename)
        logger.info(f"File saved at {directory}\n{filename}")
    return perf_list, eps_list1, eps_list2, alt_eps_list1, alt_eps_list2, dict_training


def main_exp_launcher(config):
    logger.info(f"nb seed = {config.seed_vals.shape} and nb m = {config.ms.shape}")
    n_items = config.n
    all_ranks = list(itertools.permutations(list(np.arange(n_items))))
    all_ranks = [np.array(elem) for elem in all_ranks]

    for seed_val, exp_type, delta, epochs, dist_type_sym, norm_type, ratio_norm_, nb_simu_training, m in zip(config.seed_vals, config.exp_types, config.delta_list, config.epochs_list, config.dist_type_sym_list, config.norm_type_list, config.ratio_list, config.nb_simu_training_list, config.ms):
        torch.manual_seed(seed_val)
        logger.info(f"\t \t SEED EXP {seed_val}")

        #for m in config.ms:
        logger.info(f"\t \t \t M VAL = {m}")

        if exp_type == "unimodal":
            logits = torch.sort(torch.randn(n_items), descending=True)[0].numpy()
            w = np.exp(logits) ** m
            p = proba_plackett_luce(w, all_ranks)
            p_torch = torch.from_numpy(p)
            logger.info(f"w = {w}, p_torch = {p_torch}")

        elif exp_type == "two_untied":
            import math
            p_torch = torch.zeros((1,math.factorial(n_items)))
            n_fact = math.factorial(n_items)
            p = torch.rand(n_fact) 
            p /= torch.sum(p)

            mixture_val = config.mixture_val
            gap_mode = config.gap_mode

            p_torch[0,0] += (0.5+gap_mode)
            p_torch[0,1] += (0.5-gap_mode)
            p_torch = mixture_val*p_torch + (1-mixture_val)*p
            w = f"two_untied_mix={mixture_val}_gap={gap_mode}_seed={seed_val}"


        elif exp_type == "agh":
            p_torch = torch.load("p_torch_agh.pt")
            w = f"agh_seed={seed_val}"

        elif exp_type == "debian":
            p_torch = torch.load("p_torch_debian.pt")
            w = f"debian_seed={seed_val}"

        elif exp_type == "netflix3.1":
            p_torch = torch.load("p_torch_netflix3.1.pt")
            w = f"netflix3.1_seed={seed_val}"
        elif exp_type == "netflix4.1":
            p_torch = torch.load("p_torch_netflix4.1.pt")
            w = f"netflix4.1_seed={seed_val}"
        elif exp_type == "netflix4.2":
            p_torch = torch.load("p_torch_netflix4.3.pt")
            w = f"netflix4.2_seed={seed_val}"
        elif exp_type == "netflix4.3":
            p_torch = torch.load("p_torch_netflix4.3.pt")
            w = f"netflix4.3_seed={seed_val}"
        elif exp_type == "netflix4.4":
            p_torch = torch.load("p_torch_netflix4.4.pt")
            w = f"netflix4.4_seed={seed_val}"
        elif exp_type == "netflix4.5":
            p_torch = torch.load("p_torch_netflix4.5.pt")
            w = f"netflix4.5_seed={seed_val}"
        elif exp_type == "netflix4.6":
            p_torch = torch.load("p_torch_netflix4.6.pt")
            w = f"netflix4.6_seed={seed_val}"
        elif exp_type == "netflix4.7":
            p_torch = torch.load("p_torch_netflix4.7.pt")
            w = f"netflix4.7_seed={seed_val}"

        elif exp_type == "mecha_turk1":
            p_torch = torch.load("p_torch_mecha_turk1.pt")
            w = f"mecha_turk1_seed={seed_val}"
        elif exp_type == "mecha_turk2":
            p_torch = torch.load("p_torch_mecha_turk2.pt")
            w = f"mecha_turk2_seed={seed_val}"
        elif exp_type == "mecha_turk3":
            p_torch = torch.load("p_torch_mecha_turk3.pt")
            w = f"mecha_turk3_seed={seed_val}"
        elif exp_type == "mecha_turk4":
            p_torch = torch.load("p_torch_mecha_turk4.pt")
            w = f"mecha_turk4_seed={seed_val}"

        elif exp_type == "apa1":
            p_torch = torch.load("p_torch_apa1.pt")
            w = f"apa1_seed={seed_val}"

        # THRESHOLDS
        thresholds_ = torch.tensor(config.thresholds)
        logger.info(f"thresholds = {thresholds_}")

        save = True

        # ERM
        logger.info(f"\n ERM \n")
        dist = "erm"
        perf_list_erm, eps_list_erm1, eps_list_erm2, alt_eps_list_erm1, alt_eps_list_erm2, dict_res_training_erm = launch_exp(
            all_ranks, dist, p_torch, w, delta, thresholds_, epochs, dist_type_sym=dist_type_sym, norm_type=norm_type,
            nb_simu_training=nb_simu_training, save=save, exp_type=exp_type, n_items=n_items)

        # Maxpair
        logger.info(f"\n Maxpair \n")
        dist = "maxpair"
        perf_list_maxpair, eps_list_maxpair1, eps_list_maxpair2, alt_eps_list_maxpair1, alt_eps_list_maxpair2, dict_res_training_maxpair = launch_exp(
            all_ranks, dist, p_torch, w, delta, thresholds_, epochs, dist_type_sym=dist_type_sym, norm_type=norm_type,
            nb_simu_training=nb_simu_training, save=save, exp_type=exp_type, n_items=n_items)


if __name__ == "__main__":

    my_config, args_ = get_config()
    logger.info(f"my_config = {my_config}")
    main_exp_launcher(my_config)
