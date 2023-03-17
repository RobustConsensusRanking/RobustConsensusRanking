import numpy as np
import os
from typing import NamedTuple
import torch
import itertools
import matplotlib.pyplot as plt
from launcher_classic import pairwise_matrix, expected_kendall, disymmetrized_hausdorff_on_kendall, proba_plackett_luce

import argparse
from utils._utils import get_logger

logger = get_logger("Classic Launcher")

device = "cpu"
default_tensor_type = torch.FloatTensor

# Arguments
class Config(NamedTuple):
    n : int = 10
    seed_val: int = 938
    m: int = 10
    threshold: float = 0.05


def str2bool(value):
    if value in [True, "True", "true"]:
        return True
    else:
        return False


def get_config():
    parser = argparse.ArgumentParser(description="Parser")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed_val", type=int, default=938)
    parser.add_argument("--m", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.05)

    args, _ = parser.parse_known_args()

    return Config(**args.__dict__), args.__dict__


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


def maxpair(P, threshold=0.):
    #P = pairwise_matrix(p, torch_all_ranks=torch_all_ranks, n=n)
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


def erm(P):
    #P = pairwise_matrix(p, torch_all_ranks=torch_all_ranks, n=n)
    return torch.round(P)


def main_exp_launcher(config):
    directory = f"{os.getcwd()}/get_distrib/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    n_items = config.n

    # All ranks
    file_path_all_ranks1 = f"torch_all_ranks_n={n_items}.pt"
    file_path_all_ranks2 = f"all_ranks_n={n_items}.pt"
    if os.path.isfile(directory+file_path_all_ranks1):
        torch_all_ranks = torch.load(directory+file_path_all_ranks1)
        all_ranks = torch.load(directory+file_path_all_ranks2)
    else:
        logger.info(f"Computing all ranks")
        all_ranks = list(itertools.permutations(list(np.arange(n_items))))
        all_ranks = [np.array(elem) for elem in all_ranks]
        torch_all_ranks = torch.from_numpy(np.asarray(all_ranks))
        logger.info(f"End all ranks, shape = {len(all_ranks)}")
        torch.save(torch_all_ranks, directory+file_path_all_ranks1)
        torch.save(all_ranks, directory+file_path_all_ranks2)

    # p_torch
    file_path_p_torch = f"p_torch_n={n_items}_seed={config.seed_val}_m={config.m}.pt"
    if os.path.isfile(directory+file_path_p_torch):
        p_torch = torch.load(directory+file_path_p_torch)
    else:
        logger.info(f"Computing p_torch")
        logits = torch.sort(torch.randn(n_items), descending=True)[0].numpy()
        w = np.exp(logits) ** config.m
        p = proba_plackett_luce(w, all_ranks, n_items=n_items)
        p_torch = torch.from_numpy(p)
        logger.info(f"End p_torch = {p_torch}, sum = {torch.sum(p_torch)}")
        torch.save(p_torch, directory+file_path_p_torch)

    # Pairwise mat P
    file_path_P = f"P_n={n_items}_seed={config.seed_val}_m={config.m}.pt"
    if os.path.isfile(directory+file_path_P):
        P = torch.load(directory+file_path_P)
    else:
        logger.info(f"Computing pairwise")
        P = pairwise_matrix(p_torch, torch_all_ranks, n=n_items)
        logger.info(f"End pairwise = {P}")
        torch.save(P, directory+file_path_P)


    # Compute stat (ERM or Maxpair)
    logger.info(f"Before computing stats")
    stat_p_erm = erm(P)
    stat_p_maxpair = maxpair(P, threshold=config.threshold)
    logger.info(f"Stat ERM = {stat_p_erm}, \n Maxpair = {stat_p_maxpair}")
    file_path_maxpair = f"maxpair_n={n_items}_seed={config.seed_val}_m={config.m}_threshold={config.threshold}.pt"
    file_path_erm= f"erm_n={n_items}_seed={config.seed_val}_m={config.m}.pt"
    torch.save(stat_p_erm, directory+file_path_erm)
    torch.save(stat_p_maxpair, directory+file_path_maxpair)



if __name__ == "__main__":

    my_config, args_ = get_config()
    logger.info(f"my_config = {my_config}")
    main_exp_launcher(my_config)
