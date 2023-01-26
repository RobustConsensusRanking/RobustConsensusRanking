import argparse
import copy
import itertools
from typing import NamedTuple

import mlflow
import numpy as np
import torch

from utils._mlflow_utils import set_mlflow, log_params_mlflow
from utils._utils import get_logger

logger = get_logger("Depth Objective")

mlflow.set_tracking_uri("https://mlflow.par.prod.crto.in")
mlflow.set_experiment("mg_depth_obj")

class Config(NamedTuple):
    use_mlflow : bool = False
    n : int = 4
    metric : str = "kendall_tau"
    nb_loop : int = 1
    w : str = "0.48;0.28;0.16;0.08"
    w_change_ratio : str  = "100;60;20;0"
    threshold_val : float = 0.25


def str2bool(value):
    if value in [True, "True", "true"]:
        return True
    else:
        return False

def dict_custom_to_numpy(dico, n):
    M = np.eye(n) / 2
    for k,v in dico.items():
        _, i, j = k.split('_')
        M[int(i), int(j)] = v
        M[int(j), int(i)] = 1-v
    return M

def numpy_to_custom_dict(Q):
    dico = {}
    n = Q.shape[0]
    for i in np.arange(n):
        for j in np.arange(i+1, n):
            dico['p_'+str(i)+'_'+str(j)] = Q[i,j]
    return dico

def merge_idx(M, idx):
    # i,j \notin idx -> P_ij = M_ij
    # i \in idx, j \notin idx -> P_ij = \max_{k\in idx} M_kj
    # i \notin idx, j \in idx -> P_ij = \max_{k\in idx} M_ik
    # i,j \in idx -> P_ij = 0.5
    P = M
    for i in np.arange(M.shape[0]):
        m = np.max(M[i, idx])
        for j in idx:
            P[i,j] = m
    for j in np.arange(M.shape[0]):
        m = np.max(M[idx, j])
        for i in idx:
            P[i,j] = m
    for i in idx:
        for j in idx:
            P[i,j] = 0.5
    PTRIU = np.triu(P, 0)
    P = PTRIU + np.tril(1 - PTRIU.T, -1)
    return P

def get_config():
    parser = argparse.ArgumentParser(description="Parser")
    parser.add_argument("--use_mlflow", type=str2bool, default=False)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--metric", type=str, default="kendall_tau")

    parser.add_argument("--nb_loop", type=int, default=1)
    parser.add_argument("--w", type=str, default="0.48;0.28;0.16;0.08")
    parser.add_argument("--w_change_ratio", type=str, default="100;60;20;0")
    parser.add_argument("--threshold_val", type=float, default=0.25)

    args, _ = parser.parse_known_args()

    args.w = np.array(list(map(float, str(args.w).split(";"))))
    args.w_change_ratio = np.array(list(map(float, str(args.w_change_ratio).split(";"))))

    return Config(**args.__dict__), args.__dict__



###########################
###### All Functions ######
###########################

def kendall_tau_custom(y, sigma):
    y = torch.from_numpy(y)
    sigma = torch.from_numpy(sigma)
    n = sigma.size()[-1]
    sigma_inv = torch.argsort(sigma, dim=-1)
    y_inv = torch.argsort(y, dim=-1)
    sigma_pairs = (sigma_inv.unsqueeze(dim=-1) > sigma_inv.unsqueeze(dim=-2)).float()
    y_pairs = (y_inv.unsqueeze(dim=-1) > y_inv.unsqueeze(dim=-2)).float()
    return torch.sum(torch.abs(sigma_pairs-y_pairs), dim=[-2,-1])/2


def distance_matrix(all_ranks, metric):
    L = np.zeros((len(all_ranks),len(all_ranks)))
    for i, rank1 in enumerate(all_ranks):
        for j, rank2 in enumerate(all_ranks):
            L[i,j] = metric(rank1,rank2)
    return L


def get_all_buckets(all_ranks, metric):
    list_bucket = list()

    for rank1 in all_ranks:
        for rank2 in all_ranks:
            if metric(rank1,rank2) == 1.0:
                list_bucket.append( [rank1,rank2] )
    
    for i in np.arange(4):
        temp_ranks = list()
        for rank in all_ranks:
            if rank[3] == i:
                temp_ranks.append(rank)
        list_bucket.append(temp_ranks)
        
    list_bucket.append([np.array([0,1,2,3]), np.array([0,1,3,2]),np.array([1,0,2,3]),np.array([1,0,3,2])])
    list_bucket.append([np.array([0,2,1,3]), np.array([0,2,3,1]),np.array([2,0,1,3]),np.array([2,0,3,1])])
    list_bucket.append([np.array([0,3,1,2]), np.array([0,3,2,1]),np.array([3,0,1,2]),np.array([3,0,2,1])])
    list_bucket.append([np.array([1,2,0,3]), np.array([1,2,3,0]),np.array([2,1,0,3]),np.array([2,1,3,0])]) 
    list_bucket.append([np.array([1,3,0,2]), np.array([1,3,2,0]),np.array([3,1,0,2]),np.array([3,1,2,0])])
    list_bucket.append([np.array([2,3,0,1]), np.array([2,3,1,0]),np.array([3,2,0,1]),np.array([3,2,1,0])])

    list_bucket.append(all_ranks)
    for rank in all_ranks:
        list_bucket.append(list([rank]))
    
    return list_bucket


def bucket_distrib(all_ranks, bucket):
    l_ = list()
    div = len(bucket)
    for a_ in all_ranks:
        count = 0
        for b_ in bucket:
            count += 1
            if (b_ == a_).all():
                l_.append(1.0/div)
                count += 1
                continue
        if count == div:
            l_.append(0)
    return np.matrix(l_)


def depth_metric_optim(p, L, list_bucket, all_ranks, printer=True):
    val = np.inf
    for i, bucket in enumerate(list_bucket):
        q = bucket_distrib(all_ranks, bucket)
        val_ = np.linalg.norm(p*L - q*L, 1)
        if printer:
            logger.info(f"val: {val_} for {bucket}")
        if val_ <= val:
            best_distrib = q
            val = val_
    logger.info(f"Optimal solution for depth metric pb: {best_distrib}")
    return best_distrib, val


def pairwise_proba(all_ranks, p, n=4):
    dict_pairwise = dict()
    for i in range(n):
        for j in range(i+1,n):
            count = 0
            for idx, rank in enumerate(all_ranks):
                if np.where(rank == i)[0].item() < np.where(rank == j)[0].item():
                    count += p[0,idx]
            dict_pairwise['p_'+str(i)+'_'+str(j)] = count
    return dict_pairwise


def get_pairwise_proba(i, j, dict_pairwise):
    val_key = 'p_'+str(i)+'_'+str(j)
    val_key_bis = 'p_'+str(j)+'_'+str(i)
    all_keys = list(dict_pairwise.keys())
    if val_key in all_keys:
        return dict_pairwise[val_key]
    else:
        return 1.0 - dict_pairwise[val_key_bis]


def merge_items_(dict_pairwise, item1, item2):
    pair = 'p_'+str(item1)+'_'+str(item2)
    items_merge = [pair.split('_')[1], pair.split('_')[2]]
    logger.info(f"Merging items {items_merge} in the same bucket, key: {pair}")
    if (items_merge[1] in items_merge[0]) or (items_merge[0] in items_merge[1]):
        logger.info(f"Already merged")
        return dict_pairwise
    
    items1 = list()
    items2 = list()
    new_dict = dict()
    for key in dict_pairwise.keys():
        item1, item2 = key.split('_')[1], key.split('_')[2]
        if (item1 not in items_merge) and (item2 not in items_merge):
            new_dict[key] = dict_pairwise[key]
            
        elif (item1 not in items_merge):
            items1.append(str(item1))
        
        elif (item2 not in items_merge):
            items2.append(str(item2))
        
    for item1 in items1:
        if item1 not in items_merge:
            pair1 = 'p_'+item1+'_'+items_merge[0]
            if pair1 not in dict_pairwise.keys():
                continue
            val1 = dict_pairwise[pair1]
            pair2 = 'p_'+item1+'_'+items_merge[1]
            if pair2 not in dict_pairwise.keys():
                continue
            val2 = dict_pairwise[pair2]
            new_pair = 'p_'+item1+'_'+items_merge[0]+items_merge[1]
            new_val = (val1+val2)/2

            new_dict[new_pair] = new_val

    for item2 in items2:
        if item2 not in items_merge:
            pair1 = 'p_'+items_merge[0]+'_'+item2
            logger.info(f"pair1={pair1}")
            if pair1 not in dict_pairwise.keys():
                continue
            val1 = dict_pairwise[pair1]
            pair2 = 'p_'+items_merge[1]+'_'+item2
            logger.info(f"pair2={pair2}")
            if pair2 not in dict_pairwise.keys():
                continue
            logger.info(f"item2={item2}, pair1={pair1} and pair2={pair2}")
            val2 = dict_pairwise[pair2]
            new_pair = 'p_'+items_merge[0]+items_merge[1]+'_'+item2
            new_val = (val1+val2)/2

            new_dict[new_pair] = new_val
                
    dict_pairwise = new_dict
    return dict_pairwise


def unmerge_items_(dict_pairwise, item_to_remove, whole_bucket, side="better"):
    if not dict_pairwise:
        dict_pairwise = {'p_0123_0123':0.5}
    
    bucket_unmerged = copy.copy(whole_bucket)
    bucket_unmerged.remove(item_to_remove)
    logger.info(f"bucket_unmerged = {bucket_unmerged}")
    logger.info(f"join version whole = {''.join(str(item) for item in whole_bucket)}")
    
    items1 = list()
    items2 = list()
    new_dict = dict()
    for key in dict_pairwise.keys():
        logger.info(f"key={key}")
        item1, item2 = key.split('_')[1], key.split('_')[2]
        if (item1 not in ''.join(str(item) for item in whole_bucket)) and (item2 not in ''.join(str(item) for item in whole_bucket)):
            logger.info(f"here 1 : {item1}, {item2} and key {key}")
            new_dict[key] = dict_pairwise[key]
            
        elif (item1 not in ''.join(str(item) for item in whole_bucket)):
            items1.append(str(item1))
        
        elif (item2 not in ''.join(str(item) for item in whole_bucket)):
            items2.append(str(item2))
    logger.info(f"items1 = {items1} and items2 = {items2}")
    
    if item_to_remove < np.max(whole_bucket):
        pair_un = 'p_'+str(item_to_remove)+'_'+''.join(str(item) for item in bucket_unmerged)
        if side == "better":
            val_un = 1.0
        elif side == "worse":
            val_un = 0.0    
    else:
        pair_un = 'p_'+''.join(str(item) for item in bucket_unmerged)+'_'+str(item_to_remove)
        if side == "better":
            val_un = 0.0
        elif side == "worse":
            val_un = 1.0
    new_dict[pair_un] = val_un
    logger.info(f"new_dict = {new_dict}")
    
    for item1 in items1:
        if item1 not in whole_bucket:
            pair = 'p_'+item1+'_'+''.join(str(item) for item in bucket_unmerged)
            val = dict_pairwise['p_'+item1+'_'+''.join(str(item) for item in whole_bucket)]
            new_dict[pair] = val
            
            pairbis = 'p_'+item1+'_'+str(item_to_remove)
            new_dict[pairbis] = val
    
    for item2 in items2:
        if item2 not in whole_bucket:
            pair = 'p_'+''.join(str(item) for item in bucket_unmerged)+'_'+item2
            val = dict_pairwise['p_'+''.join(str(item) for item in whole_bucket)+'_'+item2]
            new_dict[pair] = val
            
            pairbis = 'p_'+str(item_to_remove)+'_'+item2
            new_dict[pairbis] = val
                       
    dict_pairwise = new_dict
    return dict_pairwise


def merging_algo(threshold_val, dict_pairwise):
    cont = True

    while cont:
        a, vala = np.argmin(np.abs(np.array(list((dict_pairwise.values()))) - 0.5)), np.min(np.abs(np.array(list((dict_pairwise.values()))) - 0.5))
        val = list((dict_pairwise.values()))[a]
        logger.info(f"Pairwise dict: {dict_pairwise}\nValue closest to 0.5: {val} ({vala})")
        if vala > threshold_val:
            cont = False
            continue

        pair = list(dict_pairwise.keys())[list(dict_pairwise.values()).index(val)]
        item1, item2 = [pair.split('_')[1], pair.split('_')[2]]
        
        dict_pairwise = merge_items_(dict_pairwise, item1, item2)
        if not bool(dict_pairwise):
            break
    return dict_pairwise


def idx_rank(list_ranks, all_ranks):
    list_idx = list()
    for rank in list_ranks:
        c_ = 0
        for elem in all_ranks:
            if (elem == rank).all():
                list_idx.append(c_)
                break
            c_ += 1
    return list_idx


def get_proba_idxs(idxs, shape=(1,24)):
    q = np.zeros(shape)
    for idx in idxs:
        q[0,idx] = 1.0/len(idxs)
    return q


def get_complete_pairwise_dict(pairwise_dict):
    new_dict = dict()
    
    if not pairwise_dict:
        return {'p_0_1':0.5, 'p_0_2':0.5, 'p_0_3':0.5, 'p_1_2':0.5, 'p_1_3':0.5, 'p_2_3':0.5}

    for key in pairwise_dict.keys():
        item1, item2 = key.split('_')[1], key.split('_')[2]
        if len(item1)<=1 and len(item2)<=1:
            new_dict[key] = pairwise_dict[key]
        else:
            for it2 in item2:
                for it1 in item1:
                    new_dict['p_'+str(it1)+'_'+str(it2)] = pairwise_dict[key]

        if len(item1) > 1:
            for i, item_ in enumerate(item1):
                for j in range(i+1,len(item1)):
                    new_dict['p_'+item_+'_'+item1[j]] = 0.5
        
        if len(item2) > 1:
            for i, item_ in enumerate(item2):
                for j in range(i+1,len(item2)):
                    new_dict['p_'+item_+'_'+item2[j]] = 0.5
    
    return new_dict


def clean_dict(pairwise_dict):
    if not pairwise_dict:
        return {'p_0_1':0.5, 'p_0_2':0.5, 'p_0_3':0.5, 'p_1_2':0.5, 'p_1_3':0.5, 'p_2_3':0.5}
        
    new_dict = dict()
    for key in pairwise_dict.keys():
        item1, item2 = key.split('_')[1], key.split('_')[2]
        if int(item1) <= int(item2):
            new_dict[key] = pairwise_dict[key]
        else:
            new_dict['p_'+item2+'_'+item1] = 1.0 - pairwise_dict[key]
    return new_dict


def get_readable_res(pairwise_dict):
    r_items = {0:0, 1:0, 2:0, 3:0}
    for item in range(4):
        for key in pairwise_dict.keys():
            item1, item2 = key.split('_')[1], key.split('_')[2]
            if (item == int(item1)) and  pairwise_dict[key] > 0.5:
                r_items[item] +=1
            elif (item == int(item2)) and  pairwise_dict[key] < 0.5:
                r_items[item] +=1
            elif ((item == int(item1)) or (item == int(item2))) and  pairwise_dict[key] == 0.5:
                r_items[item] += 0.5
    r_list = sorted(r_items, key=lambda k:r_items[k], reverse=True)
    logger.info(r_list)
    m = str(r_list[0])
    for i in range(1,4):
        if r_items[r_list[i-1]] > r_items[r_list[i]]:
            m += " > "
            m += str(r_list[i])
        elif r_items[r_list[i-1]] == r_items[r_list[i]]:
            m += "="
            m += str(r_list[i])
    return m


def proba_plackett_luce(w, all_ranks, n=4):
    list_proba = list()

    for sigma in all_ranks:
        val_ = list()
        for r in range(n):
            num_ = w[sigma[r]]
            denom_ = 0
            for s in range(r, n):
                v_ = w[sigma[s]]
                denom_ += v_
            val_.append(num_/denom_)
        proba_ = np.prod(val_)
        list_proba.append(proba_)
    return np.matrix(list_proba)


def update_bucket_(pairwise_dict, median, j):
    bucket = [median[0][j]]
    for key in pairwise_dict.keys():
        item1, item2 = key.split('_')[1], key.split('_')[2]
        if item1 == str(j) and pairwise_dict[key] == 0.5:
            bucket.append(median[0][int(item2)])
        if item2 == str(j) and pairwise_dict[key] == 0.5:
            bucket.append(median[0][int(item1)])
    return list(np.sort(bucket))

def generalized_bubblesort(p, L, all_ranks):

    pairwise_origin = pairwise_proba(all_ranks, p)
    idx_ = np.argmin(p*L)
    median = [all_ranks[idx_]]
    l_ = len(median[0])

    idxs = idx_rank(median, all_ranks)
    q_median = get_proba_idxs(idxs)
    q_pairwise_median_ = pairwise_proba(all_ranks, q_median)
    bucket = list()  

    for i in range(l_):
        for j in range(0,l_-i-1):
            q_pairwise_median = get_complete_pairwise_dict(q_pairwise_median_)
            q_pairwise_median = clean_dict(q_pairwise_median)
            
            bucket = update_bucket_(q_pairwise_median, median, j)
            bucket = list(np.unique(bucket))
            logger.info(f"\nBucket under study: {bucket} for j={j}, j+1={j+1}\n")

            logger.info(f"pairwise median = {q_pairwise_median}")      

            # Merge two items ?
            q_pairwise_merge_ = merge_items_(q_pairwise_median_, ''.join(str(item)for item in bucket), str(median[0][j+1]))
            q_pairwise_merge = get_complete_pairwise_dict(q_pairwise_merge_)
            q_pairwise_merge = clean_dict(q_pairwise_merge)
            logger.info(f"pairwise merge = {q_pairwise_merge} and {q_pairwise_merge_}")

            bucket.append(median[0][j+1])
            bucket = list(np.unique(bucket))

            # Unmerge two items ?
            q_pairwise_unmerge_ = unmerge_items_(q_pairwise_merge_, median[0][j+1], bucket, side="better")
            q_pairwise_unmerge = get_complete_pairwise_dict(q_pairwise_unmerge_)
            q_pairwise_unmerge = clean_dict(q_pairwise_unmerge)
            logger.info(f"pairwise unmerge = {q_pairwise_unmerge}")

            comp_median = np.sum(list({key: np.abs(pairwise_origin[key] - q_pairwise_median[key]) for key in pairwise_origin.keys()}.values()))
            comp_merge = np.sum(list({key: np.abs(pairwise_origin[key] - q_pairwise_merge[key]) for key in pairwise_origin.keys()}.values()))
            comp_unmerge = np.sum(list({key: np.abs(pairwise_origin[key] - q_pairwise_unmerge[key]) for key in pairwise_origin.keys()}.values()))

            if comp_median <= comp_merge:
                logger.info(f"keeping median")
                #bucket.remove(median[0][j+1])
                q_pairwise_median_ = q_pairwise_median_
            else:
                if comp_merge <= comp_unmerge:
                    logger.info(f"merging")
                    q_pairwise_median_ = q_pairwise_merge_  
                else:
                    logger.info(f"unmerging")
                    #bucket.remove(median[0][j])
                    q_pairwise_median_ = q_pairwise_unmerge_
            logger.info(f"Result pairwise median = {q_pairwise_median_}")   

    q_pairwise_median = get_complete_pairwise_dict(q_pairwise_median_)
    return q_pairwise_median





def maxpair(P, threshold = 0.):
    s = np.sum(P, axis=1)
    sigma = np.argsort(-s)
    M = P[np.ix_(sigma,sigma)]
    m = np.max(np.abs(M-0.5)*(M!=0.5)*(np.abs(M-0.5) <= threshold))
    while m > 0:
        i,j = np.argwhere(np.abs(M-0.5) == m)[0]
        M = merge_idx(M, np.arange(i,j+1))
        m = np.max(np.abs(M - 0.5) * (M != 0.5) * (np.abs(M - 0.5) <= threshold))
    return M[np.ix_(np.argsort(sigma), np.argsort(sigma))]

############################
##### Main experiments #####
############################


def launch_exp(config):
    # Metric
    if config.metric == "kendall_tau":
        metric = kendall_tau_custom
    else:
        logger.info(f"Metric {config.metric} not implemented yet")

       
    n = config.n

    all_ranks = list(itertools.permutations(list(np.arange(n))))
    all_ranks = [np.array(elem) for elem in all_ranks]
    list_bucket = get_all_buckets(all_ranks, metric)
    L = distance_matrix(all_ranks, metric)

    if config.nb_loop == 1:
        results = dict()

        w = config.w
        p = proba_plackett_luce(w, all_ranks)
        logger.info(f"Solution of real distance pb")
        best_distrib, val = depth_metric_optim(p, L, list_bucket, all_ranks, printer=False)
        d_ = pairwise_proba(all_ranks, best_distrib, n=config.n)
        d_real_solution = get_complete_pairwise_dict(d_)
        d_real_solution = clean_dict(d_real_solution)
        readable_real_solution = get_readable_res(d_real_solution)
        logger.info(f"Pairise solution \n {d_real_solution}")
        logger.info(f"\n")

        logger.info(f"Solution of merging pb")
        dict_pairwise = pairwise_proba(all_ranks, p)
        threshold_val = 0.25
        new_dict_ = merging_algo(threshold_val, dict_pairwise)
        new_dict = dict()
        for key in new_dict_.keys():
            if new_dict_[key] > 0.5:
                new_dict[key] = 1.0
            elif new_dict_[key] < 0.5:
                new_dict[key] = 0.0
            elif new_dict_[key] == 0.5:
                new_dict[key] = 0.5
        d_merging_solution = get_complete_pairwise_dict(new_dict)
        d_merging_solution = clean_dict(d_merging_solution)
        readable_merging_solution = get_readable_res(d_merging_solution)
        logger.info(f"\n")
        
        logger.info(f"Solution of generalized BubbleSort")
        bubblesort_pairwise = generalized_bubblesort(p, L, all_ranks)
        d_bubblesort_solution = get_complete_pairwise_dict(bubblesort_pairwise)
        d_bubblesort_solution = clean_dict(d_bubblesort_solution)
        readable_bubblesort_solution = get_readable_res(d_bubblesort_solution)
        logger.info(f"\n\n")

        logger.info(f"Solution of max pair merging")
        P = dict_custom_to_numpy(dict_pairwise, n=4)
        Q = maxpair(P, threshold = 0.251)
        maxpair_pairwise = numpy_to_custom_dict(Q)
        d_maxpair_solution = get_complete_pairwise_dict(maxpair_pairwise)
        d_maxpair_solution = clean_dict(d_maxpair_solution)
        readable_maxpair_solution = get_readable_res(d_maxpair_solution)
        logger.info(f"\n\n")

        res_message = f"GENERAL SOLUTIONS\nReal: {readable_real_solution}\nMerging: {readable_merging_solution} " \
                      f"({readable_real_solution==readable_merging_solution})\n" \
                      f"BubbleSort: {readable_bubblesort_solution} ({readable_real_solution==readable_bubblesort_solution})\n"\
                      f"MaxPair: {readable_maxpair_solution} ({readable_maxpair_solution == readable_real_solution})"
        logger.info(res_message)
        results[str(w)] = res_message
        logger.info(f"\n\n\n\n")



    if config.nb_loop > 1:
        results = dict()
        for a_ in np.linspace(0, 20, config.nb_loop)[1:]:
            w = np.array([a_, a_, a_, a_]) + config.w_change_ratio
            w = w/np.sum(w)
            logger.info(f"Weigts = {w}\n")

            p = proba_plackett_luce(w, all_ranks)
            logger.info(f"Solution of real distance pb")
            best_distrib, val = depth_metric_optim(p, L, list_bucket, all_ranks, printer=False)
            d_ = pairwise_proba(all_ranks, best_distrib, n=config.n)
            d_real_solution = get_complete_pairwise_dict(d_)
            d_real_solution = clean_dict(d_real_solution)
            readable_real_solution = get_readable_res(d_real_solution)
            logger.info(f"Pairise solution \n {d_real_solution}")
            logger.info(f"\n")

            logger.info(f"Solution of merging pb")
            dict_pairwise = pairwise_proba(all_ranks, p)
            threshold_val = 0.25
            new_dict_ = merging_algo(threshold_val, dict_pairwise)
            new_dict = dict()
            for key in new_dict_.keys():
                if new_dict_[key] > 0.5:
                    new_dict[key] = 1.0
                elif new_dict_[key] < 0.5:
                    new_dict[key] = 0.0
                elif new_dict_[key] == 0.5:
                    new_dict[key] = 0.5
            d_merging_solution = get_complete_pairwise_dict(new_dict)
            d_merging_solution = clean_dict(d_merging_solution)
            readable_merging_solution = get_readable_res(d_merging_solution)
            logger.info(f"\n")
            
            logger.info(f"Solution of generalized BubbleSort")
            bubblesort_pairwise = generalized_bubblesort(p, L, all_ranks)
            d_bubblesort_solution = get_complete_pairwise_dict(bubblesort_pairwise)
            d_bubblesort_solution = clean_dict(d_bubblesort_solution)
            readable_bubblesort_solution = get_readable_res(d_bubblesort_solution)
            logger.info(f"\n\n")
            
            res_message = f"GENERAL SOLUTIONS\nReal: {readable_real_solution}\nMerging: {readable_merging_solution} ({readable_real_solution==readable_merging_solution})\nBubbleSort: {readable_bubblesort_solution} ({readable_real_solution==readable_bubblesort_solution})"
            logger.info(res_message)
            results[str(w)] = res_message
            logger.info(f"\n\n\n\n")

        for key in results.keys():
            logger.info(f"Result for {key}:")
            logger.info(f"{results[key]}\n\n")








if __name__ == "__main__":
    my_config, args_ = get_config()
    logger.info(f"my_config = {my_config}")
    if my_config.use_mlflow:
        with mlflow.start_run(run_name="Depth objective - comparison of algos", nested=True):
            set_mlflow(use_mlflow=my_config.use_mlflow)
            log_params_mlflow(args_, my_config.use_mlflow)
            launch_exp(my_config)
            
    else:
        launch_exp(my_config)