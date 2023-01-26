from math import sqrt
from typing import NamedTuple
import mlflow
import argparse
import numpy as np
import pickle

import torch
from matplotlib import pyplot as plt
from torch import argsort
from torch.autograd import Variable
from torch.distributions import MultivariateNormal
from torch.optim.lr_scheduler import LambdaLR

from depthfunction.metrics.ranking_metrics import kendall_tau, spearman_footrule, spearman_rho
from depthfunction.random_variables.ranking_rv import PlackettLuce, Thurstonian
from depthfunction.smoothing.convolution_pg import smooth_pg_2

from utils._utils import get_logger, device, default_tensor_type
from utils._mlflow_utils import set_mlflow, log_metric_mlflow, log_params_mlflow

logger = get_logger("Trimming computation SGD")

class Config(NamedTuple):
    use_mlflow : bool = False
    n : int = 5
    metric : str = "kendall_tau"
    std_dev : float = 1.0
    reg : float = 0.01
    steps : int = 10000


def str2bool(value):
    if value in [True, "True", "true"]:
        return True
    else:
        return False

def get_config():
    parser = argparse.ArgumentParser(description="Parser")
    parser.add_argument("--use_mlflow", type=str2bool, default=False)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--metric", type=str, default="kendall_tau")
    parser.add_argument("--std_dev", type=float, default=1.0)
    parser.add_argument("--reg", type=float, default=0.001)
    parser.add_argument("--steps", type=int, default=1000)

    args, _ = parser.parse_known_args()

    return Config(**args.__dict__), args.__dict__

def optim(metric, n, logits, rv_y, depth_threshold=0, std_dev=10, reg=0.01, steps=100, use_mlflow=False):
    # Smoothing
    kernel_conv = lambda _s: MultivariateNormal(_s, std_dev * torch.eye(_s.size()[-1]))
    smoothed_metric = lambda y, s: smooth_pg_2(lambda _y, _s: metric(_y, argsort(_s, dim=-1, descending=True)),
                                               kernel_conv)(y, s)
    val_max = 1

    # Initial scores
    #s = Variable(-torch.arange(n).float(), requires_grad=True).to(device).type(default_tensor_type)
    s = Variable(torch.zeros(n).float(), requires_grad=True).to(device).type(default_tensor_type)

    # Optimization
    optimizer = torch.optim.SGD([s], lr=1., momentum=0.9)
    scheduler = LambdaLR(optimizer, lambda _t: 1. / sqrt(_t + 1))
    losses = []
    previous_data = []
    for t in range(steps):
        optimizer.zero_grad()
        y = rv_y.sample().squeeze(-1).to(device).type(default_tensor_type)
        #if t == 0:
        #    logger.info(f"Step {t} : first elem = {y}")
        depth_y = compute_depth(metric, y, previous_data, val_max)
        if depth_y > depth_threshold:
            previous_data.append(y)
            loss = torch.sum(smoothed_metric(y, s)) + reg * torch.linalg.norm(s, dim=-1, ord=2) ** 2
            loss.backward()
            optimizer.step()
            scheduler.step()
            if t % (steps / 10) == 0:
                l = 0
                l_opt = 0
                M = 1000
                for i in range(M):
                    y = rv_y.sample().squeeze(-1).to(device).type(default_tensor_type)
                    f = torch.sum(metric(y, argsort(s, descending=True)))
                    f_opt = torch.sum(metric(y, argsort(logits, descending=True)))
                    l += f
                    l_opt += f_opt
                #logger.info(f"step {t}: s = {argsort(s, descending=True)}")
                log_metric_mlflow("Excess loss", np.round(l / M - l_opt / M, 3), step=t, use_mlflow=use_mlflow)
                losses.append(l / M - l_opt / M)
        #else:
            #logger.info(f"Rejected pt : {y} (depth={depth_y})")

    return losses, s

def compute_depth(metric, y, previous_data, val_max):
    d_ = 0
    if len(previous_data) > 0:
        for x in previous_data:
            d_ += metric(y, x)
        return val_max - d_/len(previous_data)
    else:
        return val_max



def compute_median(config, dt_, w_):
    # Distribution on Y
    logits = torch.arange(config.n).to(device).type(default_tensor_type)

    m = 2
    mu = torch.concat((logits,torch.flip(logits,(0,)))).reshape(m,-1)
    sigma = torch.cat([torch.tensor([1.0])]*config.n + [torch.tensor([0.75])]*config.n).reshape(m,-1)
    weight = torch.tensor([w_, 1-w_])
    rv_y = Thurstonian(config.n, m, mu, sigma, weight)

    #rv_y = PlackettLuce(logits)

    # Metric
    if config.metric == "kendall_tau":
        metric = kendall_tau
    elif config.metric == "spearman_rho":
        metric = spearman_rho
    elif config.metric == "spearman_foortule":
        metric = spearman_footrule
    else:
        logger.info(f"Metric {config.metric} not implemented")

    # Smoothing
    losses, s = optim(metric, config.n, logits, rv_y, depth_threshold=dt_, std_dev=config.std_dev, reg=config.reg, steps=config.steps)

    #logger.info(f"Final s = {argsort(s, descending=True)}")
    # Plot
    #plt.plot(losses)
    #plt.ylabel('Distance')
    #plt.xlabel('Steps')
    #plt.show()
    return s


if __name__ == "__main__":
    my_config, args_ = get_config()
    logger.info(f"my_config = {my_config}")
    res_dict = dict()
    threshold_param = [0, 0.2, 0.5, 0.8]
    weight_param = [1.0, 0.9, 0.8]

    if my_config.use_mlflow:
        with mlflow.start_run(run_name="Median computation SGD", nested=True):
            set_mlflow(use_mlflow=my_config.use_mlflow)
            log_params_mlflow(args_, my_config.use_mlflow)
            for t_ in threshold_param:
                logger.info(f"Threshold {t_}")
                res_dict["threshold_"+str(t_)] = dict()
                for w_ in weight_param:
                    logger.info(f"    Weight {w_}")
                    res_dict["threshold_" + str(t_)]["weight_"+str(w_)] = list()
                    for i in range(20):
                        logger.info(f"        Iteration {i}")
                        s = compute_median(my_config, t_, w_)
                        res_dict["threshold_" + str(t_)]["weight_" + str(w_)].append(argsort(s, descending=True))
        a_file = open(
            "/Users/m.goibert/Documents/Criteo/Projets_Recherche/P5_General_Depth_Fct/depth-functions/data.pkl", "wb")
        pickle.dump(res_dict, a_file)
        a_file.close()
    else:
        for t_ in threshold_param:
            logger.info(f"Threshold {t_}")
            res_dict["threshold_" + str(t_)] = dict()
            for w_ in weight_param:
                logger.info(f"    Weight {w_}")
                res_dict["threshold_" + str(t_)]["weight_" + str(w_)] = list()
                for i in range(20):
                    logger.info(f"        Iteration {i}")
                    s = compute_median(my_config, t_, w_)
                    res_dict["threshold_" + str(t_)]["weight_" + str(w_)].append(argsort(s, descending=True))
        a_file = open("/Users/m.goibert/Documents/Criteo/Projets_Recherche/P5_General_Depth_Fct/depth-functions/data.pkl", "wb")
        pickle.dump(res_dict, a_file)
        a_file.close()