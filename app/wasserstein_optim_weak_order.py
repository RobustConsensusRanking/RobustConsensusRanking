from math import sqrt
from typing import NamedTuple
import mlflow
import argparse
import numpy as np
from geomloss import SamplesLoss

import torch
from matplotlib import pyplot as plt
from torch import argsort
from torch.autograd import Variable
from torch.distributions import MultivariateNormal
from torch.optim.lr_scheduler import LambdaLR

from depthfunction.metrics.ranking_metrics import kendall_tau, spearman_footrule, spearman_rho
from depthfunction.random_variables.ranking_rv import PlackettLuce, BucketRk
from depthfunction.smoothing.convolution_pg import smooth_pg_2
from depthfunction.argsort.thick_argsort import compute_thick_argsort, get_buckets, plot_persistence, tree_filtration, invert_permutation


from utils._utils import get_logger, device, default_tensor_type
from utils._mlflow_utils import set_mlflow, log_metric_mlflow, log_params_mlflow, log_param_mlflow

logger = get_logger("Optim Depth - SGD")

mlflow.set_tracking_uri("https://mlflow.par.prod.crto.in")
mlflow.set_experiment("mg_depth_function")

class Config(NamedTuple):
    use_mlflow : bool = False
    n : int = 5
    metric : str = "kendall_tau"
    std_dev : float = 1.0
    reg : float = 0.01
    steps : int = 10000
    logits : str = ""
    epsilon: float = 0.0

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
    parser.add_argument("--reg", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--logits", type=str, default="")
    parser.add_argument("--epsilon", type=float, default=0.0)

    args, _ = parser.parse_known_args()

    if len(args.logits) > 0:
        args.logits = torch.tensor(list(map(float, str(args.logits).split(";"))))

    return Config(**args.__dict__), args.__dict__



def depth_optim(epsilon, metric, n, logits, rv_y=None, std_dev=10, reg=0.01, steps=10000, use_mlflow=False):
    # Source distribution
    if len(logits) == 0  and rv_y == None:
        logits = torch.arange(n).to(device).type(default_tensor_type)
        rv_y = PlackettLuce(logits)
    elif rv_y == None:
        rv_y = PlackettLuce(logits)
    logger.info(f"Logits = {logits}")

    # Wasserstein loss function
    loss_function = SamplesLoss("sinkhorn", p=2, blur=1.0)

    # Initial scores
    s = Variable(torch.zeros(n).float(), requires_grad=True).to(device).type(default_tensor_type)
    s.requires_grad = True

    # Optimization
    optimizer = torch.optim.SGD([s], lr=0.01, momentum=0.9)
    scheduler = LambdaLR(optimizer, lambda _t: 1. / sqrt(_t + 1))
    losses = []
    score_losses = []
    current_score_loss = []
    for t in range(steps):
        weak_order, total_order, _ = compute_thick_argsort(s, epsilon, use_mlflow)

        optimizer.zero_grad()
        y = rv_y.sample().to(device).type(default_tensor_type)
        y.requires_grad = True
        loss = -loss_function(y.view(1,-1), s.view(1,-1)) + reg * torch.linalg.norm(s, dim=-1, ord=2) ** 2
        loss.backward()
        optimizer.step()
        scheduler.step()
        current_score_loss.append(loss.detach().numpy())
        if t % (steps / 200) == 0:
            score_losses.append(np.mean(current_score_loss))
            log_metric_mlflow("Score loss", np.mean(current_score_loss), step=t, use_mlflow=use_mlflow)
            current_score_loss = []
            logger.info(f"s = {s} and sigma = {argsort(s, descending=True)} and weak order = {weak_order}")
            l = 0
            l_opt = 0
            M = 1000
            for i in range(M):
                y = rv_y.sample().to(device).type(default_tensor_type)
                f = metric(y, argsort(s, descending=True))
                f_opt = metric(y, argsort(logits, descending=True))
                l += f
                l_opt += f_opt
            log_metric_mlflow("Excess ranking loss", np.round(l / M - l_opt / M, 3), step=t, use_mlflow=use_mlflow)
            losses.append(l / M - l_opt / M)

    weak_order, total_order, _ = compute_thick_argsort(s, epsilon, use_mlflow)
    logger.info(f"Optimal score = {s} and weak order = {weak_order}")
    return losses, score_losses, s, weak_order, total_order


def compute_median(config):
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
    losses, score_losses, s, wo, to = depth_optim(config.epsilon, metric, config.n, config.logits, rv_y=None, std_dev=config.std_dev, reg=config.reg, steps=config.steps)

    # Plot
    steps_loss = [t for t in range(config.steps) if t % (config.steps/200) == 0]
    plt.plot(steps_loss, losses/np.linalg.norm(losses, ord=np.inf), label="Rk loss")
    plt.plot(steps_loss, score_losses/np.linalg.norm(score_losses, ord=np.inf), label="Score loss")
    plt.ylabel('Distance')
    plt.xlabel('Steps')
    plt.show(block=False)
    plt.pause(30)
    plt.close()

    # Optimal score output
    return s, wo, to


if __name__ == "__main__":
    my_config, args_ = get_config()
    logger.info(f"my_config = {my_config}")
    if my_config.use_mlflow:
        with mlflow.start_run(run_name="Wasserstein minimization SGD", nested=True):
            set_mlflow(use_mlflow=my_config.use_mlflow)
            log_params_mlflow(args_, my_config.use_mlflow)
            s, wo, to = compute_median(my_config)
            log_param_mlflow("Optimal score", s.detach().numpy(), use_mlflow=my_config.use_mlflow)
            log_param_mlflow("Total order", to, use_mlflow=my_config.use_mlflow)
            log_param_mlflow("Weak order", wo, use_mlflow=my_config.use_mlflow)
    else:
        compute_median(my_config)