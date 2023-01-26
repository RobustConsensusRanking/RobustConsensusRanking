from typing import NamedTuple
import mlflow
import argparse

import torch
from matplotlib import pyplot as plt
from torch import argsort
from torch.autograd import Variable
from torch.distributions import MultivariateNormal, Categorical, MixtureSameFamily
from torch.optim.lr_scheduler import LambdaLR

from depthfunction.metrics.ranking_metrics import spearman_footrule, spearman_rho, kendall_tau
from depthfunction.random_variables.ranking_rv import PlackettLuce, Dirac
from depthfunction.smoothing.convolution_pg import smooth_pg, smooth_pg_2

from utils._utils import get_logger, device, default_tensor_type
from utils._mlflow_utils import set_mlflow, log_param_mlflow

logger = get_logger("Median computation SGD")

class Config(NamedTuple):
    use_mlflow : bool = False
    n : int = 5
    metric : str = "kendall_tau"
    std_dev : float = 1.0
    reg : float = 0.01
    sample_size : int = 100000
    nb_distrib : int = 3
    weight_distrib : list = [0.5, 2e-1]


def str2bool(value):
    if value in [True, "True", "true"]:
        return True
    else:
        return False

def get_config():
    parser = argparse.ArgumentParser(description="Parser")
    parser.add_argument("--use_mlflow", type=str2bool, default=False)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--nb_distrib", type=int, default=3)
    parser.add_argument("--weight_distrib", type=list, default=[0.5, 2e-1])
    parser.add_argument("--metric", type=str, default="kendall_tau")
    parser.add_argument("--std_dev", type=float, default=1.0)
    parser.add_argument("--reg", type=float, default=0.01)
    parser.add_argument("--sample_size", type=int, default=10000)

    args, _ = parser.parse_known_args()

    return Config(**args.__dict__), args.__dict__


def compute_expected_loss(s, rv_y, metric, s_star, M=1000):
    l = 0
    l_opt = 0
    for i in range(M):
        y = rv_y.sample()
        f = metric(y, argsort(s, descending=True))
        f_opt = metric(y, argsort(s_star, descending=True))
        l += f
        l_opt += f_opt
        l_opt = 0
    return l / M - l_opt / M


def init_distrib(n, nb_distrib=3, weight_distrib=[0.5, 2e-1]):

    # distribution on Y
    logits = torch.arange(n)[None, :] * torch.ones(nb_distrib)[:, None]
    logits[1, [0, -1]] = logits[1, [-1, 0]]
    comp = PlackettLuce(logits)
    mix = Categorical(torch.Tensor([1 - weight_distrib[0] - weight_distrib[1], weight_distrib[0], weight_distrib[1]]))
    rv_y = MixtureSameFamily(mix, comp)

    return rv_y, logits

def optim(rv_y, metric, n, logits, std_dev=1.0, sample_size=100000, use_mlflow=False):
    kernel_conv = lambda _s: MultivariateNormal(_s, std_dev * torch.eye(_s.size()[-1]))
    smoothed_metric = lambda y, s: smooth_pg_2(lambda _y, _s: metric(_y, argsort(_s, dim=-1, descending=True)), kernel_conv)(y, s)

    # initial scores
    s = Variable(torch.zeros(n).float(), requires_grad=True)
    optimizer = torch.optim.SGD([s], lr=1., momentum=0.)
    losses = []
    val_before = compute_expected_loss(s, rv_y, metric, logits)
    log_param_mlflow("Loss before one-shot", val_before, use_mlflow=use_mlflow)
    losses.append(val_before)

    optimizer.zero_grad()
    y = rv_y.sample(sample_shape=(sample_size,))
    loss = torch.sum(smoothed_metric(y, s))
    loss.backward()
    print(s.grad.data)
    optimizer.step()

    val_after = compute_expected_loss(s, rv_y, metric, logits, M=10000)
    log_param_mlflow("Loss after one-shot", val_after, use_mlflow=use_mlflow)
    losses.append(val_after)

    print(losses)
    print(argsort(s, descending=True))

    return losses

def one_shot_comput(config):
    # Distribution on Y
    rv_y, logits = init_distrib(config.n, nb_distrib=config.nb_distrib, weight_distrib=config.weight_distrib)

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
    losses = optim(rv_y, metric, config.n, logits, std_dev=config.std_dev, sample_size=config.sample_size,
                    use_mlflow=config.use_mlflow)

    return losses

if __name__ == "__main__":
    my_config, args_ = get_config()
    logger.info(f"my_config = {my_config}")
    if my_config.use_mlflow:
        with mlflow.start_run(run_name="Median computation SGD", nested=True):
            set_mlflow(use_mlflow=my_config.use_mlflow)
            log_params_mlflow(args_, my_config.use_mlflow)
            one_shot_comput(my_config)
    else:
        one_shot_comput(my_config)