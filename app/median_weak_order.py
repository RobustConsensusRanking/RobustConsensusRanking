from math import sqrt
from typing import NamedTuple
import mlflow
import argparse
import numpy as np
import gudhi

import torch
from matplotlib import pyplot as plt
from torch import argsort
from torch.autograd import Variable
from torch.distributions import MultivariateNormal
from torch.optim.lr_scheduler import LambdaLR

from depthfunction.metrics.ranking_metrics import kendall_tau, spearman_footrule, spearman_rho
from depthfunction.random_variables.ranking_rv import PlackettLuce
from depthfunction.smoothing.convolution_pg import smooth_pg_2
from depthfunction.argsort.thick_argsort import compute_thick_argsort, get_buckets, plot_persistence, tree_filtration, invert_permutation
from median_computation_torch import compute_median

from utils._utils import get_logger, device, default_tensor_type
from utils._mlflow_utils import set_mlflow, log_metric_mlflow, log_params_mlflow, log_param_mlflow

logger = get_logger("Thick argsort")

mlflow.set_tracking_uri("https://mlflow.par.prod.crto.in")
mlflow.set_experiment("mg_depth_function")

class Config(NamedTuple):
    use_mlflow : bool = False
    n : int = 5
    metric : str = "kendall_tau"
    std_dev : float = 1.0
    reg : float = 0.01
    steps : int = 1000
    logits: str = ""
    epsilon : float = 0.0


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

def argsort_from_median(config):
    scores = compute_median(config)
    weak_order, total_order, scores = compute_thick_argsort(scores, config.epsilon, config.use_mlflow)

    return weak_order, total_order, scores

if __name__ == "__main__":
    my_config, args_ = get_config()
    logger.info(f"my_config = {my_config}")
    if my_config.use_mlflow:
        with mlflow.start_run(run_name="Thick argsort", nested=True):
            set_mlflow(use_mlflow=my_config.use_mlflow)
            log_params_mlflow(args_, my_config.use_mlflow)
            wo, to, scores = argsort_from_median(my_config)
            logger.info(
                f"Source logits = {my_config.logits}, computed score = {scores}\n total order = {to}\n weak order = {wo}")
            log_param_mlflow("Optimal score", scores.detach().numpy(), use_mlflow = my_config.use_mlflow)
            log_param_mlflow("Total order", to, use_mlflow=my_config.use_mlflow)
            log_param_mlflow("Weak order", wo, use_mlflow=my_config.use_mlflow)
    else:
        wo, to, scores = argsort_from_median(my_config)
        logger.info(
            f"Source logits = {my_config.logits}, computed score = {scores}\n total order = {to}\n epsilon = {my_config.epsilon} and weak order = {wo}")

