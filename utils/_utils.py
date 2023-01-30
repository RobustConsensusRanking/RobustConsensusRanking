import torch
import os
import logging
from pathlib import Path

# Rootpath
rootpath = "{}".format(os.path.dirname(os.path.abspath(__file__)))

# Define logger
logging.basicConfig(level=logging.INFO)
root_logger = logging.getLogger()
root_logger.handlers = list()

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    my_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s -- %(name)s - %(levelname)s -- %(message)s')
    my_handler.setFormatter(formatter)
    logger.handlers = [my_handler]

    return logger

# Device
logger = get_logger("Device")
nb_cuda_devices = torch.cuda.device_count()
if nb_cuda_devices > 0:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
if os.environ.get("FORCE_CPU", "0") == "1":
    device = torch.device("cpu")
logger.info(f"Device is {device}")

# Default precision
default_tensor_type = torch.FloatTensor
