import os
import yaml
import argparse
import logging
import random
import numpy as np

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Dead Tree Dataset Training and Evaluation Pipeline")
    parser.add_argument(
        "--config", type=str, default="configs/debug.yaml",
        help="Path to the YAML configuration file"
    )
    parser.add_argument(
        "--exp_id", type=str, default="0",
        help="Experiment ID for logging and output directory"
    )
    parser.add_argument(
        "--model_name", type=str, default="unet",
        help="Model name from a list of [unet, deeplabv3, d3_tf, vitseg, segformer, dofa]"
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0,
        help="GPU ID to use for training and evaluation"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8,
        help="Ratio of training data to total data in random split"
    )    
    parser.add_argument(
        "--overwrite_cfg", type=bool, default=False,
        help="If True, overwrite the config with command-line arguments"
    )
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg
  
  
def overwrite_config(args, cfg):
    """Overwrite config with command-line arguments."""
    cfg['experiment']['id'] = args.exp_id
    cfg['model']['name'] = args.model_name
    cfg['experiment']['gpu_id'] = args.gpu_id
    cfg['data']['split']['random']['train_ratio'] = args.train_ratio
    return cfg


def setup_logging(log_dir, log_name, level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{log_name}.log")

    # Create a formatter that includes time, level, and message
    fmt = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    # File handler (writes to exp_name.log)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(formatter)
    fh.setLevel(level)

    # Console handler (prints to stdout)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(level)

    # Root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def set_seed(seed: int):
  """Set random seed for reproducibility."""
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  
  
def save_results(metrics: dict, exp_name: str, output_dir: str):
    """Write out your metrics dict as a YAML file."""
    os.makedirs(output_dir, exist_ok=True)
    metrics = {k: float(v) if isinstance(v, np.generic) else v for k, v in metrics.items()}
    out_path = os.path.join(output_dir, f"{exp_name}_metrics.yaml")
    with open(out_path, "w") as f:
        yaml.dump(metrics, f)

