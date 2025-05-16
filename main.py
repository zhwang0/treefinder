#!/usr/bin/env python3
"""
main.py: Training and evaluation pipeline for dead tree dataset.

This script supports flexible train-test splits (random, geographic, state-based) and multiple model backbones (CNNs, ViTs, SegFormer, etc.).
Usage:
    python main.py --config configs.yaml --exp_name exp1

Recommendation: use a YAML config (e.g., configs.yaml) to centralize parameters and ensure reproducibility, following common ML conventions.
"""


import yaml
import logging
from pathlib import Path
import time

from utils.tools import (
    parse_args, load_config, overwrite_config,
    setup_logging, set_seed, save_results
)
from data_loader import get_dataloader
from models import get_model
from exps.train import train_model
from exps.evaluate import evaluate_model


def main():
    st_time = time.time()
    
    # Parse command-line args and load config
    args = parse_args()
    cfg = load_config(args.config)
    if args.overwrite_cfg:
        cfg = overwrite_config(args, cfg)
    
    exp_name = cfg['experiment']['id'].zfill(3)
    exp_name = f"exp{exp_name}_{cfg['model']['name']}"

    # Log experiment configuration
    logger = setup_logging(cfg['logging']['log_dir'], exp_name)
    logger.info(f"Starting experiment: {exp_name}")
    logger.info("Configuration:\n" + yaml.dump(cfg, sort_keys=False))
    
    # Set random seed for reproducibility
    set_seed(cfg['experiment']['seed'])

    # Prepare data loaders
    train_loader, val_loader, test_loader = get_dataloader(cfg)
    
    # Initialize model
    model = get_model(cfg['model'])
    
    # Train model
    train_metrics = train_model(
        model,
        train_loader,
        val_loader,
        cfg,
        exp_name
    )

    # Evaluate model
    eval_metrics = evaluate_model(
        model,
        test_loader,
        cfg,
        exp_name
    )

    # Combine and save metrics
    results_root = Path(cfg['output']['results_dir']) / exp_name
    all_metrics = {**train_metrics, **eval_metrics}
    save_results(all_metrics, exp_name, str(results_root))
    logger.info(f"Spent {(time.time()-st_time)/3600:.1f}h: Experiment {exp_name} completed. Combined metrics: {all_metrics}")


if __name__ == "__main__":
    main()
