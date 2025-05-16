from torch.utils.data import DataLoader


def get_dataloader(cfg: dict):
    """Return train, val, test DataLoaders based on split method and scenario."""
    method = cfg["data"]["split"]["method"]
    if method == "random":
        from .random_split import RandomSplitDataset as DS
    elif method == "by_state":
        from .by_state_split import StateSplitDataset as DS
    elif method == "by_climate":
        from .by_climate_split import ClimateSplitDataset as DS
    elif method == "by_tree_type":
        from .by_tree_split import TreeSplitDataset as DS
    else:
        raise ValueError(f"Unknown split method: {method}")

    train_ds = DS(cfg, split="train")
    val_ds   = DS(cfg, split="val")
    test_ds  = DS(cfg, split="test")

    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg['training'].get("batch_size",32), 
        shuffle=True, 
        num_workers=cfg['experiment'].get("num_workers", 16),
        pin_memory=True,
        prefetch_factor=2)
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg['training'].get("batch_size",32), 
        shuffle=False, 
        num_workers=cfg['experiment'].get("num_workers", 16),
        pin_memory=True,
        prefetch_factor=2)
    test_loader = DataLoader(
        test_ds, 
        batch_size=cfg['training'].get("batch_size",32), 
        shuffle=False, 
        num_workers=cfg['experiment'].get("num_workers", 16),
        pin_memory=True,
        prefetch_factor=2)
    return train_loader, val_loader, test_loader
