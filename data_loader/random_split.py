# File: data_loader/random_split.py
import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from .utils import load_tile, augment_tile


class RandomSplitDataset(Dataset):
    def __init__(self, cfg: dict, split: str = 'train'):
        # Persist config
        self.cfg = cfg
        self.split = split
        
        train_ratio = cfg['data']['split']['random']['train_ratio']       # e.g., 0.6
        test_ratio = cfg['data']['split']['random']['test_ratio']         # e.g., 0.2
        train_val_ratio = cfg['data']['split']['random']['train_val_ratio']  # e.g., 0.1
        
        assert 0 < train_ratio < 1 and 0 <= test_ratio < 1
        assert train_ratio + test_ratio <= 1.0
        
        info = pd.read_csv(os.path.join(cfg['data']['root_dir'], cfg['data']['dataset_info']))
        shuffle_att = 'FileName' if cfg['data']['split']['random'].get('shuffle_by_tile', True) else 'ImageRawPath'
        
        
        # Shuffle metadata
        all_tiles = info[shuffle_att].unique().tolist()
        rng = np.random.RandomState(cfg['experiment']['seed'])
        rng.shuffle(all_tiles)
        
        
        # --- Fixed Test Split ---
        n_total = len(all_tiles)
        n_test = int(test_ratio * n_total)
        test_tiles_raw = all_tiles[:n_test]
        test_df = info[info[shuffle_att].isin(test_tiles_raw)]
        test_tiles = test_df['FileName'].unique().tolist() # change to tile names

        
        # --- Train/Val Pool ---
        n_trainval = int(train_ratio * n_total)
        trainval_tiles = all_tiles[n_test:n_test + n_trainval]
        trainval_df = info[info[shuffle_att].isin(trainval_tiles)]
        n_trainval = len(trainval_df) # convert to number of tiles
        
        # --- Positive/Negative Split ---
        pos_threshold = cfg['data']['split'].get('pos_threshold', 0)
        pos_frac = cfg['data']['split'].get('pos_frac', 0)

        is_pos = trainval_df['LabelSize'] >= pos_threshold
        pos_tiles = trainval_df[is_pos]['FileName'].unique().tolist() # change to tile names
        neg_tiles = trainval_df[~is_pos]['FileName'].unique().tolist()
        rng.shuffle(pos_tiles)
        rng.shuffle(neg_tiles)
        
        if pos_frac > 0:
            n_pos = min(int(pos_frac * n_trainval), len(pos_tiles))
            n_neg = min(int(n_pos / pos_frac - n_pos), len(neg_tiles)) # make sure the pos/neg ratio is correct
        else:
            # using all data
            n_pos = len(pos_tiles)
            n_neg = len(neg_tiles)

        
        selected_pos = pos_tiles[:n_pos]
        selected_neg = neg_tiles[:n_neg]
        balanced_tiles = selected_pos + selected_neg
        rng.shuffle(balanced_tiles)
        
        # --- Train/Val Split ---
        n_val = int(train_val_ratio * len(balanced_tiles))
        val_tiles = balanced_tiles[:n_val]
        train_tiles = balanced_tiles[n_val:]

        # # Shuffle metadata
        # all_tiles = info['ImageRawPath'].unique().tolist()
        # rng = np.random.RandomState(cfg['experiment']['seed'])
        # rng.shuffle(all_tiles)
        
        # # split train-val-test
        # n_total = len(all_tiles)
        # n_trainval = int(train_ratio * n_total)
        # n_test = int(test_ratio * n_total)
        # n_val = int(train_val_ratio * n_trainval)
        # n_train = n_trainval - n_val
        
        # test_tiles = all_tiles[:n_test]
        # trainval_tiles = all_tiles[n_test:n_test + n_trainval]
        # val_tiles = trainval_tiles[:n_val]
        # train_tiles = trainval_tiles[n_val:]
        
        split_tiles = {
            'train': train_tiles,
            'val': val_tiles,
            'test': test_tiles
        }[split]
        
        # build data path
        df = info[info['FileName'].isin(split_tiles)]
        base = os.path.join(cfg['data']['root_dir'], cfg['data']['dataset_dir'])
        self.paths = [os.path.join(base, row['FileName']) for _, row in df.iterrows()]
    

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img, lab, no_data_mask = load_tile(
            path,
            no_data_value=self.cfg['data']['no_data_value'],
            normalize=self.cfg['data']['normalize']
        )
        
        # Augment data
        if self.split == 'train':
            img, lab, no_data_mask = augment_tile(
                img, lab, no_data_mask, self.cfg['data']['augmentation']
            )
        
        # Convert to tensors
        image = torch.from_numpy(img).float()
        label = torch.from_numpy(lab).long()
        no_data = torch.from_numpy(no_data_mask).bool()
        cls_label = torch.tensor((lab > 0).any(), dtype=torch.long)
        return {
            'image': image,
            'label': label,
            'no_data_mask': no_data,
            'cls_label': cls_label
        }
