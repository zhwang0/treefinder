# File: data_loader/by_state_split.py
import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from data_loader.utils import load_tile, augment_tile
from sklearn.model_selection import train_test_split


class TreeSplitDataset(Dataset):
    def __init__(self, cfg:dict, split: str='train'):
        self.cfg = cfg
        self.split = split
        aux_filename = cfg['data']['dataset_info'].split('.')[0] + '_aux.csv'
        info = pd.read_csv(os.path.join(cfg['data']['root_dir'], aux_filename))
        info = info[info['TreeTypes'].notnull()]
        all_tree = info['TreeTypes'].unique().tolist()
        
        # Filter by state
        bs = cfg['data']['split']['by_tree_type']
        test_exc = bs.get('test_exclude_train', False)
        train_exc = bs.get('train_exclude_test', False)
        

        # Fallback: split by filenames (not climates) if val_climate is missing or empty
        if (split in ['train', 'val']) and (not bs.get('val_tree_types')):
            trees = bs['train_tree_types']
            
            # Check if there are any training trees in the dataset
            if len(trees) == 0: 
                if train_exc:
                    # train on ALL except test
                    trees = [c for c in all_tree if c not in bs['test_tree_types']]
                else:
                    raise ValueError("No training trees found. Please check your configuration.")
                
            df_train = info[info['TreeTypes'].isin(trees)].reset_index(drop=True) 
            df_train_split, df_val_split = train_test_split(df_train, test_size=0.1, random_state=cfg['experiment']['seed'])

            if split == 'train':
                df = df_train_split
            else:  # split == 'val'
                df = df_val_split
            
        else: 
            if split == 'test' and test_exc:
                # test on ALL except train
                trees = [c for c in all_tree if c not in bs['train_tree_types']]
            else:
                key = {'train': 'train_tree_types', 'val': 'val_tree_types', 'test': 'test_tree_types'}[split]
                trees = bs[key]
            df = info[info['TreeTypes'].isin(trees)].reset_index(drop=True)     

        base = os.path.join(cfg['data']['root_dir'], cfg['data']['dataset_dir'])
        self.paths = [os.path.join(base, row['FileName']) for _, row in df.iterrows()]
        # self.tree_types = df['TreeTypes'].tolist()
    
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

