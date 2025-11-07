# File: data_loader/by_state_split.py
import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from data_loader.utils import load_tile, augment_tile
from sklearn.model_selection import train_test_split


class ClimateSplitDataset(Dataset):
    def __init__(self, cfg:dict, split: str='train'):
        self.cfg = cfg
        self.split = split
        aux_filename = cfg['data']['dataset_info'].split('.')[0] + '_aux.csv'
        info = pd.read_csv(os.path.join(cfg['data']['root_dir'], aux_filename))
        all_climate = info['ClimateType'].unique().tolist()
        
        # Filter by state
        bs = cfg['data']['split']['by_climate']
        test_exc = bs.get('test_exclude_train', False)
        train_exc = bs.get('train_exclude_test', False)
        

        # Fallback: split by filenames (not climates) if val_climate is missing or empty
        if (split in ['train', 'val']) and (not bs.get('val_climate')):
            climates = bs['train_climate']
            
            # Check if there are any training climates in the dataset
            if len(climates) == 0:
                if train_exc:
                    # train on ALL except test
                    climates = [c for c in all_climate if c not in bs['test_climate']]
                else:
                    raise ValueError("No training climates found. Please check your configuration.")
                
            df_train = info[info['ClimateType'].isin(climates)].reset_index(drop=True)
            df_train_split, df_val_split = train_test_split(df_train, test_size=0.1, random_state=cfg['experiment']['seed'])

            if split == 'train':
                df = df_train_split
            else:  # split == 'val'
                df = df_val_split
            
        else: 
            if split == 'test' and test_exc:
                # test on ALL except train
                climates = [c for c in all_climate if c not in bs['train_climate']]
            else:
                key = {'train': 'train_climate', 'val': 'val_climate', 'test': 'test_climate'}[split]
                climates = bs[key]
            df = info[info['ClimateType'].isin(climates)].reset_index(drop=True)     

        base = os.path.join(cfg['data']['root_dir'], cfg['data']['dataset_dir'])
        self.paths = [os.path.join(base, row['FileName']) for _, row in df.iterrows()]
        self.df = df        
    
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

