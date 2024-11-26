import glob
import numpy as np
from torch.utils.data import Dataset
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

import time


class Ballet_Dataset(Dataset):
    def __init__(self, args, type_):
        
        if type_ == 'train':
            root = args.train_dir
        elif type_ == 'val':
            root = args.val_dir

        self.file_list = []        
        for root_dir in root:
            self.file_list += glob.glob(f"{root_dir}/*.npz")    
        self.file_list.sort()
        assert len(self.file_list) != 0
        if args.max_samples is not None:
            print(f'max samples is limited to {args.max_samples}')
            self.file_list = self.file_list[:args.max_samples]
        self.num_ep = len(self.file_list)
        print(f'number of files in {root}/{type_}: {self.num_ep}')
        
        # action dataset
        subfolder = 'train' if type_ == 'train' else 'eval'
        self.time_action_file_list = glob.glob(f"{args.action_dir}/{args.random_walk.num_rooms}_rooms/{subfolder}/agent_poses_times_*.npy")
        self.time_action_file_list.sort()
        self.pm_action_file_list = glob.glob(f"{args.action_dir}/{args.random_walk.num_rooms}_rooms/{subfolder}/pm_agent_poses_times_*.npy")
        self.pm_action_file_list.sort()
        print(f'number of files in {args.action_dir}/{type_}: {len(self.time_action_file_list)}')
        
    def __len__(self):
        return self.num_ep

    def __getitem__(self, idx):
        file = self.file_list[idx] 
        data = np.load(file)
        time_action_file = self.time_action_file_list[idx]
        time_action_data = np.load(time_action_file)
        pm_action_file = self.pm_action_file_list[idx]
        pm_action_data = np.load(pm_action_file)
        
        batch = {}
        for key in data:
            batch[key] = torch.tensor(data[key])
        batch['time_action'] = torch.tensor(time_action_data)
        batch['pm_action'] = torch.tensor(pm_action_data)
        return batch


class Ballet_Dataset_Single(Dataset):
    def __init__(self, args, type_):
        
        if type_ == 'train':
            root = args.train_dir
        elif type_ == 'val':
            root = args.val_dir

        self.file_list = []        
        for root_dir in root:
            self.file_list += glob.glob(f"{root_dir}/*.npz")    
        self.file_list.sort()
        assert len(self.file_list) != 0
        if args.max_samples is not None:
            print(f'max samples is limited to {args.max_samples}')
            self.file_list = self.file_list[:args.max_samples]
        self.num_ep = len(self.file_list)
        
    def __len__(self):
        return self.num_ep

    def __getitem__(self, idx):
        file = self.file_list[idx] 
        data = np.load(file)
        batch = {}
        for key in data:
            batch[key] = torch.tensor(data[key])
        return batch
    

class Ballet_DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super(Ballet_DataModule, self).__init__()
        self.args = args
        
    def setup(self, stage = None):
        self.train_dataset = Ballet_Dataset(root=self.args.train_dir)
        self.valid_dataset = Ballet_Dataset(root=self.args.val_dir)
        self.test_dataset = Ballet_Dataset(root=self.args.test_dir)
        
        if 'ddp' in self.args.accelerator:
            self.batch_size = self.args.train.batch_size // len(self.args.train.gpus)
        else:
            self.batch_size = self.args.train.batch_size    
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.args.train.batch_size, shuffle=True,
            num_workers=self.args.train.num_workers, pin_memory=True, drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset, batch_size=self.args.train.batch_size, shuffle=True,
            num_workers=1, pin_memory=True, drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.args.train.batch_size, shuffle=False,
            num_workers=1, pin_memory=True, drop_last=True
        )


