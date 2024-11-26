import os
import time
import torch
import math
import yaml
import shutil
import wandb

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets as dset
import torch.optim
from torch import nn
import pytorch_lightning as pl
from omegaconf import OmegaConf

from htm.ffhq.htm_pl import Hierarchical_Temporal_Memory

class Pos_Dataset(Dataset):
    def __init__(self, args, type_):
        
        if type_ == 'train':
            root = args.train_pos_dir
        elif type_ == 'val':
            root = args.val_pos_dir

        data = np.load(root)
        self.positions = torch.tensor(data['positions'])
        self.actions = torch.tensor(data['actions'])
        self.knn_idxs = torch.tensor(data['knn_idxs'])
        self.sort_idxs_new = torch.tensor(data['sort_idxs_new'])
        self.num_data = len(self.positions)
        
    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        batch = {'position': self.positions[idx],
                 'action': self.actions[idx],
                 'knn_idx': self.knn_idxs[idx],
                 'sort_idx_new': self.sort_idxs_new[idx],
                 }
        return batch


class ConcatDataset(torch.utils.data.Dataset):
            def __init__(self, *datasets):
                self.datasets = datasets

            def __getitem__(self, i):
                return tuple(d[i % len(d)] for d in self.datasets)

            def __len__(self):
                return max(len(d) for d in self.datasets)
            

def main():

    # argument setting
    exp_config = 'ffhq'
    args_base = OmegaConf.load(f'./htm/config/{exp_config}.yaml')
    args_cli = OmegaConf.from_cli()
    args = OmegaConf.merge(args_base, args_cli)
    assert args.model_type in ['tr', 'tra', 'cam', 'cama', 'cams', 'hcams', 'space_model']
    assert args.time_emb in ['sine']
    assert args.core_emb in ['none', 'learn1d', 'learn2d', 'sine1d', 'sine2d']

    if args.model_type == 'cam':
        args.hcam.mem_chunk_size = 1
        args.hcam.topk_mems = args.data.memorize
        args.core_emb = 'none'
    elif args.model_type == 'cama':
        args.hcam.mem_chunk_size = 1
        args.hcam.topk_mems = args.data.memorize * 2
        args.core_emb = 'none'
    elif args.model_type == 'cams':
        args.hcam.mem_chunk_size = 1
        args.hcam.topk_mems = args.data.memorize
        args.core_emb = 'sine2d_space'
    elif args.model_type == 'hcams' or args.model_type == 'space_model':
        args.hcam.mem_chunk_size = 8
        args.hcam.topk_mems = 4
        args.core_emb = 'sine2d_space'
    elif args.model_type == 'tr' or args.model_type == 'tra':
        args.core_emb = 'none'

    print(f'argument list: {args}')
    opt_jm = f'gr_{args.data.len_mv}_wi_{args.data.width}_mv_{args.data.mv}_im_{args.data.imagine}'

    # env setting
    pl.trainer.seed_everything(args.train.seed)
    if 'tr' in args.model_type:
        name_indication = {
            'model': args.model_type,
            'temb': args.time_emb,
            'semb': args.core_emb,
            'seed': args.train.seed,
            }
    elif 'cam' in args.model_type:
        name_indication = {
            'model': args.model_type,
            'temb': args.time_emb,
            'semb': args.core_emb,
            'topk': args.hcam.topk_mems,
            'seed': args.train.seed,
            }
    elif args.model_type == 'space_model':
        name_indication = {
            'model': args.model_type,
            'temb': args.time_emb,
            'semb': args.core_emb,
            'topk': args.hcam.topk_mems,
            'cls': args.cluster_rand,
            'seed': args.train.seed,
            }
    name = ''
    for k, v in name_indication.items():
        name += f'{k}:({v})__'
    
    pr_name = f"{exp_config}_{opt_jm}"
    if args.mode == 'train':
        wandb.login()
        wandb.init(
            project=pr_name,
            entity="",
            config=args,
            mode=args.wandb_mode,
            name=name,
        )

        # make result folder    
        args.result_dir = os.path.join(args.result_dir, pr_name, name)
        args.ckpt_dir = os.path.join(args.result_dir, 'checkpoints')

        data_transform = T.Compose([T.ToTensor(), 
                                    T.Resize(size=args.data.width + (args.data.len_mv-1) * args.data.mv),])
        train_image_dataset = dset.ImageFolder(root=args.train_dir, transform=data_transform)
        valid_image_dataset = dset.ImageFolder(root=args.val_dir, transform=data_transform)
        train_pos_dataset = Pos_Dataset(args, 'train')
        valid_pos_dataset = Pos_Dataset(args, 'val')
        
        train_dataset = ConcatDataset(train_image_dataset, train_pos_dataset)
        valid_dataset = ConcatDataset(valid_image_dataset, valid_pos_dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=args.train.batch_size, shuffle=True,
                                    num_workers=args.train.num_workers, pin_memory=True, drop_last=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.train.batch_size, shuffle=True,
                                    num_workers=args.train.num_workers, pin_memory=True, drop_last=True)

    elif args.mode == 'test':
        data_transform = T.Compose([T.ToTensor(), 
                                    T.Resize(size=args.data.width + (args.data.len_mv-1) * args.data.mv),])
        valid_image_dataset = dset.ImageFolder(root=args.test_dir, transform=data_transform)
        valid_pos_dataset = Pos_Dataset(args, 'val')
        valid_dataset = ConcatDataset(valid_image_dataset, valid_pos_dataset)
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.train.batch_size, shuffle=False,
                                    num_workers=args.train.num_workers, pin_memory=True, drop_last=True)

    # model
    model = Hierarchical_Temporal_Memory(args)

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        devices=args.train.gpus,
        accelerator=args.accelerator,
        val_check_interval=args.train.val_check_interval,
        max_epochs=args.train.max_epochs,
        gradient_clip_val=args.train.gradient_clip_val,
        limit_val_batches=args.train.limit_val_batches,
    )

    if args.mode == 'train':
        trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=valid_dataloader,
        )
    elif args.mode == 'test':
        ckpt_path = f'./htm/result/{name}/checkpoints/model_epoch_075.ckpt'
        print(f'loading the following model: {ckpt_path}')
        model = model.load_from_checkpoint(
            checkpoint_path=ckpt_path,
        )
        model.checkpoint_path = ckpt_path

        trainer.test(
            model=model,
            dataloaders=valid_dataloader,
        )
        

if __name__ == '__main__':
    main()
