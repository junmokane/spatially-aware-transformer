import os
import time
import torch
import math
import yaml
import shutil
import wandb

import numpy as np
from torch.utils.data import DataLoader
import torch.optim
from torch import nn
import pytorch_lightning as pl
from omegaconf import OmegaConf

# from ballet_dataset.dataset import Ballet_DataModule
from ballet_dataset.dataset import Ballet_Dataset
from htm.rb_shortstay.htm_pl import Hierarchical_Temporal_Memory


def main():

    # argument setting
    exp_config = 'rb_shortstay'
    args_base = OmegaConf.load(f'./htm/config/{exp_config}.yaml')
    args_cli = OmegaConf.from_cli()
    args = OmegaConf.merge(args_base, args_cli)
    assert args.model_type in ['cams', 'thcams', 'shcams']
    assert args.task in ['3_easy', '4_easy', '5_easy']
    assert args.time_emb in ['sine']
    assert args.core_emb in ['none', 'sine1d']

    if '3' in args.task:
        args.data.n_space = 9
        args.opt_jm += '3x3'
        args.random_walk.num_rooms = 9
    elif '4' in args.task:
        args.data.n_space = 16
        args.opt_jm += '4x4'
        args.random_walk.num_rooms = 16
    elif '5' in args.task:
        args.data.n_space = 25
        args.opt_jm += '5x5'
        args.random_walk.num_rooms = 25

    if args.model_type == 'cam':
        args.hcam.mem_chunk_size = 1
        args.hcam.topk_mems = args.data.imagine * (args.data.dance_time + args.data.delay_time)
        args.time_emb = 'sine'
        args.core_emb = 'none'
    elif args.model_type == 'cama':
        args.hcam.mem_chunk_size = 1
        args.hcam.topk_mems = args.data.imagine * (args.data.dance_time + args.data.delay_time + 1)
        args.time_emb = 'sine'
        args.core_emb = 'none'
    elif args.model_type == 'cams':
        args.hcam.mem_chunk_size = 1
        args.hcam.topk_mems = args.data.imagine * (args.data.dance_time + args.data.delay_time)
        args.time_emb = 'sine'
        args.core_emb = 'sine1d'
    elif args.model_type == 'thcams' or args.model_type == 'shcams':
        args.hcam.mem_chunk_size = args.data.dance_time + args.data.delay_time
        args.hcam.topk_mems = 4
        args.time_emb = 'sine'
        args.core_emb = 'sine1d'
    elif args.model_type == 'tr' or args.model_type == 'tra':
        args.core_emb = 'none'

    print(f'argument list: {args}')

    # env setting
    pl.trainer.seed_everything(args.train.seed)
    if 'tr' in args.model_type:
        name_indication = {
            'task': args.task,
            'model': args.model_type,
            'temb': args.time_emb,
            'semb': args.core_emb,
            'seed': args.train.seed,
            }
    elif 'cam' in args.model_type:
        name_indication = {
            'task': args.task,
            'model': args.model_type,
            'temb': args.time_emb,
            'semb': args.core_emb,
            'topk': args.hcam.topk_mems,
            'seed': args.train.seed,
            }
    name = ''
    for k, v in name_indication.items():
        name += f'{k}:({v})__'

    pr_name = f"{exp_config}_{args.opt_jm}"

    if args.mode == 'train':
        # wandb setting
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

        train_dataset = Ballet_Dataset(args, 'train')
        valid_dataset = Ballet_Dataset(args, 'val')
        train_dataloader = DataLoader(train_dataset, batch_size=args.train.batch_size, shuffle=True,
                                    num_workers=args.train.num_workers, pin_memory=True, drop_last=True)
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

    

if __name__ == '__main__':
    main()
