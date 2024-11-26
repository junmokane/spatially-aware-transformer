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
from ballet_dataset.dataset import Ballet_Dataset_Single
from htm.rb_multi.htm_pl import Hierarchical_Temporal_Memory


def main():

    # argument setting
    exp_config = 'rb_multi'
    args_base = OmegaConf.load(f'./htm/config/{exp_config}.yaml')
    args_cli = OmegaConf.from_cli()
    args = OmegaConf.merge(args_base, args_cli)
    assert args.heur in ['fifo', 'lifo', 'mvfo', 'lvfo', 'ama']
    assert args.task in ['fifo', 'lifo', 'mvfo', 'lvfo', 'all']
    assert args.time_emb in ['sine']
    assert args.core_emb in ['none', 'sine1d']

    args.hcam.mem_chunk_size = 1
    args.hcam.topk_mems = (args.data.n_dance // 2) * (args.data.dance_time + args.data.delay_time)
    args.time_emb = 'sine'
    args.core_emb = 'sine1d'

    print(f'argument list: {args}')

    # env setting
    pl.trainer.seed_everything(args.train.seed)
    name_indication = {
        'task': args.task,
        'heur': args.heur,
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

        train_dataset = Ballet_Dataset_Single(args, 'train')
        valid_dataset = Ballet_Dataset_Single(args, 'val')
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
