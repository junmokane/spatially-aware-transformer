import os
import copy
import numpy as np
import torch
import random
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pytorch_lightning as pl
import wandb
from einops import rearrange, repeat, reduce
import time 

from htm.transformer import SinusoidalPosition
from htm.model import CNN


class Hierarchical_Temporal_Memory(pl.LightningModule):

    def __init__(self, args):
        super(Hierarchical_Temporal_Memory, self).__init__()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.save_hyperparameters(args)
        self.args = args
        num_digit = int(np.log10(self.args.train.max_epochs)) + 1
        self.save_name_template = f'model_epoch_{{0:0{num_digit}d}}.ckpt'
        self.step = 0

        # encoder
        self.encoder = CNN(args)
        
        # transformer model
        if args.model_type == 'thcams' or args.model_type == 'cams':
            from htm.hcam import HCAMLayer
            layers = []
            for _ in range(self.args.hcam.num_layers):
                layers.append(HCAMLayer(args))
            self.memory_model = nn.Sequential(*layers)    
        elif args.model_type == 'shcams':
            from htm.space_model import HCAMLayer
            layers = []
            for _ in range(self.args.hcam.num_layers):
                layers.append(HCAMLayer(args))
            self.memory_model = nn.Sequential(*layers)
    
        # mlp for downstream classification
        self.mlp = nn.Linear(args.hcam.dim, args.data.dance_type)
        # action embedding
        self.action_emb = nn.Embedding(self.args.data.n_action, args.hcam.dim)
        # loss
        self.celoss = nn.CrossEntropyLoss(reduction='none')
        
        # time embedding
        if args.time_emb == 'sine':
            self.time_emb = SinusoidalPosition(dim=args.hcam.dim)
        elif args.time_emb == 'const':
            self.time_emb = SinusoidalPosition(dim=args.hcam.dim)
            self.const_time = nn.Parameter(torch.randn(1280, args.hcam.dim))
        
        # core embedding (space)
        if args.core_emb == 'none':
            self.core_emb = None
        elif args.core_emb == 'sine1d':
            self.core_emb = SinusoidalPosition(dim=self.args.hcam.dim)
        
    def breakpoint_here(self):
        breakpoint()
        return

    def inference(self, batch, mode='train'):
        '''
        data preparation
        '''
        # n_space = self.args.data.n_space
        num_rooms = self.args.random_walk.num_rooms
        dance_time = self.args.data.dance_time
        # len_mv = self.args.data.len_mv
        # imagine = self.args.data.imagine
        # n_action = self.args.data.n_action
        delay_time = self.args.data.delay_time
        image = batch['image'].float() / 255.  # b,36*16,h,w,c
        motion = batch['motions'].long()  # b,36
        image = image[:, :num_rooms*dance_time]  # b,25*16,h,w,c
        motion = motion[:, :num_rooms]  # b,25
        b, _ = motion.size()
        tm_idxs = batch['time_action']  # b,1280,2  (place, time)
        pm_idxs = batch['pm_action']  # b,2080,2  (place, time)
        
        label_indices = np.random.choice(num_rooms, b)
        if 'easy' in self.args.task:
            label = motion[torch.arange(b), label_indices]  # b,
        
        '''
        preprocessing 
        '''
        image_ch = rearrange(image, 'b (nd l) h w c ->  b nd l h w c', l=dance_time)
        image_delay = repeat(image_ch[:, :, -1], 'b nd h w c -> b nd l h w c', l=delay_time)
        image_memory = torch.cat((image_delay, image_ch), dim=2)  # b,25,32,h,w,c
        image_memory = rearrange(image_memory, 'b l ch h w c -> b (l ch) h w c')  # b,25*32,h,w,c
        image_query = image_ch[torch.arange(b), label_indices, -1]  # b,h,w,c
        image_query = image_query[:, None]  # b,1,h,w,c
        # image_query = repeat(image_query, 'b h w c -> b l h w c', l=dance_time)  # b,dance_time,h,w,c
        image = torch.cat((image_memory, image_query), dim=1)  # b,num_rooms*dance_time+1,h,w,c
        _, l, _, _, _ = image.size()
        feat = self.encoder(rearrange(image, ('b l h w c -> (b l) c h w')))  
        feat = rearrange(feat, '(b l) e -> b l e', b=b, l=l) 
        memory, query = feat[:, :image_memory.size(1)], feat[:, image_memory.size(1):]  # b,num_rooms*dance_time,e / b,1,e
        memory_ch = rearrange(memory, 'b (nd l) e ->  b nd l e', nd=num_rooms)  # b,num_room,dance_time+delay_time,e
        
        '''
        make mask and time index % chunk_size
        '''
        tm_mask = torch.ones(b, tm_idxs.shape[1]).to(torch.bool).to(memory.device) # b,tm_len
        pm_mask = (pm_idxs[:, :, 1] != -1).to(memory.device) # b, pm_len
        pm_idxs[pm_idxs[:, :, 1] == -1] = 0 # it will be masked
        tm_times = copy.copy(tm_idxs[:, :, 1]) # b,1280
        pm_times = copy.copy(pm_idxs[:, :, 1]) # b,2080
        tm_idxs[:, :, 1] %= dance_time + delay_time
        pm_idxs[:, :, 1] %= dance_time + delay_time
        
        tm_memory = memory_ch[torch.arange(b).view(-1, 1).expand(-1, tm_idxs.shape[1]),
                              tm_idxs[:, :, 0],
                              tm_idxs[:, :, 1]]
        pm_memory = memory_ch[torch.arange(b).view(-1, 1).expand(-1, pm_idxs.shape[1]),
                              pm_idxs[:, :, 0],
                              pm_idxs[:, :, 1]]

        '''
        get time & core embedding
        '''
        # time embedding
        if self.args.time_emb == 'sine':
            time_emb = self.time_emb(torch.zeros(1, tm_times.size(1), 1).to(feat.device))
            time_emb = repeat(time_emb, 'm e -> b m e', b=b)  # b,1280,e
            tm_memory += time_emb[torch.arange(b).view(-1, 1), tm_times]  # b,1280,e
            pm_memory += time_emb[torch.arange(b).view(-1, 1), pm_times]  # b,2080,e
        elif self.args.time_emb == 'const':
            time_emb = self.time_emb(torch.zeros(1, tm_times.size(1), 1).to(feat.device)) + self.const_time
            time_emb = repeat(time_emb, 'm e -> b m e', b=b)  # b,1280,e
            tm_memory += time_emb[torch.arange(b).view(-1, 1), tm_times]  # b,1280,e
            pm_memory += time_emb[torch.arange(b).view(-1, 1), pm_times]  # b,2080,e
        
        # core embedding (space)
        if self.args.core_emb == 'none':
            core_emb_mem = torch.zeros_like(feat).to(memory.device)
        elif self.args.core_emb == 'sine1d':
            core_emb = self.core_emb(torch.zeros(1, num_rooms, 1).to(feat.device))
            core_emb = repeat(core_emb, 'm e -> b m e', b=b)  # b,num_rooms,e
            tm_memory += core_emb[torch.arange(b).view(-1, 1), tm_idxs[:, :, 0]]  # b,1280,e
            pm_memory += core_emb[torch.arange(b).view(-1, 1), pm_idxs[:, :, 0]]  # b,2080,e
        elif self.args.core_emb == 'fourier':
            tm_memory += self.core_emb(tm_idxs[:, :, 0][..., None, None].float()) # b,1280,e
            pm_memory += self.core_emb(pm_idxs[:, :, 0][..., None, None].float()) # b,1280,e
        '''
        model forward 
        tm_memory: [b,1280,e]
        pm_memory: [b,2080,e]
        query: [b,1,e]
        '''
        if self.args.model_type == 'thcams' or self.args.model_type == 'cams':
            o, _, _ = self.memory_model((query, tm_memory, tm_mask))  # b,query_len,e
            o_mu = o.mean(dim=1)
        elif self.args.model_type == 'shcams':    
            o, _, _ = self.memory_model((query, pm_memory, pm_mask))  # b,query_len,e
            o_mu = o.mean(dim=1)
        else:
            raise NotImplementedError('wrong model type')

        out = {}
        
        # downstream classification
        if self.args.task_type == "prediction":
            preds = self.mlp(o_mu)  # b,8
            loss_batch = self.celoss(preds, label) # [b]
            celoss = torch.mean(loss_batch)
            pred_label = preds.argmax(dim=-1).long()
            acc = (pred_label == label).sum() / len(label)
            out.update({
                'loss': celoss,
                'loss_batch': loss_batch,
                'celoss': celoss,
                'acc': acc,
            })        
        
        del out['loss_batch']    
        return out


    def forward(self, batch, mode):

        return self.inference(batch, mode)


    def training_step(self, batch, batch_idx):

        out = self(batch, mode='train')

        out_log = {}
        for k, v in out.items():
            if 'loss' in k or 'acc' in k:
                out_log[k] = out[k].item()
            else:
                out_log[k] = out[k]
        
        self.training_step_outputs.append(out_log)

        phase_log = self.global_step % self.args.log.print_step_freq == 0

        if phase_log and self.global_step > self.args.log.print_step_after:
            self.step += 1
            self.on_train_step_end()

        if self.current_epoch % self.args.log.save_epoch_freq == 0:
            self.trainer.save_checkpoint(
                os.path.join(
                    self.args.ckpt_dir,
                    self.save_name_template.format(self.current_epoch)
                )
            )
        return out


    def on_train_step_end(self):
        
        vals = self.training_step_outputs
        log = {}
        for k, v in vals[0].items():
            if 'loss' in k or 'acc' in k:
                mu = torch.tensor([val[k] for val in vals]).mean().item()
                log[k] = mu
        if 'pred_imgs' in vals[0].keys():
            pred_imgs = vals[0]['pred_imgs'][0] # [dance_time,45,45,3]
            org_imgs = vals[0]['org_imgs'][0] # [dance_time,45,45,3]
            viz_imgs = torch.cat([org_imgs, pred_imgs], dim=1) # [dance_time,90,45,3]
            wandb.log({'train_imgs_from_generation': [wandb.Image(image.permute(2,0,1)) for image in viz_imgs]}, step=self.step)
        self.log_tb(log, 'train')
        self.training_step_outputs.clear()  # free memory


    def validation_step(self, batch, batch_idx):
        
        out = self(batch, mode='eval')

        out_log = {}
        for k, v in out.items():
            if 'loss' in k or 'acc' in k:
                out_log[k] = out[k].item()
            else:
                out_log[k] = out[k]
        
        self.validation_step_outputs.append(out_log)
        return out


    def on_validation_epoch_end(self):
        
        vals = self.validation_step_outputs
        log = {}
        for k, v in vals[0].items():
            if 'loss' in k or 'acc' in k:
                mu = torch.tensor([val[k] for val in vals]).mean().item()
                log[k] = mu
        if 'pred_imgs' in vals[0].keys():
            pred_imgs = vals[0]['pred_imgs'][0] # [dance_time,45,45,3]
            org_imgs = vals[0]['org_imgs'][0] # [dance_time,45,45,3]
            viz_imgs = torch.cat([org_imgs, pred_imgs], dim=1) # [dance_time,90,45,3]
            wandb.log({'eval_imgs_from_generation': [wandb.Image(image.permute(2,0,1)) for image in viz_imgs]}, step=self.step)
        self.log_tb(log, 'eval')
        self.validation_step_outputs.clear()  # free memory


    def configure_optimizers(self):
        
        param = [
            {
                'params': [v for k, v in self.named_parameters()],
                'lr': self.args.train.lr
            }
        ]

        optimizer = torch.optim.Adam(param)
        
        return optimizer


    def log_tb(self, log, type) -> None:
        # log loss
        losses = {'step': self.global_step}
        for k, v in log.items():
            if 'loss' in k or 'acc' in k:
                losses[f'{type}/{k}'] = v
        
        wandb.log(losses, step=self.step)
        return
    

    def on_train_start(self) -> None:
        self.trainer.save_checkpoint(
            os.path.join(
                self.args.ckpt_dir,
                self.save_name_template.format(-1)
            )
        )
        return
