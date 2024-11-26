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
        from htm.hcam import HCAMLayer
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

        # q network for heuristics decision
        if self.args.heur == 'ama':
            self.n_task = 4
            self.n_heur = 4
            self.q = nn.Sequential(
                nn.Linear(self.n_task, 64), nn.ReLU(),
                nn.Linear(64, self.n_heur),)  
            # epsilon annealing
            self.anneal_time = args.train.anneal_time  # how many time steps to play
            self.final_eps = args.train.final_eps  # final epsilon
            self.init_eps = args.train.init_eps  # starting value of epsilon      
            self.eps = self.init_eps
        
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
        task_heur: indices of time when heur is applied to space
        task_label: index of time when task is given 
        both above has fifo/lifo/slfu/smfu tasks.
        '''
        n_space = self.args.data.n_space
        n_dance = self.args.data.n_dance
        n_action = self.args.data.n_action
        delay_time = self.args.data.delay_time
        dance_time = self.args.data.dance_time
        dance_type = self.args.data.dance_type
        assert delay_time % 16 == 0 and delay_time > 0
        image = batch['image'].float() / 255.  # b,18*16,h,w,c
        motion = batch['motion'].long()  # b,18
        space = batch['space'].long()  # b,18
        task_heur = batch['task_heur'].long()  # b,4,9
        task_label = batch['task_label'].long()  # b,4
        
        '''
        task: fifo / lifo / mvfo / lvfo / all
        '''
        b, _ = motion.size()
        if self.args.task == 'fifo':
            task_idxs = (torch.ones(b) * 0).long()
            label_time = task_label[torch.arange(b), task_idxs]  # b,
            label = motion[torch.arange(b), label_time]  # b,
        elif self.args.task == 'lifo':
            task_idxs = (torch.ones(b) * 1).long()
            label_time = task_label[torch.arange(b), task_idxs]  # b,
            label = motion[torch.arange(b), label_time]  # b,
        elif self.args.task == 'lvfo':
            task_idxs = (torch.ones(b) * 2).long()
            label_time = task_label[torch.arange(b), task_idxs]  # b,
            label = motion[torch.arange(b), label_time]  # b,
        elif self.args.task == 'mvfo':
            task_idxs = (torch.ones(b) * 3).long()
            label_time = task_label[torch.arange(b), task_idxs]  # b,
            label = motion[torch.arange(b), label_time]  # b,
        elif self.args.task == 'all':
            task_idxs = torch.arange(b) % 4
            label_time = task_label[torch.arange(b), task_idxs]  # b,
            label = motion[torch.arange(b), label_time]  # b,

        '''
        make delayed observations and attach query corresponding to the task
        '''
        image_ch = rearrange(image, 'b (nd l) h w c ->  b nd l h w c', nd=n_dance)
        image_delay = repeat(image_ch[:, :, -1], 'b nd h w c -> b nd l h w c', l=delay_time)
        image_memory = torch.cat((image_delay, image_ch), dim=2)
        image_query = image_ch[torch.arange(b), label_time, -1]
        image_query = repeat(image_query, 'b h w c -> b l h w c', l=dance_time)
        
        '''
        from image and space, use heuristics to manage mem with capacity
        '''
        if self.args.heur == 'fifo':
            heur_idxs = (torch.ones(b) * 0).long()
        elif self.args.heur == 'lifo':
            heur_idxs = (torch.ones(b) * 1).long()
        elif self.args.heur == 'slfu':
            heur_idxs = (torch.ones(b) * 2).long()
        elif self.args.heur == 'smfu':
            heur_idxs = (torch.ones(b) * 3).long()
        elif self.args.heur == 'ama':
            with torch.no_grad():
                task_idxs_1hot = F.one_hot(task_idxs.to(self.device), num_classes=self.n_task)
                heur_logit = self.q(task_idxs_1hot.float())
                heur_idxs_greedy = heur_logit.argmax(dim=1)
                heur_idxs_eps = torch.randint(self.n_heur, size=(b,)).to(self.device)
                if mode == 'train':
                    rnd_idx = torch.bernoulli(torch.ones_like(heur_idxs_greedy) * self.eps).bool()
                else:
                    rnd_idx = torch.bernoulli(torch.ones_like(heur_idxs_greedy) * 0.0).bool()
                heur_idxs = torch.where(rnd_idx, heur_idxs_eps, heur_idxs_greedy)
    
        mem_time_idx = task_heur[torch.arange(b), heur_idxs]  # b,9
        image_memory_heur = image_memory[torch.arange(b).view(-1, 1), mem_time_idx]
        mem_space = space[torch.arange(b).view(-1, 1), mem_time_idx]  # b,9
        image_memory_heur = rearrange(image_memory_heur, 'b l ch h w c -> b (l ch) h w c')
        image = torch.cat((image_memory_heur, image_query), dim=1)

        '''
        model forward
        '''
        _, l, _, _, _ = image.size()
        feat = self.encoder(rearrange(image, ('b l h w c -> (b l) c h w')))  
        feat = rearrange(feat, '(b l) e -> b l e', b=b, l=l)  # b,l,d   
        mem_len = (n_dance // 2) * (delay_time + dance_time)  # capacity is half
        memory, query = feat[:, :mem_len], feat[:, mem_len:]
        
        # time embedding
        time_emb = self.time_emb(torch.zeros(1, memory.size(1), 1).to(feat.device)) 
        time_emb = repeat(time_emb, 'm e -> b m e', b=b)  # b,l,e
        memory += time_emb
        
        # core embedding (space)
        if self.args.core_emb == 'none':
            core_emb = torch.zeros_like(feat).to(memory.device)
        elif self.args.core_emb == 'sine1d':
            core_emb = self.core_emb(torch.zeros(1, n_space, 1).to(memory.device))
            core_emb = repeat(core_emb, 'm e -> b m e', b=b)  # b,s,e
            core_emb = core_emb[torch.arange(b).view(-1, 1), mem_space]  # b,c,e
            core_emb = repeat(core_emb, 'b m e -> b m d e', d=dance_time+delay_time)
            core_emb = rearrange(core_emb, 'b m d e -> b (m d) e')
            memory += core_emb

        o, _, _ = self.memory_model((query, memory, None))  # b,l,d
        o_mu = o.mean(dim=1)  # b,d
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
            
        if self.args.heur == 'ama':
            heur_logit = self.q(task_idxs_1hot.float())
            reward = -out['loss_batch'].clone().detach()
            qloss = torch.mean((reward - heur_logit[torch.arange(b), heur_idxs]) ** 2)
            out['qloss'] = qloss
            out['loss'] += 0.1 * out['qloss']
        
        del out['loss_batch']    
        return out

    def forward(self, batch, mode):

        return self.inference(batch, mode)

    def training_step(self, batch, batch_idx):
        
        if self.args.heur == 'ama':
            # epsilon annealing
            if self.eps > self.final_eps:
                self.eps -= (self.init_eps - self.final_eps) / self.anneal_time

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
            if self.args.heur == 'ama':
                wandb.log({'eps': self.eps}, step=self.step)

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
