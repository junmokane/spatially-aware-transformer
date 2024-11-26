import os
import numpy as np
import torch
import random
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.cluster import KMeans
import pytorch_lightning as pl
import wandb
from einops import rearrange, repeat, reduce
import time 


from htm.utils import random_walk_wo_wall_ffhq
from htm.transformer import SinusoidalPosition
from htm.model import CNNSmall, DeCNNSmall


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
        self.encoder = CNNSmall(args)
        # decoder 
        self.decoder = DeCNNSmall(args)
        
        # transformer model
        if 'tr' in args.model_type:
            from htm.transformer import TrLayer
            layers = []
            for _ in range(self.args.tr.num_layers):
                layers.append(TrLayer(args))
            self.memory_model = nn.Sequential(*layers)
            # action embedding
            self.action_emb = nn.Embedding(self.args.data.n_action, args.tr.dim)
            # time embedding
            self.time_emb = SinusoidalPosition(dim=args.tr.dim)
        elif 'hcam' in args.model_type or 'cam' in args.model_type:
            from htm.hcam import HCAMLayer
            layers = []
            for _ in range(self.args.hcam.num_layers):
                layers.append(HCAMLayer(args))
            self.memory_model = nn.Sequential(*layers)
            # action embedding
            self.action_emb = nn.Embedding(self.args.data.n_action, args.hcam.dim)
            # time embedding
            self.time_emb = SinusoidalPosition(dim=args.hcam.dim)
        elif args.model_type == 'space_model':
            from htm.space_model import HCAMLayer
            layers = []
            for _ in range(self.args.hcam.num_layers):
                layers.append(HCAMLayer(args))
            self.memory_model = nn.Sequential(*layers)
            # action embedding
            self.action_emb = nn.Embedding(self.args.data.n_action, args.hcam.dim)
            # time embedding
            self.time_emb = SinusoidalPosition(dim=args.hcam.dim)
    
        # loss
        self.mseloss = nn.MSELoss(reduction='none')
        
        # core embedding
        if 'none' in self.args.core_emb:
            self.core_emb = None
        elif 'space' in self.args.core_emb:
            if 'learn1d' in self.args.core_emb:
                self.core_emb = nn.Embedding(self.args.data.len_mv ** 2, self.args.tr.dim)
            elif 'learn2d' in self.args.core_emb:
                self.core_emb = nn.Embedding(self.args.data.len_mv * 2, self.args.tr.dim)
            elif 'sine1d' in self.args.core_emb:
                self.core_emb = SinusoidalPosition(dim=self.args.tr.dim)
            elif 'sine2d' in self.args.core_emb:
                self.core_emb = SinusoidalPosition(dim=self.args.tr.dim)
            else:
                raise NotImplementedError('wrong core embedding')
        else:
            raise NotImplementedError('wrong core embedding')
        
        self.unfold = nn.Unfold(kernel_size=self.args.data.width,
                                stride=self.args.data.mv,
                                padding=0)

    def breakpoint_here(self):
        breakpoint()
        return

    def inference(self, batch, mode='train'):
        '''
        data preparation
        '''
        len_mv = self.args.data.len_mv
        width = self.args.data.width
        mv = self.args.data.mv
        memorize = self.args.data.memorize
        imagine = self.args.data.imagine
        n_action = self.args.data.n_action
        chunk_size = self.args.hcam.mem_chunk_size
        n_space = len_mv * len_mv
        len_mv_q = len_mv - 1 
        image, _ = batch[0]
        position = batch[1]['position'].long()  # b,imagine+memorize+1,2 
        action = batch[1]['action'].long()  # b,imagine+memorize
        cluster_rand = self.args.cluster_rand
        knn_idx =  batch[1]['knn_idx'][:, cluster_rand].long()  # b,9,memorize
        sort_idx_new = batch[1]['sort_idx_new'][:, cluster_rand].long()  # b,9,memorize*2
        b, c, h, _ = image.size()
        # randomly shuffle position and action so that trajectory randomly changes for each image
        if mode == 'train':
            rand_batch = torch.randperm(b)
            position = position[rand_batch]
            action = action[rand_batch]
            knn_idx = knn_idx[rand_batch]
            sort_idx_new = sort_idx_new[rand_batch]
        assert action.size(1) == imagine + memorize and position.size(1) == imagine + memorize + 1 and knn_idx.size(1) == memorize
        
        # make patches and action embedding and space
        patches = rearrange(self.unfold(image), 'b (c h w) yx -> b yx c h w', c=c, h=width)  # b,y*x,c,p,p
        action_emb = self.action_emb(action)  # b,memorize+imagine,e
        space = position[..., 0] * len_mv + position[..., 1]  # b,memorize+imagine+1
        
        '''
        image sequence and gt image
        '''
        patches = patches[torch.arange(b).view(-1, 1), space]  # b,memorize+imagine+1,c,p,p
        gt_imgs = patches[:, memorize:]  # b,imagine+1,c,p,p

        '''
        image encoding
        '''
        _, l, _, _, _ = patches.size()
        feat = self.encoder(rearrange(patches, ('b l c h w -> (b l) c h w')))
        feat = rearrange(feat, '(b l) e -> b l e', b=b, l=l)  # b,memorize+imagine+1,e
        memory, query = feat[:, :memorize], feat[:, memorize:memorize+1]  # b,memorize,e / b,1,e
        query = torch.cat((query, action_emb[:, memorize:]), dim=1) # b,1+imagine,e

        '''
        get core embedding
        '''
        if 'space' in self.args.core_emb:    
            if 'learn1d' in self.args.core_emb:
                core_emb_mem = self.core_emb(space)  # b,memorize+imagine+1,e
            elif 'learn2d' in self.args.core_emb:
                core_emb_mem_y = self.core_emb(position[..., 0] + len_mv)  # b,memorize+imagine+1,e
                core_emb_mem_x = self.core_emb(position[..., 1])  # b,memorize+imagine+1,e
                core_emb_mem = (core_emb_mem_y + core_emb_mem_x) / 2.  # b,memorize+imagine+1,e
            elif 'sine1d' in self.args.core_emb:
                core_emb = self.core_emb(torch.zeros(1, n_space, 1).to(image.device))
                core_emb = repeat(core_emb, 'm e -> b m e', b=b)  # b,space,e
                core_emb_mem = core_emb[torch.arange(b).view(-1, 1), space]  # b,memorize+imagine+1,e
            elif 'sine2d' in self.args.core_emb:
                core_emb = self.core_emb(torch.zeros(1, len_mv * 2, 1).to(image.device))
                core_emb = repeat(core_emb, 'm e -> b m e', b=b)  # b,space,e
                core_emb_mem_y = core_emb[torch.arange(b).view(-1, 1), position[..., 0] + len_mv]  # b,memorize+imagine+1,e
                core_emb_mem_x = core_emb[torch.arange(b).view(-1, 1), position[..., 1]]  # b,memorize+imagine+1,e
                core_emb_mem = (core_emb_mem_y + core_emb_mem_x) / 2.
            core_emb_mem = core_emb_mem[:, :memorize]  # b,memorize,e
        else:
            core_emb_mem = 0.
        
        '''
        model forward 
        memory: [b,memorize,e]      img(t0)-img(t1)-img(t2)-...
        action_emb: [b,memorize+imagine,e]  a0-a1-a2-...
        core_emb_mem: [b,memorize,e]  s0-s1-s2-...
        query: [b,1+imagine,e]  img(t)-a(t)-a(t+1)-...
        add core_emb_mem only on img part of the memory
        '''
        if 'tr' in self.args.model_type: 
            # add core embedding
            memory = memory + core_emb_mem
            if self.args.model_type == 'tra':
                # concat action embedding
                memory = torch.cat((memory, action_emb[:, :memorize]), dim=-1)
                memory = rearrange(memory, 'b l (n d) -> b (l n) d', n=2)  # b,memorize*2,e
            # add time embedding
            mem_que = torch.cat((memory, query), dim=1)
            time_emb = self.time_emb(torch.zeros(1, mem_que.size(1), 1).to(image.device))
            time_emb = repeat(time_emb, 'm e -> b m e', b=b)
            mem_que = mem_que + time_emb
            o = self.memory_model(mem_que)[:, memory.size(1):]  # b,1+imagine,e
        elif 'hcam' in self.args.model_type or 'cam' in self.args.model_type:
            # add core embedding
            memory = memory + core_emb_mem
            if self.args.model_type == 'cama':
                # concat action embedding
                memory = torch.cat((memory, action_emb[:, :memorize]), dim=-1)
                memory = rearrange(memory, 'b l (n d) -> b (l n) d', n=2)  # b,memorize*2,e
            # add time embedding
            mem_que = torch.cat((memory, query), dim=1)
            time_emb = self.time_emb(torch.zeros(1, mem_que.size(1), 1).to(image.device))
            time_emb = repeat(time_emb, 'm e -> b m e', b=b)
            mem_que = mem_que + time_emb
            memory, query = mem_que[:, :memory.size(1)], mem_que[:, memory.size(1):]
            o, _, _ = self.memory_model((query, memory, None))  # b,1+imagine,e
        elif self.args.model_type == 'space_model':    
            # add core embedding
            memory = memory + core_emb_mem
            # add time embedding
            mem_que = torch.cat((memory, query), dim=1)
            time_emb = self.time_emb(torch.zeros(1, mem_que.size(1), 1).to(image.device))
            time_emb = repeat(time_emb, 'm e -> b m e', b=b)
            mem_que = mem_que + time_emb
            memory, query = mem_que[:, :memory.size(1)], mem_que[:, memory.size(1):]
            
            ''' # space sorting with knn
            sort_idx = torch.argsort(knn_idx, dim=1)
            position_sort = position[torch.arange(b).view(-1, 1), sort_idx]
            knn_idx_sort = knn_idx[torch.arange(b).view(-1, 1), sort_idx]
    
            print(rearrange(knn_idx_sort[0], '(y x) -> y x', x=chunk_size))
            knn_idx_sort_new = knn_idx[torch.arange(b).view(-1, 1), sort_idx_new]
            print(rearrange(knn_idx_sort_new[0], '(y x) -> y x', x=chunk_size))
            print(rearrange(sort_idx[0], '(y x) -> y x', x=chunk_size))
            print(rearrange(sort_idx_new[0], '(y x) -> y x', x=chunk_size))
            
            sort_idx_new = pad_sequence(sort_idx_new, batch_first=True, padding_value=-1)
            print(sort_idx_new.size())
            print(rearrange(sort_idx_new[0], '(b l) -> b l', l=chunk_size))
            '''
            
            # process knn_idx to sort_idx_new for chunking. this is already available
            sort_idx_new_reshape = rearrange(sort_idx_new, 'b (y x) -> b y x', x=chunk_size)
            tmp = sort_idx_new_reshape[:, :, 0]  # From -1, it is redundant
            max_pad_idx = (memorize * 2 // chunk_size - (tmp == -1).sum(dim=1).min()) * chunk_size            
            sort_idx_new = sort_idx_new[:, :max_pad_idx]
            
            memory = memory[torch.arange(b).view(-1, 1), sort_idx_new]
            core_emb_mem = core_emb_mem[torch.arange(b).view(-1, 1), sort_idx_new]
            mask = torch.where(sort_idx_new == -1, False, True)
            o, _, _ = self.memory_model((query, memory, mask))  # b,1+imagine,e
        else:
            raise NotImplementedError('wrong model type')

        out = {}
        # downstream generation
        if self.args.task_type == "generation":
            pred_imgs = self.decoder(rearrange(o, 'b l e -> (b l) e'))
            pred_imgs = rearrange(pred_imgs, '(b l) h w c ->  b l c h w', b=b)  # b,imagine+1,c,h,w
            loss_batch = torch.mean(self.mseloss(pred_imgs[:, 1:], gt_imgs[:, 1:]), dim=[1,2,3,4]) # [b]
            mseloss = torch.mean(loss_batch)
            out.update({
                'loss': mseloss,
                # 'loss_batch': loss_batch,
                'pred_imgs': pred_imgs[:, 1:],
                'gt_imgs': gt_imgs[:, 1:],
            })
        else:
            raise NotImplementedError(f"task type {self.args.task_type} not implemented!")
        # s4 = time.time()
        # print(s4 - s3)
        # del out['loss_batch']    
        return out


    def forward(self, batch, mode):

        return self.inference(batch, mode)


    def training_step(self, batch, batch_idx):

        out = self(batch, mode='train')

        out_log = {}
        for k, v in out.items():
            if 'loss' in k or 'acc' in k:
                out_log[k] = out[k].item()
        
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
        # if 'pred_imgs' in vals[0].keys():
        #     pred_imgs = vals[0]['pred_imgs'][0] # l,c,h,w
        #     gt_imgs = vals[0]['gt_imgs'][0]    # l,c,h,w
        #     viz_imgs = torch.cat([gt_imgs, pred_imgs], dim=2) # l,c,2*h,w
        #     wandb.log({'train_imgs_from_generation': [wandb.Image(image) for image in viz_imgs]}, step=self.step)
        self.log_tb(log, 'train')
        self.training_step_outputs.clear()  # free memory


    def validation_step(self, batch, batch_idx):
        
        out = self(batch, mode='eval')

        out_log = {}
        for k, v in out.items():
            if 'loss' in k or 'acc' in k:
                out_log[k] = out[k].item()
            else:
                if batch_idx == 0:
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
            pred_imgs = vals[0]['pred_imgs'][0] # l,c,h,w
            gt_imgs = vals[0]['gt_imgs'][0]    # l,c,h,w
            viz_imgs = torch.cat([gt_imgs, pred_imgs], dim=2) # l,c,2*h,w
            wandb.log({'eval_imgs_from_generation': [wandb.Image(image) for image in viz_imgs]}, step=self.step)
        self.log_tb(log, 'eval')
        self.validation_step_outputs.clear()  # free memory


    def test_step(self, batch, batch_idx):
        checkpoint_dir = '/'.join(self.checkpoint_path.split('/')[:-1])
        save_dir = f'{checkpoint_dir}/result'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
        out = self(batch, mode='eval')
        print(out['loss'])
        print(out['loss_batch'])
        pred_imgs = (out['pred_imgs'].cpu().numpy()* 255).astype(np.uint8) # b,l,c,h,w
        gt_imgs = (out['gt_imgs'].cpu().numpy()* 255).astype(np.uint8)     # b,l,c,h,w
        viz_imgs = np.concatenate([gt_imgs, pred_imgs], axis=3) # b,l,c,2*h,w
        b = viz_imgs.shape[0]
        
        for j in range(b):
            viz_img = rearrange(viz_imgs[j], 'l c h w -> l h w c')  # l 2*h w c
            viz_img_re = rearrange(viz_img, 'l h w c -> h (l w) c')
            plt.imsave(f'{save_dir}/full_gen_{j}.png', viz_img_re)
            
            if j == 8:
                le, h, _, _ = viz_img.shape
                for i in range(le):
                    plt.imsave(f'{save_dir}/gt_vis_{j}_{i}.png', viz_img[i, 0:h//2])
                    plt.imsave(f'{save_dir}/pr_vis_{j}_{i}.png', viz_img[i, h//2:])
            
        exit()
        return out


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
        # self.trainer.save_checkpoint(
        #     os.path.join(
        #         self.args.ckpt_dir,
        #         self.save_name_template.format(-1)
        #     )
        # )
        return
