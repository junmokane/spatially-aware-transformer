import numpy as np
import torch
from torch import nn
from einops import rearrange, repeat, reduce


# Learnable Fourier Features for Multi-Dimensional Spatial Positional Encoding
class LearnableFourierFeatures(nn.Module):
    def __init__(self, pos_dim, f_dim, h_dim, d_dim, g_dim=1, gamma=1.0):
        super(LearnableFourierFeatures, self).__init__()
        assert f_dim % 2 == 0, 'number of fourier feature dimensions must be divisible by 2.'
        assert d_dim % g_dim == 0, 'number of D dimension must be divisible by the number of G dimension.'
        enc_f_dim = int(f_dim / 2)
        dg_dim = int(d_dim / g_dim)
        self.Wr = nn.Parameter(torch.randn([enc_f_dim, pos_dim]) * (gamma ** 2))
        self.mlp = nn.Sequential(
            nn.Linear(f_dim, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, dg_dim)
        )
        self.div_term = np.sqrt(f_dim)

    def forward(self, pos):
        # input pos dim: (B L G M)
        # output dim: (B L D)
        # L stands for sequence length. all dimensions must be flattened to a single dimension.
        XWr = torch.matmul(pos, self.Wr.T)
        F = torch.cat([torch.cos(XWr), torch.sin(XWr)], dim=-1) / self.div_term
        Y = self.mlp(F)
        pos_enc = rearrange(Y, 'b l g d -> b l (g d)')

        return pos_enc

def random_walk_wo_wall_ffhq(batch_size,
                             len_mv,
                             length,
                             n_action,
                             init_poss=None):
    
    # required values
    b = batch_size
    w = len_mv
    l = length
    a = n_action

    # construct position and action
    position = np.zeros((b, l+1, 2), np.int32)
    action_selection = np.zeros((b, l), np.int32)
    
    for idx in range(b):
        
        new_continue_action_flag = True
        for t in range(l+1):
            
            if t == 0:
                if init_poss is not None:
                    position[idx, t] = init_poss[idx]
                else:
                    position[idx, t] = np.random.randint(0, w, size=(2))
            else:
                if new_continue_action_flag:
                    new_continue_action_flag = False
                    need_to_stop = False

                    while 1:
                        action_random_selection = np.random.randint(0, 4, size=(1))
                        if not (action_random_selection == 0 and position[idx, t - 1, 1] == w-1):
                            if not (action_random_selection == 1 and position[idx, t - 1, 1] == 0):
                                if not (action_random_selection == 2 and position[idx, t - 1, 0] == 0):
                                    if not (action_random_selection == 3 and position[idx, t - 1, 0] == w-1):
                                        break
                    
                    # action_duriation = np.random.poisson(2, 1)
                    action_duriation = 1

                if action_duriation > 0:
                    if not need_to_stop:
                        if action_random_selection == 0:
                            if position[idx, t - 1, 1] == w-1:
                                need_to_stop = True
                                position[idx, t] = position[idx, t - 1]
                                action_selection[idx, t - 1] = 4
                            else:
                                position[idx, t] = position[idx, t - 1] + np.array([0, 1])
                                action_selection[idx, t - 1] = action_random_selection
                        elif action_random_selection == 1:
                            if position[idx, t - 1, 1] == 0:
                                need_to_stop = True
                                position[idx, t] = position[idx, t - 1]
                                action_selection[idx, t - 1] = 4
                            else:
                                position[idx, t] = position[idx, t - 1] + np.array([0, -1])
                                action_selection[idx, t - 1] = action_random_selection
                        elif action_random_selection == 2:
                            if position[idx, t - 1, 0] == 0:
                                need_to_stop = True
                                position[idx, t] = position[idx, t - 1]
                                action_selection[idx, t - 1] = 4
                            else:
                                position[idx, t] = position[idx, t - 1] + np.array([-1, 0])
                                action_selection[idx, t - 1] = action_random_selection
                        elif action_random_selection == 3:
                            if position[idx, t - 1, 0] == w-1:
                                need_to_stop = True
                                position[idx, t] = position[idx, t - 1]
                                action_selection[idx, t - 1] = 4
                            else:
                                position[idx, t] = position[idx, t - 1] + np.array([1, 0])
                                action_selection[idx, t - 1] = action_random_selection
                    else:
                        position[idx, t] = position[idx, t - 1]
                        action_selection[idx, t - 1] = 4
                    action_duriation -= 1
                else:
                    action_selection[idx, t - 1] = 4
                    position[idx, t] = position[idx, t - 1]
                if action_duriation <= 0:
                    new_continue_action_flag = True

    action_selection = torch.from_numpy(action_selection)
    position = torch.from_numpy(position)
    return position, action_selection


def random_walk_wo_wall(batch_size, len_mv, imagine):
    
    # required values
    b = batch_size
    w = len_mv
    l = imagine

    # construct position and action
    position = np.zeros((b, l+1, 2), np.int32)
    action_selection = np.zeros((b, l), np.int32)
    
    for idx in range(b):
        
        new_continue_action_flag = True
        for t in range(l+1):
            
            if t == 0:
                position[idx, t] = np.random.randint(0, w, size=(2))
            else:
                if new_continue_action_flag:
                    new_continue_action_flag = False
                    need_to_stop = False

                    while 1:
                        action_random_selection = np.random.randint(0, 4, size=(1))
                        if not (action_random_selection == 0 and position[idx, t - 1, 1] == w-1):
                            if not (action_random_selection == 1 and position[idx, t - 1, 1] == 0):
                                if not (action_random_selection == 2 and position[idx, t - 1, 0] == 0):
                                    if not (action_random_selection == 3 and position[idx, t - 1, 0] == w-1):
                                        break
                    
                    # action_duriation = np.random.poisson(2, 1)
                    action_duriation = 1

                if action_duriation > 0:
                    if not need_to_stop:
                        if action_random_selection == 0:
                            if position[idx, t - 1, 1] == w-1:
                                need_to_stop = True
                                position[idx, t] = position[idx, t - 1]
                                action_selection[idx, t - 1] = 4
                            else:
                                position[idx, t] = position[idx, t - 1] + np.array([0, 1])
                                action_selection[idx, t - 1] = action_random_selection
                        elif action_random_selection == 1:
                            if position[idx, t - 1, 1] == 0:
                                need_to_stop = True
                                position[idx, t] = position[idx, t - 1]
                                action_selection[idx, t - 1] = 4
                            else:
                                position[idx, t] = position[idx, t - 1] + np.array([0, -1])
                                action_selection[idx, t - 1] = action_random_selection
                        elif action_random_selection == 2:
                            if position[idx, t - 1, 0] == 0:
                                need_to_stop = True
                                position[idx, t] = position[idx, t - 1]
                                action_selection[idx, t - 1] = 4
                            else:
                                position[idx, t] = position[idx, t - 1] + np.array([-1, 0])
                                action_selection[idx, t - 1] = action_random_selection
                        elif action_random_selection == 3:
                            if position[idx, t - 1, 0] == w-1:
                                need_to_stop = True
                                position[idx, t] = position[idx, t - 1]
                                action_selection[idx, t - 1] = 4
                            else:
                                position[idx, t] = position[idx, t - 1] + np.array([1, 0])
                                action_selection[idx, t - 1] = action_random_selection
                    else:
                        position[idx, t] = position[idx, t - 1]
                        action_selection[idx, t - 1] = 4
                    action_duriation -= 1
                else:
                    action_selection[idx, t - 1] = 4
                    position[idx, t] = position[idx, t - 1]
                if action_duriation <= 0:
                    new_continue_action_flag = True

    action_selection = torch.from_numpy(action_selection)
    position = torch.from_numpy(position)
    return position, action_selection