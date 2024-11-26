'''
This is the code for generating pos, action sequence generation.
We additionally add knn result for faster learning
'''

import random
import numpy as np
from omegaconf import OmegaConf
import skvideo.io
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from einops import rearrange, repeat
import time
np.random.seed(0)

def save_video(video, dir):
    fps = '8'
    crf = '17'
    vid_out = skvideo.io.FFmpegWriter(f'{dir}.mp4', 
                inputdict={'-r': fps},
                outputdict={'-r': fps, '-c:v': 'libx264', '-crf': crf, 
                            '-preset': 'ultrafast', '-pix_fmt': 'yuv444p'}
    )

    for frame in video:
        vid_out.writeFrame(frame)
    vid_out.close()


def random_walk_wo_wall_ffhq(len_mv, length, n_action, init_poss=None):
    
    # required values
    w = len_mv
    l = length
    a = n_action

    # construct position and action
    position = np.zeros((l+1, 2), np.uint8)
    action_selection = np.zeros((l), np.uint8)
    new_continue_action_flag = True
    
    for t in range(l+1):
        
        if t == 0:
            if init_poss is not None:
                position[t] = init_poss
            else:
                position[t] = np.random.randint(0, w, size=(2))
        else:
            if new_continue_action_flag:
                new_continue_action_flag = False
                need_to_stop = False

                while 1:
                    action_random_selection = np.random.randint(0, 4, size=(1))
                    if not (action_random_selection == 0 and position[t - 1, 1] == w-1):
                        if not (action_random_selection == 1 and position[t - 1, 1] == 0):
                            if not (action_random_selection == 2 and position[t - 1, 0] == 0):
                                if not (action_random_selection == 3 and position[t - 1, 0] == w-1):
                                    break
                
                # action_duriation = np.random.poisson(2, 1)
                action_duriation = 1

            if action_duriation > 0:
                if not need_to_stop:
                    if action_random_selection == 0:
                        if position[t - 1, 1] == w-1:
                            need_to_stop = True
                            position[t] = position[t - 1]
                            action_selection[t - 1] = 4
                        else:
                            position[t] = position[t - 1] + np.array([0, 1])
                            action_selection[t - 1] = action_random_selection
                    elif action_random_selection == 1:
                        if position[t - 1, 1] == 0:
                            need_to_stop = True
                            position[t] = position[t - 1]
                            action_selection[t - 1] = 4
                        else:
                            position[t] = position[t - 1] + np.array([0, -1])
                            action_selection[t - 1] = action_random_selection
                    elif action_random_selection == 2:
                        if position[t - 1, 0] == 0:
                            need_to_stop = True
                            position[t] = position[t - 1]
                            action_selection[t - 1] = 4
                        else:
                            position[t] = position[t - 1] + np.array([-1, 0])
                            action_selection[t - 1] = action_random_selection
                    elif action_random_selection == 3:
                        if position[t - 1, 0] == w-1:
                            need_to_stop = True
                            position[t] = position[t - 1]
                            action_selection[t - 1] = 4
                        else:
                            position[t] = position[t - 1] + np.array([1, 0])
                            action_selection[t - 1] = action_random_selection
                else:
                    position[t] = position[t - 1]
                    action_selection[t - 1] = 4
                action_duriation -= 1
            else:
                action_selection[t - 1] = 4
                position[t] = position[t - 1]
            if action_duriation <= 0:
                new_continue_action_flag = True

    return position, action_selection


def weak_self_avoiding_walk_wo_wall_ffhq(len_mv, length, n_action, init_poss=None):
    
    # required values
    w = len_mv
    l = length
    a = n_action

    # construct position and action
    position = np.zeros((l+1, 2), np.uint8)
    action_selection = np.zeros((l), np.uint8)
    visit_count = np.zeros((w, w), np.int32)
    deltas = [np.array([0, 1]), np.array([0, -1]), np.array([-1, 0]), np.array([1, 0])]
    
    for t in range(l+1):
        
        if t == 0:
            if init_poss is not None:
                position[t] = init_poss
            else:
                position[t] = np.random.randint(0, w, size=(2))
        else:
            possible_actions = []
            if not position[t - 1, 1] == w-1:
                possible_actions.append(0)
            if not position[t - 1, 1] == 0:
                possible_actions.append(1) 
            if not position[t - 1, 0] == 0:
                possible_actions.append(2)    
            if not position[t - 1, 0] == w-1:
                possible_actions.append(3)       
            possible_actions = np.array(possible_actions)
            
            visits = []
            for act in possible_actions:
                pos = position[t - 1] + deltas[act]
                visits.append(visit_count[pos[0], pos[1]])
            
            visits = np.array(visits)
            possible_actions = possible_actions[np.min(visits) == visits]
            action_random_selection = np.random.choice(possible_actions)
            
            if action_random_selection == 0:
                    position[t] = position[t - 1] + np.array([0, 1])
                    action_selection[t - 1] = action_random_selection
            elif action_random_selection == 1:
                    position[t] = position[t - 1] + np.array([0, -1])
                    action_selection[t - 1] = action_random_selection
            elif action_random_selection == 2:
                    position[t] = position[t - 1] + np.array([-1, 0])
                    action_selection[t - 1] = action_random_selection
            elif action_random_selection == 3:
                    position[t] = position[t - 1] + np.array([1, 0])
                    action_selection[t - 1] = action_random_selection

        visit_count[position[t][0], position[t][1]] += 1
        
    return position, action_selection


exp_config = 'ffhq'
args_base = OmegaConf.load(f'./htm/config/{exp_config}.yaml')
args_cli = OmegaConf.from_cli()
args = OmegaConf.merge(args_base, args_cli)
len_mv = args.data.len_mv
n_action = args.data.n_action
memorize = args.data.memorize
imagine = args.data.imagine
n_clusters_list = [8, 16, 32]
rand_case = 3
chunk_size = 8

type_ = {
        # 'train': 372000, 
        # 'eval': 7000,  
        'train': 1000,
        'eval': 1000, 
        }

# make knn clusters 
grid_np = []
for j in range(len_mv):
    for i in range(len_mv):
        grid_np.append([j, i])
grid_np = np.array(grid_np)
grid_norm = grid_np.astype(np.float32) / (len_mv - 1)

cluster_list = np.zeros((len(n_clusters_list), rand_case, len_mv, len_mv)).astype(np.int32)

for j, n_clusters in enumerate(n_clusters_list):
    for i in range(rand_case):
        kmeans = KMeans(n_clusters=n_clusters, random_state=i).fit(grid_norm)
        clusters = kmeans.labels_.astype(np.uint8)
        cluster_list[j, i] = rearrange(clusters, '(y x) -> y x', y=len_mv)

cluster_list = rearrange(cluster_list, 'n r y x -> (n r) (y x)')  # 9,y*x
print(rearrange(cluster_list, 'n (y x) -> n y x', y=len_mv))

for i, cluster in enumerate(cluster_list):
    cluster = rearrange(cluster, '(y x) -> y x', y=len_mv)
    nc = np.max(cluster) + 1
    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(nc)]
    fig = plt.figure(figsize=(5, 5))
    fig.clf()
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_aspect('equal')
    plt.axis([-1, len_mv, -1, len_mv])
    plt.gca().invert_yaxis()
    
    for g in range(nc):
        # print(grid_np.shape, cluster.shape)
        ix = np.where(rearrange(cluster, 'y x -> (y x)') == g)
        # print(ix)
        plt.scatter(grid_np[:, 1][ix], grid_np[:, 0][ix], color=colors[g], label=g)
    
    plt.tight_layout()
    plt.savefig(f'./ffhq_dataset/clusters_{i}.png')
    plt.close()


for t in type_.keys():
    
    positions = []
    actions = []
    knn_idxs = []
    sort_idxs_new = []
    
    s0 = time.time()
    for i in range(type_[t]):
        if (i % 1000 == 0):
            print(f'generating {t} {i}-th samples...')
            s1 = time.time()
            print(s1 - s0)
            s0 = s1
        # memorization phase
        position_m, action_m = weak_self_avoiding_walk_wo_wall_ffhq(len_mv=len_mv,
                                                                    length=memorize,
                                                                    n_action=n_action)  # memorize+1,2  memorize
        position_m, action_m = position_m[:-1], action_m

        # generation phase -- start from new position
        position_q, action_q = weak_self_avoiding_walk_wo_wall_ffhq(len_mv=len_mv,
                                                                    length=imagine,
                                                                    n_action=n_action)  # imagine+1,2  imagine
        position = np.concatenate((position_m, position_q), axis=0)  # memorize+imagine+1,2
        action = np.concatenate((action_m, action_q), axis=0)  # memorize+imagine
        positions.append(position)
        actions.append(action)
        
        # knn index 
        posix = position[:memorize, 0] * len_mv + position[:memorize, 1]  # memorize
        knn_idx = cluster_list[np.arange(len(cluster_list)).reshape(-1, 1), posix]  # 9,memorize
        knn_idxs.append(knn_idx)
        
        sort_idx = np.argsort(knn_idx, axis=1)
        knn_idx_sort = knn_idx[np.arange(9).reshape(-1, 1), sort_idx]
        # print(rearrange(knn_idx_sort[8], '(y x) -> y x', x=chunk_size))
        sort_idx_new = np.ones((9, memorize*2)).astype(np.int32) * -1
        for l in range(9):
            cur_pos1 = 0
            cur_pos2 = 0
            n_clusters = np.max(knn_idx[l]) + 1
            for k in range(n_clusters):
                n_samples_cls = (knn_idx_sort[l] == k).sum()
                remain = n_samples_cls % chunk_size 
                attach = chunk_size - remain
                if n_samples_cls < chunk_size:
                    sort_idx_new[l, cur_pos2:cur_pos2+n_samples_cls] = sort_idx[l, cur_pos1:cur_pos1+n_samples_cls]
                    sort_idx_new[l, cur_pos2+n_samples_cls:cur_pos2+n_samples_cls+attach] = -1
                    cur_pos2 += n_samples_cls + attach
                else:
                    if attach > chunk_size // 2:
                        sort_idx_new[l, cur_pos2:cur_pos2+n_samples_cls-remain] = sort_idx[l, cur_pos1:cur_pos1+n_samples_cls-remain]
                        cur_pos2 += n_samples_cls - remain
                    else:
                        sort_idx_new[l, cur_pos2:cur_pos2+n_samples_cls] = sort_idx[l, cur_pos1:cur_pos1+n_samples_cls]
                        sort_idx_new[l, cur_pos2+n_samples_cls:cur_pos2+n_samples_cls+attach] = -1
                        cur_pos2 += n_samples_cls + attach
                cur_pos1 += n_samples_cls
        
        sort_idxs_new.append(sort_idx_new)
        
    positions = np.array(positions)  # n,memorize+imagine+1,2
    actions = np.array(actions)  # n,memorize+imagine
    knn_idxs = np.array(knn_idxs)  # n,9,memorize
    sort_idxs_new = np.array(sort_idxs_new)  # n,9,memorize*2
    
    dict = {'positions': positions, 
            'actions': actions,
            'knn_idxs': knn_idxs,
            'sort_idxs_new': sort_idxs_new}
    np.savez(f'./ffhq_dataset/ffhq_pos_data/{t}_{type_[t]}_knn_sort.npz', **dict)

