import os
import math
import numpy as np
import itertools
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

num_rooms = 25
space = np.arange(num_rooms)
action_set = np.arange(4)
max_steps = 32*40

def act(agent_pos, action):
    if action == 0: # up
        if agent_pos < int(math.sqrt(num_rooms)):
            return agent_pos
        else:
            return agent_pos - int(math.sqrt(num_rooms))
    elif action == 1: # down
        if agent_pos > num_rooms - int(math.sqrt(num_rooms)) - 1:
            return agent_pos
        else:
            return agent_pos + int(math.sqrt(num_rooms))
    elif action == 2: # left
        if agent_pos % int(math.sqrt(num_rooms)) == 0:
            return agent_pos
        else:
            return agent_pos - 1
    elif action == 3: # right
        if agent_pos % int(math.sqrt(num_rooms)) == (int(math.sqrt(num_rooms)) - 1):
            return agent_pos
        else:
            return agent_pos + 1
    else:
        raise ValueError(f"action {action} not recognized!")
    
chunk_size = 32
max_len = num_rooms * chunk_size + max_steps
batch_agent_poses_times = []
pm_batch_agent_poses_times = []

def process_task(b, num_in_thread, space, action_set, act, num_rooms, max_steps, chunk_size, max_len, train=True):
    for _b in range(b, b+num_in_thread):
        agent_pos, step = np.random.choice(space), 0
        agent_poses_times = [[agent_pos, step]]
        pm_agent_poses_times = [[] for _ in range(num_rooms)]
        pm_agent_poses_times[agent_pos].append([agent_pos, step])
        while step < max_steps - 1:
            action = np.random.choice(action_set)
            agent_pos = act(agent_pos, action)
            agent_poses_times.append([agent_pos, step + 1])
            pm_agent_poses_times[agent_pos].append([agent_pos, step + 1])
            step += 1

        for i in range(num_rooms):
            padding_len = chunk_size - len(pm_agent_poses_times[i]) % chunk_size
            if padding_len > 0:
                pm_agent_poses_times[i] += [[0, -1] for _ in range(padding_len)]
        
        pm_agent_poses_times = list(itertools.chain.from_iterable(pm_agent_poses_times))
        padding_len = max_len - len(pm_agent_poses_times)
        if padding_len > 0:
            pm_agent_poses_times = np.concatenate((pm_agent_poses_times, np.array([[0, -1] for _ in range(padding_len)])), axis=0)
    
        subfolder = "train" if train else "eval"
        # make folder if not exist
        if not os.path.exists(f"./ballet_dataset/dance_random_actions/{num_rooms}_rooms/{subfolder}"):
            os.makedirs(f"./ballet_dataset/dance_random_actions/{num_rooms}_rooms/{subfolder}")
        np.save(f"./ballet_dataset/dance_random_actions/{num_rooms}_rooms/{subfolder}/agent_poses_times_{_b}.npy", agent_poses_times)
        np.save(f"./ballet_dataset/dance_random_actions/{num_rooms}_rooms/{subfolder}/pm_agent_poses_times_{_b}.npy", pm_agent_poses_times)

# num_data = 108000
num_data = 1200
for b in tqdm(range(num_data)):
    process_task(b, 1, space, action_set, act, num_rooms, max_steps, chunk_size, max_len, True)
# num_data = 12000
num_data = 1200
for b in tqdm(range(num_data)):
    process_task(b, 1, space, action_set, act, num_rooms, max_steps, chunk_size, max_len, False)