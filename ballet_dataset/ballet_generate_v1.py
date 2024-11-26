import numpy as np
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import skvideo
import skvideo.io
from einops import rearrange

import multiprocessing as mp

def save_video(video, dir, name):
    fps = '4'
    crf = '17'
    vid_out = skvideo.io.FFmpegWriter(f'{dir}/{name}.mp4',
                inputdict={'-r': fps},
                outputdict={'-r': fps, '-c:v': 'libx264', '-crf': crf,
                            '-preset': 'ultrafast', '-pix_fmt': 'yuv444p'}
    )

    for frame in video:
        vid_out.writeFrame(frame)
    vid_out.close()


dance_gt_map = {"circle_cw": 0,
                "circle_ccw": 1,
                "up_and_down": 2,
                "left_and_right": 3,
                "diagonal_uldr": 4,
                "diagonal_urdl": 5,
                "plus_cw": 6,
                "plus_ccw": 7,}
'''
v1: dance_delay=1, num_dancers=8,
v2: dance_delay=0, num_dancers=18
v3: dance_delay=0, num_dancers=3
'''

map_version = 'v3'
num_ep = 120000
dance_delay = 0
num_dancers = 36
print('env is associative')
import ballet_dataset.ballet_environment_v1 as ballet_env
import ballet_dataset.ballet_environment_core_v1 as ballet_core
rs = 2

max_steps = len(ballet_core.DANCER_POSITIONS) * (16 + dance_delay) # dance time is 16 steps

root_path = f'./ballet_dataset/{map_version}/dance_{num_dancers}_delay_{dance_delay}_STT_medium3'
vis_path = f'{root_path}/visualize'
Path(vis_path).mkdir(parents=True, exist_ok=True)

def generate_episode(max_steps, dance_delay, rs, path, episode_num):
    np.random.seed(episode_num)
    env = ballet_env.BalletEnvironment(
        num_dancers=num_dancers, dance_delay=dance_delay, max_steps=max_steps,
        rng=np.random.default_rng(seed=episode_num))
    obs = env.reset().observation[0]
    obss = [obs]
    info_dict = env.info_dict
    rooms = info_dict['rooms']
    positions = info_dict['positions']
    motions = info_dict['motions']
    label_time = info_dict['label_time']

    for i in range(max_steps-1):
        obs = env.step(0).observation[0]
        obss.append(obs)

    obss = (np.array(obss) * 255).astype(np.uint8)

    # modify the data for task purpose
    image = []
    for i in range(len(positions)):
        y, x = positions[i]
        image.append(obss[i*(ballet_core.DANCE_SEQUENCE_LENGTHS + dance_delay):(i+1)*(ballet_core.DANCE_SEQUENCE_LENGTHS + dance_delay),
                        (y-rs)*9:(y+rs+1)*9, (x-rs)*9:(x+rs+1)*9])

    # image, label, positions
    image = np.concatenate(image, axis=0)  # l,h,w,c
    rooms = np.array(rooms).astype(np.uint8) # l
    motions = np.array(motions) # l
    label_time = np.array(label_time)

    print(f'generating {episode_num}th episode...')
    dict = {'image': image, 'target_idx': 0, 'dancers': 0, 'rooms': rooms, 'motions': motions}
    np.savez(f'{path}/{episode_num}.npz', **dict)

    # save
    if episode_num < 10 and 'train' in path:
        full_vid = obss
        part_vid = image
        # store as gif
        skvideo.io.vwrite(f'{vis_path}/full_video_{episode_num}.gif', full_vid, inputdict={'-r': '4'}, outputdict={'-r': '4'})
        #save_video(full_vid, dir=vis_path, name=f'full_video_{episode_num}')
        #save_video(part_vid, dir=vis_path, name=f'part_video_{episode_num}')
        #if dance_delay == 16:  # image
        #    traj_img = rearrange(image, '(n_r n_c) h w c -> (n_r h) (n_c w) c', n_c=16)
        #    plt.imsave(f'{vis_path}/traj_{episode_num}.png', traj_img)

pool = mp.Pool(processes=10)

data_type = {'train': int(num_ep * 0.9), 'eval': int(num_ep * 0.1)}
used_nums = 0
total_eposode_nums = list(range(num_ep))
for type_, n_ep in data_type.items():
    path = f'{root_path}/{type_}'
    Path(path).mkdir(parents=True, exist_ok=True)
    episode_nums = total_eposode_nums[used_nums:n_ep]
    used_nums += n_ep

    # use multiprocessing to parallelize the loop over episodes
    pool.starmap(generate_episode, [(max_steps, dance_delay, rs, path, episode_num) for episode_num in episode_nums])

pool.close()
