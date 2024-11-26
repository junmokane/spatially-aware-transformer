import numpy as np
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import skvideo.io
from einops import rearrange


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

map_version = 'v2'
num_ep = 120000
dance_delay = 0
num_dancers = 18
max_steps = num_dancers * (16 + dance_delay)
data_type = {'train': int(num_ep * 0.9), 'eval': int(num_ep * 0.1)}

print('env is v2')
from ballet_dataset import ballet_environment_v2 as ballet_env
from ballet_dataset import ballet_environment_core_v2 as ballet_core
rs = 2


env = ballet_env.BalletEnvironment(
    num_dancers=num_dancers, dance_delay=dance_delay, max_steps=max_steps,
    rng=np.random.default_rng(seed=0))


root_path = f'./ballet_dataset/{map_version}/dance_{num_dancers}_delay_{dance_delay}_xxx'
vis_path = f'{root_path}/visualize'
Path(vis_path).mkdir(parents=True, exist_ok=True)

for type_, n_ep in data_type.items():

    path = f'{root_path}/{type_}'
    Path(path).mkdir(parents=True, exist_ok=True)
    
    for j in range(n_ep):
        
        obs = env.reset().observation[0]
        obss = [obs]
        info_dict = env.info_dict
        positions = info_dict['positions']
        spaces = info_dict['spaces']
        motions = info_dict['motions']
        task_heur = info_dict['task_heur']
        task_label = info_dict['task_label']
        
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
        # positions = np.array(positions).astype(np.uint8)
        spaces = np.array(spaces).astype(np.uint8)
        motions_ = np.array([dance_gt_map[motions[i]] for i in range(num_dancers)]).astype(np.uint8)
        task_heur = np.array(task_heur).astype(np.uint8)
        task_label = np.array(task_label).astype(np.uint8)

        print(f'generating {j}th episode...')
        dict = {'image': image, 'space': spaces, 'motion': motions_, 
        'task_heur': task_heur, 'task_label': task_label}
        np.savez(f'{path}/{j}.npz', **dict)
        
        # save
        if j < 10 and type_=='train':
            full_vid = obss
            part_vid = image
            save_video(full_vid, dir=vis_path, name=f'full_video_{j}')
            save_video(part_vid, dir=vis_path, name=f'part_video_{j}')
            traj_img = rearrange(image, '(n_r n_c) h w c -> (n_r h) (n_c w) c', n_c=16)
            plt.imsave(f'{vis_path}/traj_{j}.png', traj_img)
    



