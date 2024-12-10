# Spatially-Aware Transformers for Embodied Agents

This repository is the official implementation of [Spatially-Aware Transformers for Embodied Agents](https://arxiv.org/abs/2402.15160). We provide the code for core experiments in our paper.

## Requirements

To install requirements:

```setup
conda create -n sat python=3.7
conda activate sat
pip install -r requirements.txt
```

## Room Ballet Short-Stay 
### Dataset

Room Ballet Short-Stay dataset consists of two parts. **Dancer dataset** and **Trajectory dataset**. You need both datasets to run the Room Ballet experiments.

**Dancer dataset**
Dancer dataset consists of videos with dancers' dancing! Each video is composed of 36 dancers where each dancer dances for 16 time steps sequentially. Dataset is available [here](https://drive.google.com/drive/folders/1COlErYgyL8wk1iseL23giedDSEbCfytv) in short_stay_dataset folder. Put the dataset folder in `ballet_dataset/v3` folder. If you want to generate the data by yourself, run the following command.
```
python -m ballet_dataset.ballet_generate_v1
```


**Trajectory dataset**
Trajectory dataset consists of agent's trajectory. Each file has agent's random walk trajectory in $N \times N$ grid. Here, each grid means dancer's room. Dataset is available [here](https://drive.google.com/drive/folders/1COlErYgyL8wk1iseL23giedDSEbCfytv) in short_stay_dataset folder. If you want to generate this by yourself, run `ballet_dataset/generate_rw.py` for data generation. 

### Training

Here is the command for running SAT-PM-PH on Short-Stay task. 
```
python -m htm.rb_shortstay.train model_type='shcams'  # SAT-PM-PH
```
Different models can be trained by changing `model_type`. Use `model_type='cams'`for SAT-FIFO, and `model_type='thcams'` for SAT-FIFO-TH.


## Room Ballet Multi-Task 
### Dataset

Dataset is available [here](https://drive.google.com/drive/folders/1COlErYgyL8wk1iseL23giedDSEbCfytv) in multi_tasks_dataset folder. It consists of dancers' dances and agent's trajectory. It also includes the memory index information for each strategy (FIFO, LIFO, MVFO, LVFO). If you want to generate the data by yourself, run the following command.
```
python -m ballet_dataset.ballet_generate_v2
```

### Training

Here is the command for running SAT-AMA on Room Ballet Multi-Task.
```
python -m htm.rb_multi.train task='all' heur='ama'  # SAT-AMA
```
Different heuristics can be used by changing `heur`. There are 5 options: `fifo`, `lifo`, `lvfo`, `mvfo`, `ama`. Also, different task can be used by changing `task`. There are 5 options: `fifo`, `lifo`, `lvfo`, `mvfo`, `all`. 

## FFHQ Generation

### Dataset

FFHQ dataset consists of two parts like in Room Ballet dataset. **Image dataset** and **Trajectory dataset**. You need both datasets to run the FFHQ Generation experiments.

**Image dataset** We use the `thumbnail128x128` from FFHQ dataset. You can download it [here](https://github.com/NVlabs/ffhq-dataset). Put `thumbnail128x128` folder in `ffhq_dataset` folder. 

**Trajectory dataset**
Trajectory dataset consists of agent's trajectory. Dataset is available [here](https://drive.google.com/drive/folders/1COlErYgyL8wk1iseL23giedDSEbCfytv) (knn_sort.zip file). Each file has agent's trajectory in $10 \times 10$ grid. In data generating process, random place cluster is generated first, and the agent randomly walks in the grid space. If you want to generate data by yourself, run `ffhq_dataset/ffhq_pos_gen.py`. 

### Training

Here is the command for running SAT-PM($N$) on FFHQ generation task.
```
python -m htm.ffhq.train model_type='space_model' cluster_rand=0
```
`cluster_rand` controls the number of place clusters $N$. Use `cluster_rand=0`for 8 clusters, `cluster_rand=3`for 16 clusters, and `cluster_rand=6`for 32 clusters.


Different models can be trained by changing `model_type`. Use `model_type='cams'`for SAT-FIFO, `model_type='cam'` for T-FIFO, and `model_type='cama'` for T-FIFO-A.

```
python -m htm.ffhq.train model_type='cams'
```

## Contact
If there is any problem, please open an issue on this repository or send email to Junmo Cho (junmokane12@gmail.com)
