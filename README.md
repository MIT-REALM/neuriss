# NeurISS

[![Conference](https://img.shields.io/badge/L4DC-Accepted-success)](https://mit-realm.github.io/neuriss-website/) [![Arxiv](http://img.shields.io/badge/arxiv-2303.14564-B31B1B.svg)](https://arxiv.org/abs/2303.14564)[![Conference](https://img.shields.io/badge/Project%20page-success)](https://mit-realm.github.io/neuriss-website/)

Official implementation of the L4DC 2023 paper: [S Zhang](https://syzhang092218-source.github.io), [Y Xiu](https://yumengxiu.github.io/), [G Qu](https://www.guannanqu.com/), [C Fan](https://chuchu.mit.edu): "[Compositional Neural Certificates for Networked Dynamical Systems](https://mit-realm.github.io/neuriss-website/)".

## Dependencies

We recommend to use [CONDA](https://www.anaconda.com/) to install the requirements:

```bash
conda create -n neuriss python=3.10
conda activate neuriss
pip install -r requirements.txt
```

## Install

Install NeurISS: 

```bash
pip install -e .
```

## Run

### Environments

We provide 3 environments including `Microgrid-IEEE5`, `PlatoonSin5`, and `PlanarDroneSin2x2`. One can also create other environments by changing the number of agents. For example, one can use `PlatoonSin10`, `PlanarDroneSin3x2`. However, for the microgrid environment, we only provide `Microgrid-IEEE4` and `Microgrid-IEEE5` because this environment depends on the data provided in [[1]](#references). 

### Hyper-parameters

To reproduce the results shown in our paper, one can refer to [`hyperparams.yaml`](hyperparams.yaml).

### Train NeurISS

Use the following command to train NeurISS:

```bash
python train_neuriss.py --env Microgrid-IEEE5 --n-iter 10000 --init-iter 0
```

Training options:

- `--env`: name of the environment
- `--n-iter`: number of training iterations
- `--init-iter`: number of initializing iterations
- `--batch-size`: batch size of training data
- `--adjust-weight`: decrease the weight of controller loss after some iteration
- `--u-nominal`: path to the pre-defined nominal controller
- `--gpus`: index of the training gpu
- `--no-cuda`: disable cuda
- `--log-path`: path to save training logs
- `--seed`: random seed

### Train RL

Use the following command to train RL algorithms:

```bash
python train_rl.py --env Microgrid-IEEE5 --algo ppo --steps 5000000
```

We provide the implementation of 3 RL algorithms including PPO [[2]](#references), LYPPO [[3]](#references), and MAPPO [[4]](#references). 

Training options:

- `--env`: name of the environment
- `--algo`: name of the algorithm
- `--steps`: number of training steps
- `--gpus`: index of the training gpu
- `--seed`: random seed
- `--log-path`: path to save training logs
- `--no-cuda`: disable cuda

### Train NCLF

Use the following command to train NCLF [[5]](#references):

```bash
python train_clf.py --env Microgrid-IEEE5 --n-iter 10000
```

Training options: 

- `--env`: name of the environment
- `--n-iter`: number of training iterations
- `--gpus`: index of the training gpu
- `--seed`: random seed
- `--no-cuda`: disable cuda
- `--log-path`: path to save training logs
- `--batch-size`: batch size of training data

### Evaluation

Use the following command to test the agents:

```bash
python test.py --path <path-to-log> --epi 20
```

Use the following command to test the nominal controller

```bash
python test.py --env Microgrid-IEEE5 --epi 20
```

Options:

- `--path`: path to the log folder (start with `seed`)
- `--epi`: number of episode
- `--env`: name of the environment (no need to specify if you want to test on the training environment)
- `--iter`: the iteration of the controller (no need to specifiy if you want to test the best controller)
- `--no-cuda`: disable cuda
- `--no-video`: do not make video (for faster test)

To test in an environment with different number of agents, use `--env` to specify the environment you want to test on.

### Tracking Training Process

We use [TensorBoard](https://www.tensorflow.org/tensorboard) to track the training process. Use the following line to see how everything goes on during the training:

```bash
tensorboard --logdir='<path-to-log>'
```

### Pre-trained Models

We provide the pre-trained models in the folder [`./pretrained`](pretrained). 

## Citation

```
@inproceedings{zhang2023neuriss,
      title={Compositional Neural Certificates for Networked Dynamical Systems},
      author={Songyuan Zhang and Yumeng Xiu and Guannan Qu and Chuchu Fan},
      booktitle={5th Annual Learning for Dynamics {\&} Control Conference},
      year={2023},
}
```

## References
[[1]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9559389) Tong Huang, Sicun Gao, and Le Xie. A neural lyapunov approach to transient stability assessment of power electronics-interfaced networked microgrids. *IEEE Transactions on Smart Grid*, 13(1): 106–118, 2021.

[[2]](https://arxiv.org/pdf/1707.06347.pdf) John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*, 2017.

[[3]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9560886) Ya-Chien Chang and Sicun Gao. Stabilizing neural control using self-learned almost lyapunov critics. In *2021 IEEE International Conference on Robotics and Automation (ICRA)*, pages 1803–1809. IEEE, 2021.

[[4]](https://proceedings.neurips.cc/paper_files/paper/2022/file/9c1535a02f0ce079433344e14d910597-Paper-Datasets_and_Benchmarks.pdf) Chao Yu, Akash Velu, Eugene Vinitsky, Yu Wang, Alexandre Bayen, and Yi Wu. The surprising effectiveness of ppo in cooperative multi-agent games. *arXiv preprint arXiv:2103.01955*, 2021.
