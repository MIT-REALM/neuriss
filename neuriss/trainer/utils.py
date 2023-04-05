import os
import torch
import numpy as np
import datetime
import yaml

from torch.utils.tensorboard import SummaryWriter
from typing import Tuple


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def init_logger(
        log_path: str,
        env_name: str,
        algo_name: str,
        seed: int,
        args: dict = None,
        hyper_params: dict = None,
) -> Tuple[str, SummaryWriter, str]:
    """
    Initialize the logger. The logger dir should include the following path:
        - <log folder>
            - <env name>
                - <algo name>
                    - seed<seed>_<experiment time>
                        - settings.yaml: the experiment setting
                        - summary: training summary
                        - models: saved models
    Parameters
    ----------
    log_path: str
        name of the log folder
    env_name: str
        name of the training environment
    algo_name: str
        name of the algorithm
    seed: int
        random seed used
    args: dict
        arguments to be written down: {argument name: value}
    hyper_params: dict
        hyper-parameters for training

    Returns
    -------
    log_path: str
        path of the log
    writer: SummaryWriter
        summary dir
    model_path: str
        models dir
    """
    # make log path
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    # make path with specific env
    if not os.path.exists(os.path.join(log_path, env_name)):
        os.mkdir(os.path.join(log_path, env_name))

    # make path with specific algorithm
    if not os.path.exists(os.path.join(log_path, env_name, algo_name)):
        os.mkdir(os.path.join(log_path, env_name, algo_name))

    # record the experiment time
    start_time = datetime.datetime.now()
    start_time = start_time.strftime('%Y%m%d%H%M%S')
    if not os.path.exists(os.path.join(log_path, env_name, algo_name, f'seed{seed}_{start_time}')):
        os.mkdir(os.path.join(log_path, env_name, algo_name, f'seed{seed}_{start_time}'))

    # set up log, summary writer
    log_path = os.path.join(log_path, env_name, algo_name, f'seed{seed}_{start_time}')
    writer = SummaryWriter(log_dir=os.path.join(log_path, 'summary'))

    # make path for saving models
    model_path = os.path.join(log_path, 'models')
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    # write args
    log = open(os.path.join(log_path, 'settings.yaml'), 'w')
    if args is not None:
        for key in args.keys():
            log.write(f'{key}: {args[key]}\n')
    if 'algo' not in args.keys():
        log.write(f'algo: {algo_name}\n')
    if hyper_params is not None:
        log.write('hyper_params:\n')
        for key1 in hyper_params.keys():
            if type(hyper_params[key1]) == dict:
                log.write(f'  {key1}: \n')
                for key2 in hyper_params[key1].keys():
                    log.write(f'    {key2}: {hyper_params[key1][key2]}\n')
            else:
                log.write(f'  {key1}: {hyper_params[key1]}\n')
    else:
        log.write('hyper_params: using default hyper-parameters')
    log.close()

    return log_path, writer, model_path


def read_settings(path: str):
    with open(os.path.join(path, 'settings.yaml')) as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
    return settings
