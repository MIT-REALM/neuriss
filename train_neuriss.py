import argparse
import torch
import time
import os
import yaml

from neuriss.env.env import make_env
from neuriss.env.affine import ControlAffineEnv
from neuriss.controller.iss_lyapunov import NeuralISSLyapunov
from neuriss.trainer.iss_lyapunov_trainer import ISSLyapunovTrainer
from neuriss.trainer.utils import set_seed, init_logger


def train_iss(args):
    # set up training devices and loggers
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    device = torch.device('cuda' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu')
    set_seed(args.seed)
    log_path, writer, model_path = init_logger(args.log_path, args.env, 'neuriss', args.seed, vars(args))

    # set up environment
    env = make_env(env_id=args.env, device=device)
    if isinstance(env, ControlAffineEnv):
        if args.u_nominal is not None:
            print('> Nominal controller loaded')
            env.load_nominal_controller(args.u_nominal)
        else:
            if env.use_linearized_controller:
                print('> Using computed linearized controller')
                env.compute_linearized_controller()
            else:
                print('> Using predefined nominal controller')
    env.train()

    # load training hyperparams
    cur_path = os.getcwd()
    if os.path.exists(os.path.join(cur_path, 'neuriss/env/hyperparams', f'{env}.yaml')):
        print('> Using tuned hyper-parameters')
        with open(os.path.join(cur_path, 'neuriss/env/hyperparams', f'{env}.yaml')) as f:
            hyper_params = yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise KeyError(f'Cannot find hyper-parameters for {env}. '
                       f'Please put {env}.yaml in neuriss/env/hyperparams to specify hyper-parameters!')

    # set up iss lyapunov controller
    lyapunov = NeuralISSLyapunov(
        state_dims=env.n_dims,
        ctrl_dims=env.n_controls,
        share_lyapunov=env.share_lyapunov,
        share_ctrl=env.share_ctrl,
        device=device,
        goal_point=env.goal_point,
        state_std=env.state_std,
        u_eq=env.u_eq,
        env=env,
        lambd=hyper_params['lambda'],
        residue=hyper_params['residue']
    )

    # set up trainer
    trainer = ISSLyapunovTrainer(
        env=env,
        lyapunov_func=lyapunov,
        writer=writer,
        save_path=model_path,
        controller_ref=env.u_nominal,
        hyper_params=hyper_params
    )

    # start training
    print('> Training ISS Lyapunov controller...')
    start_time = time.time()
    trainer.train(
        n_iter=args.n_iter,
        batch_size=args.batch_size,
        init_iter=args.init_iter,
        eval_interval=args.n_iter // 50,
        adjust_weight=args.adjust_weight
    )
    print(f'> Done in {time.time() - start_time:.0f}s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # custom
    parser.add_argument('--env', type=str, required=True,
                        help='name of the environment')
    parser.add_argument('--n-iter', type=int, default=10000,
                        help='number of training iterations')
    parser.add_argument('--init-iter', type=int, default=3000,
                        help='number of initializing iterations')

    # default
    parser.add_argument('--batch-size', type=int, default=2048,
                        help='batch size of training data')
    parser.add_argument('--adjust-weight', action='store_true', default=False,
                        help='decrease the weight of controller loss after some iteration')
    parser.add_argument('--u-nominal', type=str, default=None,
                        help='path to the pre-defined nominal controller')
    parser.add_argument('--gpus', type=int, default=0,
                        help='index of the training gpu')
    parser.add_argument('--log-path', type=str, default='./logs',
                        help='path to save training logs')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disable cuda')

    args = parser.parse_args()
    train_iss(args)
