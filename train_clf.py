import argparse
import torch
import os
import time
import yaml

from neuriss.trainer.utils import set_seed, init_logger
from neuriss.trainer.clf_trainer import CLFTrainer
from neuriss.controller.neural_clf_controller import NeuralCLFController
from neuriss.env.env import make_env


def train_clf(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    set_seed(args.seed)
    device = torch.device('cuda' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu')
    print(f'> Training with {device}')
    env = make_env(env_id=args.env, device=device)
    env.train()

    log_path, writer, model_path = init_logger(args.log_path, args.env, 'clf', args.seed, vars(args))

    # load training hyperparams
    cur_path = os.getcwd()
    if os.path.exists(os.path.join(cur_path, 'neuriss/env/hyperparams', f'{env}.yaml')):
        print('> Using tuned hyper-parameters')
        with open(os.path.join(cur_path, 'neuriss/env/hyperparams', f'{env}.yaml')) as f:
            hyper_params = yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise KeyError(f'Cannot find hyper-parameters for {env}. '
                       f'Please put {env}.yaml in neuriss/env/hyperparams to specify hyper-parameters!')

    clf_controller = NeuralCLFController(
        state_dim=sum(env.n_dims),
        action_dim=sum(env.n_controls),
        device=device,
        goal_point=env.goal_point,
        u_eq=env.u_eq,
        state_std=env.state_std,
        clf_lambda=hyper_params['lambda'],
        env=env,
        residue=hyper_params['residue']
    ).to(device)

    # start training
    print('> Training CLF controller...')
    start_time = time.time()
    clf_trainer = CLFTrainer(
        controller=clf_controller,
        env=env,
        writer=writer,
        save_path=model_path,
        controller_ref=env.u_nominal,
        hyper_params=hyper_params
    )
    clf_trainer.train(
        n_iter=args.n_iter,
        batch_size=args.batch_size,
        eval_interval=args.n_iter // 50
    )
    print(f'> Done in {time.time() - start_time:.0f}s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # custom
    parser.add_argument('--env', type=str, required=True,
                        help='name of the environment')
    parser.add_argument('--gpus', type=int, default=0,
                        help='index of the training gpu')
    parser.add_argument('--n-iter', type=int, default=10000,
                        help='number of training iterations')

    # default
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disable cuda')
    parser.add_argument('--batch-size', type=int, default=2048,
                        help='batch size of training data')
    parser.add_argument('--log-path', type=str, default='./logs',
                        help='path to save training logs')

    args = parser.parse_args()
    train_clf(args)
