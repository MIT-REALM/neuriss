import os
import torch
import argparse
import time

from neuriss.trainer.utils import set_seed, init_logger
from neuriss.env.env import make_env
from neuriss.rl.agents import get_agent
from neuriss.trainer.rl_trainer import RLTrainer


def train_rl(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    set_seed(args.seed)
    device = torch.device('cuda' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu')
    print(f'> Training with {device}')
    env = make_env(env_id=args.env, device=torch.device('cpu'))
    env.train()

    # set up logger
    log_path, writer, model_path = init_logger(args.log_path, args.env, args.algo, args.seed, vars(args))

    # get agent
    agent = get_agent(
        algo=args.algo,
        state_dims=env.n_dims,
        action_dims=env.n_controls,
        device=device,
        goal_point=env.goal_point,
        u_eq=env.u_eq,
        state_std=env.state_std,
        ctrl_std=torch.ones(sum(env.n_controls), device=device),
        env=env
    )

    # setup RL trainer
    trainer = RLTrainer(
        env=env,
        agent=agent,
        writer=writer,
        model_dir=model_path,
        num_steps=args.steps,
        eval_interval=args.steps // 50,
    )
    print(f'> Training {args.algo.upper()}...')
    start_time = time.time()
    trainer.train()
    print(f'> Done in {time.time() - start_time:.0f}s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # custom
    parser.add_argument('--env', type=str, required=True,
                        help='name of the environment')
    parser.add_argument('--steps', type=int, required=True,
                        help='number of training steps')
    parser.add_argument('--gpus', type=int, default=0,
                        help='index of the training gpu')
    parser.add_argument('--algo', type=str, default='ppo',
                        help='name of the algorithm')

    # default
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--log-path', type=str, default='./logs',
                        help='path to save training logs')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disable cuda')

    args = parser.parse_args()
    train_rl(args)
