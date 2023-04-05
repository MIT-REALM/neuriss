import torch
import yaml
import os
import cv2
import numpy as np
import argparse
import time

from neuriss.trainer.utils import set_seed, read_settings
from neuriss.env.env import make_env
from neuriss.env.affine import ControlAffineEnv
from neuriss.controller.iss_lyapunov import NeuralISSLyapunov
from neuriss.controller.neural_clf_controller import NeuralCLFController
from neuriss.rl.agents import AGENTS, get_agent


def test_policy(args):
    set_seed(args.seed)
    device = torch.device('cpu')
    try:
        settings = read_settings(args.path)
    except TypeError:
        settings = None
    env = make_env(settings['env'] if args.env is None else args.env, device=device)

    # load hyperparams
    cur_path = os.getcwd()
    if os.path.exists(os.path.join(cur_path, 'neuriss/env/hyperparams', f'{env}.yaml')):
        print('> Using tuned hyper-parameters')
        with open(os.path.join(cur_path, 'neuriss/env/hyperparams', f'{env}.yaml')) as f:
            hyper_params = yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise KeyError(f'Cannot find hyper-parameters for {env}. '
                       f'Please put {env}.yaml in neuriss/env/hyperparams to specify hyper-parameters!')

    if args.path is None:
        # evaluate the nominal controller
        if isinstance(env, ControlAffineEnv):
            if env.use_linearized_controller:
                env.compute_linearized_controller()
            policy = env.u_nominal
            args.path = f'./logs/{args.env}'
            if not os.path.exists('./logs'):
                os.mkdir('./logs')
            if not os.path.exists(args.path):
                os.mkdir(args.path)
            video_path = os.path.join(args.path, 'nominal', 'videos')
            if not os.path.exists(os.path.join(args.path, 'nominal')):
                os.mkdir(os.path.join(args.path, 'nominal'))
            if env.use_linearized_controller:
                env.save_nominal_controller(os.path.join(args.path, 'nominal', f'{args.env}.pkl'))
        else:
            raise KeyError(f'> Env: {args.env} do not have a nominal controller.')
    elif settings['algo'] in AGENTS:
        agent = get_agent(
            algo=settings['algo'],
            state_dims=env.n_dims,
            action_dims=env.n_controls,
            device=device,
            goal_point=env.goal_point,
            u_eq=env.u_eq,
            state_std=env.state_std,
            env=env,
            ctrl_std=torch.ones(sum(env.n_controls), device=device),
        )
        model_path = os.path.join(args.path, 'models')
        if args.iter is not None:
            agent.load(os.path.join(model_path, f'step{args.iter}.pkl'), device=device)
        else:
            # load the last controller
            controller_name = os.listdir(model_path)
            controller_name = [i for i in controller_name if 'step' in i]
            controller_id = sorted([int(i.split('step')[1].split('.')[0]) for i in controller_name])
            agent.load(os.path.join(model_path, f'step{controller_id[-1]}'), device=device)
        policy = agent.act
        video_path = os.path.join(args.path, 'videos')
    elif settings['algo'] == 'clf':
        lyapunov = NeuralCLFController(
            state_dim=sum(env.n_dims),
            action_dim=sum(env.n_controls),
            device=device,
            goal_point=env.goal_point,
            u_eq=env.u_eq,
            state_std=env.state_std,
            env=env,
            residue=hyper_params['residue']
        )
        if args.iter is not None:
            lyapunov.load(os.path.join(args.path, 'models', f'iter{args.iter}'), device=device)
        else:
            iter_name = os.listdir(os.path.join(args.path, 'models'))
            if 'best' in iter_name:
                lyapunov.load(os.path.join(os.path.join(args.path, 'models'), f'best'), device=device)
            else:
                iter_name = [i for i in iter_name if 'iter' in i]
                iter_id = sorted([int(i.split('iter')[1].split('.')[0]) for i in iter_name])
                lyapunov.load(os.path.join(os.path.join(args.path, 'models'), f'iter{iter_id[-1]}'), device=device)
        policy = lyapunov.act
        video_path = os.path.join(args.path, 'videos')
    else:
        # evaluate the leaned controller
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
            residue=hyper_params['residue']
        )
        if args.iter is not None:
            lyapunov.load(os.path.join(args.path, 'models', f'iter{args.iter}'), device=device)
        else:
            iter_name = os.listdir(os.path.join(args.path, 'models'))
            if 'best' in iter_name:
                lyapunov.load(os.path.join(os.path.join(args.path, 'models'), f'best'), device=device)
            else:
                iter_name = [i for i in iter_name if 'iter' in i]
                iter_id = sorted([int(i.split('iter')[1].split('.')[0]) for i in iter_name])
                lyapunov.load(os.path.join(os.path.join(args.path, 'models'), f'iter{iter_id[-1]}'), device=device)
        policy = lyapunov.act
        video_path = os.path.join(args.path, 'videos')

    # mkdir for the video and the figures
    if not args.no_video and not os.path.exists(video_path):
        os.mkdir(video_path)

    # set up video writer
    out = None
    if not args.no_video:
        env.reset()
        data = env.render()
        out = cv2.VideoWriter(
            os.path.join(video_path, f'reward{0.0}.mov'),
            cv2.VideoWriter_fourcc(*'mp4v'),
            25,
            (data.shape[1], data.shape[0])
        )

    # evaluate policy
    rewards = []
    lengths = []
    print('> Processing...')
    start_time = time.time()
    for i_epi in range(args.epi):
        epi_length = 0
        epi_reward = 0
        epi_states = []
        epi_next_states = []
        epi_ctrls = []
        state = env.reset()
        t = 0
        while True:
            epi_states.append(state.unsqueeze(0))
            action = policy(state)
            epi_ctrls.append(action)
            next_state, reward, done, _ = env.step(action)
            epi_length += 1
            epi_reward += reward
            epi_next_states.append(next_state.unsqueeze(0))
            state = next_state
            t += 1

            if not args.no_video:
                if t % 3 == 0:
                    out.write(env.render())
            if done:
                print(f'epi: {i_epi}, reward: {epi_reward:.2f}, length: {epi_length}')
                rewards.append(epi_reward)
                lengths.append(epi_length)
                break

    # release the video
    if not args.no_video:
        out.release()
        os.rename(os.path.join(video_path, f'reward{0.0}.mov'),
                  os.path.join(video_path, f'reward{np.mean(rewards):.2f}.mov'))

    # print evaluation results
    print(f'average reward: {np.mean(rewards):.2f}, average length: {np.mean(lengths):.2f}')
    print(f'> Done in {time.time() - start_time:.0f}s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # custom
    parser.add_argument('--env', type=str, default=None)
    parser.add_argument('--path', type=str, default=None)
    parser.add_argument('--iter', type=int, default=None)
    parser.add_argument('--epi', type=int, default=20)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--no-video', action='store_true', default=False)

    # default
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    test_policy(args)
