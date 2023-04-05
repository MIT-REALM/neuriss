import torch
import numpy as np
import os
import torch.nn as nn
import shutil

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from typing import Callable, Tuple
from torch.nn.utils import clip_grad_norm_

from neuriss.controller.iss_lyapunov import NeuralISSLyapunov
from neuriss.env.base import Env


class ISSLyapunovTrainer:
    """Trainer for NeurISS."""

    def __init__(
            self,
            env: Env,
            lyapunov_func: NeuralISSLyapunov,
            writer: SummaryWriter,
            save_path: str,
            controller_ref: Callable = None,
            hyper_params: dict = None
    ):
        self.env = env
        self.lyapunov_func = lyapunov_func
        self.writer = writer
        self.save_path = save_path
        self.controller_ref = controller_ref
        self.max_grad_norm = 1.
        self.num_eval_episodes = 5

        # set up optimizers
        self.optim_lyapunov = []
        self.optim_chi = []
        self.optim_u = []
        for i_func in range(self.lyapunov_func.n_lyapunov):
            self.optim_lyapunov.append(Adam(self.lyapunov_func.lyapunov_funcs[i_func].parameters(),
                                            lr=3e-4, weight_decay=1e-3))
        self.optim_chi = [Adam(self.lyapunov_func.chis.parameters(), lr=1e-3)]
        for i_ctrl in range(self.lyapunov_func.n_controller):
            self.optim_u.append(Adam(self.lyapunov_func.controllers[i_ctrl].parameters(), lr=5e-4, weight_decay=1e-3))
        self.optim = self.optim_lyapunov + self.optim_chi + self.optim_u

        if hyper_params is None:
            self.hyper_params = {
                'loss_coefs': {
                    'goal': 100.,
                    'decrease': 1.,
                    'condition': 0.01,
                    'control': 1.,
                },
                'loss_eps': {
                    'decrease': 1.0,
                }
            }
        else:
            self.hyper_params = hyper_params

    def _adjust_weight(self, i_iter, n_iter):
        if i_iter % (n_iter // 3) == 0:
            self.hyper_params['loss_coefs']['control'] *= 0.1

    def _initial_steps(self, n_iter: int, batch_size: int):
        loss_fun = nn.MSELoss()
        for i_iter in tqdm(range(1, n_iter + 1), ncols=80):
            states = self.env.sample_states(batch_size)
            u = self.lyapunov_func.u(states)
            u_ref = self.env.u_nominal(states)
            loss = loss_fun(u, u_ref)
            for i_optim in range(len(self.optim_u)):
                self.optim_u[i_optim].zero_grad()
            loss.backward()
            for i_optim in range(len(self.optim_u)):
                self.optim_u[i_optim].step()
            if i_iter % (n_iter // 5) == 0:
                tqdm.write(f'iter: {i_iter}, loss: {loss.item():.2e}')
        self.evaluate(n_iter)
        print(f'> Initial controller training finished')

        for i_iter in tqdm(range(1, n_iter + 1), ncols=80):
            states = self.env.sample_states(batch_size)
            loss, _ = self.calculate_loss(states, ['init' for _ in range(self.lyapunov_func.n_systems)])
            total_loss = torch.tensor(0.0).type_as(states)
            for key, value in loss.items():
                total_loss += value
            for i_optim in range(len(self.optim_lyapunov)):
                self.optim_lyapunov[i_optim].zero_grad()
            total_loss.backward()
            for i_optim in range(len(self.optim_lyapunov)):
                self.optim_lyapunov[i_optim].step()
            if i_iter % (n_iter // 5) == 0:
                tqdm.write(f'iter: {i_iter}, loss: {total_loss.item():.2e}')
        print(f'> Initial ISS-Lyapunov training finished')

    def train(
            self,
            n_iter: int,
            batch_size: int,
            init_iter: int,
            eval_interval: int,
            adjust_weight: bool = False,
    ):
        self._initial_steps(n_iter=init_iter, batch_size=batch_size)
        best_reward = -1000000.
        for i_iter in tqdm(range(1, n_iter + 1), ncols=80):
            if adjust_weight:
                self._adjust_weight(i_iter, n_iter)
            states = self.env.sample_states(batch_size)

            loss, evaluation = self.calculate_loss(
                states, status=['train' for _ in range(self.lyapunov_func.n_systems)])

            total_loss = torch.tensor(0.0).type_as(states)
            for key, value in loss.items():
                total_loss += value

            for i_optim in range(len(self.optim)):
                self.optim[i_optim].zero_grad()
            total_loss.backward()
            for i_optim in range(len(self.optim)):
                self.optim[i_optim].step()

            for loss_name in loss.keys():
                self.writer.add_scalar(f'loss/{loss_name}', loss[loss_name].item(), i_iter)
            for eval_name in evaluation.keys():
                self.writer.add_scalar(f'eval/{eval_name}', evaluation[eval_name].item(), i_iter)
            self.writer.add_scalar(f'loss/total loss', total_loss.item(), i_iter)

            if i_iter % eval_interval == 0:
                tqdm.write(f'iter: {i_iter}, loss: {total_loss.item():.2e}')
                os.mkdir(os.path.join(self.save_path, f'iter{i_iter}'))
                self.lyapunov_func.save(os.path.join(self.save_path, f'iter{i_iter}'))
                eval_reward = self.evaluate(i_iter)
                if eval_reward > best_reward:
                    best_reward = eval_reward
                    if os.path.exists(os.path.join(self.save_path, 'best')):
                        shutil.rmtree(os.path.join(self.save_path, 'best'))
                    os.mkdir(os.path.join(self.save_path, f'best'))
                    self.lyapunov_func.save(os.path.join(self.save_path, f'best'))
                    with open(os.path.join(
                            os.path.join(self.save_path, f'best', f'reward{best_reward:.0f}.txt')), 'w') as f:
                        f.write(f'{best_reward}')
        print(f'Best reward: {best_reward}')

    def calculate_loss(self, states, status: list) -> Tuple[dict, dict]:
        loss = {}
        evaluation = {}
        Vs, Js = self.lyapunov_func.lyapunov_with_jacobian(states)
        u = self.lyapunov_func.u(states)
        xdot = self.env.closed_loop_dynamics(states, u)
        x_next = states + xdot * self.env.dt
        Vs_next = self.lyapunov_func.lyapunov_values(x_next)
        goal_masks = self.env.goal_mask(states)
        goal_states = self.env.sample_goal(batch_size=states.shape[0] // 5)
        Vs_goal = self.lyapunov_func.lyapunov_values(goal_states)

        # V = 0 at goal
        for i_system in range(self.env.n_systems):
            V_goal_1 = Vs[i_system][goal_masks[i_system]]
            V_goal_2 = Vs_goal[i_system]
            V_goal = torch.cat([V_goal_1, V_goal_2])
            goal_term = (torch.abs(V_goal)).mean()
            loss[f'system {i_system} goal term'] = self.hyper_params['loss_coefs']['goal'] * goal_term

        # update decent term iteratively
        eps = self.hyper_params['loss_eps']['decrease']
        system_xdot = torch.split(xdot, list(self.env.n_dims), dim=1)
        for i_system in range(self.lyapunov_func.n_systems):
            # evaluation
            with torch.no_grad():
                condition_true = torch.relu(
                    Vs[i_system]
                    - self.lyapunov_func.gains(states, i_system)
                )
                condition_true_mask = torch.nonzero(condition_true)[:, 0]
                evaluation[f'system {i_system} condition activation'] = \
                    torch.count_nonzero(condition_true) / states.shape[0]

                decent_term_lin = torch.relu(
                    torch.bmm(Js[i_system][condition_true_mask],
                              system_xdot[i_system][condition_true_mask].unsqueeze(-1)).squeeze(1)
                    + self.lyapunov_func.lambdas[i_system] * Vs[i_system][condition_true_mask]
                )
                evaluation[f'system {i_system} decent violation lin'] = \
                    torch.count_nonzero(decent_term_lin) / torch.count_nonzero(condition_true)

                decent_term_sim = torch.relu(
                    ((Vs_next[i_system][condition_true_mask] - Vs[i_system][condition_true_mask]) / self.env.dt)
                    + self.lyapunov_func.lambdas[i_system] * Vs[i_system][condition_true_mask]
                )
                evaluation[f'system {i_system} decent violation sim'] = \
                    torch.count_nonzero(decent_term_sim) / torch.count_nonzero(condition_true)

                evaluation[f'system {i_system} chi coef'] = self.lyapunov_func.chis.a[i_system]

            # calculate loss
            if status[i_system] == 'init':
                decent_term_lin = torch.relu(
                    torch.bmm(Js[i_system], system_xdot[i_system].unsqueeze(-1)).squeeze(1)
                    + self.lyapunov_func.lambdas[i_system] * Vs[i_system]
                    + eps
                )
                if not torch.isnan(decent_term_lin.mean()):
                    loss[f'system {i_system} decent term lin'] = \
                        self.hyper_params['loss_coefs']['decrease'] * decent_term_lin.mean()
                decent_term_sim = torch.relu(
                    ((Vs_next[i_system] - Vs[i_system]) / self.env.dt)
                    + self.lyapunov_func.lambdas[i_system] * Vs[i_system]
                    + eps
                )
                if not torch.isnan(decent_term_sim.mean()):
                    loss[f'system {i_system} decent term sim'] = \
                        self.hyper_params['loss_coefs']['decrease'] * decent_term_sim.mean()
            else:
                condition_term = torch.relu(
                    Vs[i_system]
                    - self.lyapunov_func.gains(states, i_system)
                    + eps
                )
                if not torch.isnan(condition_term.mean()):
                    loss[f'system {i_system} condition term'] = \
                        self.hyper_params['loss_coefs']['condition'] * condition_term.mean()
                decent_term_lin = torch.relu(
                    torch.bmm(Js[i_system],
                              system_xdot[i_system].unsqueeze(-1)).squeeze(1)
                    + self.lyapunov_func.lambdas[i_system] * Vs[i_system]
                    + eps
                )
                if not torch.isnan(decent_term_lin.mean()):
                    loss[f'system {i_system} decent term lin'] = \
                        self.hyper_params['loss_coefs']['decrease'] * decent_term_lin.mean()
                decent_term_sim = torch.relu(
                    ((Vs_next[i_system] - Vs[i_system]) / self.env.dt)
                    + self.lyapunov_func.lambdas[i_system] * Vs[i_system]
                    + eps
                )
                if not torch.isnan(decent_term_sim.mean()):
                    loss[f'system {i_system} decent term sim'] = \
                        self.hyper_params['loss_coefs']['decrease'] * decent_term_sim.mean()

        # controller deviation
        if self.controller_ref is not None:
            u_ref = self.controller_ref(states)
            loss['Controller deviation'] = self.hyper_params['loss_coefs']['control'] * ((u - u_ref) ** 2).mean()
        else:
            loss['Controller deviation'] = self.hyper_params['loss_coefs']['control'] * (u ** 2).mean()

        return loss, evaluation

    def evaluate(self, i_iter: int):
        returns = []
        lengths = []
        self.env.test()

        for _ in range(self.num_eval_episodes):
            state = self.env.reset()
            episode_return = 0.0
            t = 0

            while t < self.env.max_episode_steps:
                action = self.lyapunov_func.act(state)
                state, reward, done, _ = self.env.step(action)
                episode_return += reward
                t += 1
                if done:
                    break

            returns.append(episode_return)
            lengths.append(t)

        self.writer.add_scalar('eval/reward', np.mean(returns), i_iter)
        tqdm.write(f'Num steps: {i_iter:<5}, '
                   f'Return: {np.mean(returns):<5.1f}, '
                   f'Min/Max Return: {np.min(returns):<5.1f}/{np.max(returns):<5.1f}, '
                   f'Length: {np.mean(lengths):<5.1f}')
        self.env.train()

        return np.mean(returns)
