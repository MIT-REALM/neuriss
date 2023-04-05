import torch
import torch.optim as optim
import numpy as np
import os
import shutil

from tqdm import tqdm
from typing import Dict, Tuple, Callable
from torch.utils.tensorboard import SummaryWriter

from neuriss.controller.neural_clf_controller import NeuralCLFController
from neuriss.env.base import Env


class CLFTrainer:
    """
    Trainer for Neural CLF.

    References
    ----------
    [1] Charles Dawson, Zengyi Qin, Sicun Gao, and Chuchu Fan.
    Safe nonlinear control using robust neural lyapunov-barrier functions.
    Conference on Robot Learning, 2021.
    """

    def __init__(
            self,
            controller: NeuralCLFController,
            env: Env,
            writer: SummaryWriter,
            save_path: str,
            controller_ref: Callable = None,
            hyper_params: dict = None
    ):
        self.controller = controller
        self.controller_ref = controller_ref
        self.env = env
        self.writer = writer
        self.optim_lyapunov = optim.Adam(self.controller.lyapunov.parameters(), lr=3e-4, weight_decay=1e-3)
        self.optim_controller = optim.Adam(self.controller.controller.parameters(), lr=3e-4, weight_decay=1e-3)
        self.hyper_params = hyper_params
        self.save_path = save_path
        self.num_eval_episodes = 5

        # set default hyper-parameters
        if hyper_params is None:
            self.hyper_params = {
                'loss_coefs': {
                    'goal': 10.,
                    'decrease': 1.,
                    'condition': 1.,
                    'control': 1e-3,
                },
                'loss_eps': {
                    'decrease': 1.0,
                }
            }

    def train(
            self,
            n_iter: int,
            batch_size: int,
            eval_interval: int
    ):
        """
        Train the CLF controller

        Parameters
        ----------
        n_iter: int
            number of iterations
        batch_size: int
            batch size
        eval_interval: int
        """
        # jointly train the lyapunov function and the controller
        best_reward = -1000000.
        for i_iter in tqdm(range(1, n_iter + 1), ncols=80):
            # sample states
            states = self.env.sample_states(batch_size)

            loss = {}
            goal_mask = torch.ones(states.shape[0]).type_as(states)
            for mask in self.env.goal_mask(states):
                goal_mask = torch.logical_and(goal_mask, mask)
            loss.update(self.boundary_loss(
                states,
                goal_mask
            ))
            decent_loss, evaluation = self.descent_loss(states, self.controller_ref(states))
            loss.update(decent_loss)

            total_loss = torch.tensor(0.0).type_as(states)
            for value in loss.values():
                total_loss += value

            self.optim_lyapunov.zero_grad()
            self.optim_controller.zero_grad()
            total_loss.backward()
            self.optim_lyapunov.step()
            self.optim_controller.step()

            for loss_name in loss.keys():
                self.writer.add_scalar(f'loss/{loss_name}', loss[loss_name].item(), i_iter)
            self.writer.add_scalar(f'loss/total loss', total_loss.item(), i_iter)

            for eval_name in evaluation.keys():
                self.writer.add_scalar(f'eval/{eval_name}', evaluation[eval_name].item(), i_iter)

            if i_iter % eval_interval == 0:
                tqdm.write(f'iter: {i_iter}, loss: {total_loss.item():.2e}')
                os.mkdir(os.path.join(self.save_path, f'iter{i_iter}'))
                self.controller.save(os.path.join(self.save_path, f'iter{i_iter}'))
                eval_reward = self.evaluate(i_iter)
                if eval_reward > best_reward:
                    best_reward = eval_reward
                    if os.path.exists(os.path.join(self.save_path, 'best')):
                        shutil.rmtree(os.path.join(self.save_path, 'best'))
                    os.mkdir(os.path.join(self.save_path, f'best'))
                    self.controller.save(os.path.join(self.save_path, f'best'))
                    with open(os.path.join(
                            os.path.join(self.save_path, f'best', f'reward{best_reward:.0f}.txt')), 'w') as f:
                        f.write(f'{best_reward}')
        print(f'Best reward: {best_reward}')

    def boundary_loss(
            self,
            x: torch.Tensor,
            goal_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the CLF boundary loss according to the following requirements:
            1.) CLF should be minimized on the goal point
            2.) V >= 0 in all region
        Parameters
        ----------
        x: torch.Tensor
            input states (normalized)
        goal_mask: torch.Tensor
            same dimension as x, 1 is in the goal region and 0 is not
        Returns
        -------
        loss: Dict[str, torch.Tensor]
            dict of loss terms, including
                1.) CLF goal term
                2.) CLF positive term
        """
        loss = {}
        V = self.controller.V(x)

        # CLF should be minimized on the goal point
        V_goal_1 = self.controller.V(self.env.sample_goal(batch_size=x.shape[0] // 5))
        V_goal_2 = V[goal_mask]
        V_goal = torch.cat([V_goal_1, V_goal_2], dim=0)
        goal_term = (V_goal ** 2).mean()
        loss['CLBF goal term'] = self.hyper_params['loss_coefs']['goal'] * goal_term

        return loss

    def descent_loss(
            self,
            x: torch.Tensor,
            u_ref: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Compute the CLF decent loss. The CLF decrease condition requires that V is decreasing
        everywhere. We'll encourage this in two ways:
            1.) Compute the CLF decrease at each point by linearizing
            2.) Compute the CLF decrease at each point by simulating
        Parameters
        ----------
        x: torch.Tensor
            input states
        u_ref: torch.Tensor
            control signal of the reference nominal controller
        Returns
        -------
        loss: Dict[str, torch.Tensor]
            dict of loss terms, including:
                1.) QP relaxation
                2.) CLF descent term (linearized)
                3.) CLF descent term (simulated)
        """
        loss = {}
        evaluation = {}
        x.requires_grad = True
        V, JV = self.controller.V_with_Jacobian(x)
        u = self.controller.u(x)
        xdot = self.env.closed_loop_dynamics(x, u)

        # Now compute the decrease using linearization
        eps = self.hyper_params['loss_eps']['decrease']
        clbf_descent_term_lin = torch.tensor(0.0).type_as(x)
        # Get the current value of the CLF and its Lie derivatives
        Vdot = torch.bmm(JV, xdot.unsqueeze(-1)).squeeze(1)
        violation = torch.relu(eps + Vdot + self.controller.clf_lambda * V)
        clbf_descent_term_lin += violation.mean()
        loss['CLBF descent term (linearized)'] = self.hyper_params['loss_coefs']['decrease'] * clbf_descent_term_lin

        # Now compute the decrease using simulation
        clbf_descent_term_sim = torch.tensor(0.0).type_as(x)
        x_next = x + xdot * self.env.dt
        V_next = self.controller.V(x_next)
        violation = torch.relu(eps + ((V_next - V) / self.env.dt) + self.controller.clf_lambda * V)
        clbf_descent_term_sim += violation.mean()
        loss['CLBF descent term (simulated)'] = self.hyper_params['loss_coefs']['decrease'] * clbf_descent_term_sim
        with torch.no_grad():
            real_violation = torch.relu((V_next - V) / self.env.dt + self.controller.clf_lambda * V)
        evaluation['CLF descent violation ratio'] = torch.count_nonzero(real_violation) / real_violation.shape[0]

        # compute the loss of deviation from the reference controller
        loss['Controller deviation'] = self.hyper_params['loss_coefs']['control'] * ((u - u_ref) ** 2).mean()

        return loss, evaluation

    def evaluate(self, i_iter: int):
        returns = []
        self.env.test()

        for _ in range(self.num_eval_episodes):
            state = self.env.reset()
            episode_return = 0.0
            t = 0

            while t < self.env.max_episode_steps:
                action = self.controller.act(state)
                state, reward, done, _ = self.env.step(action)
                episode_return += reward
                t += 1
                if done:
                    break

            returns.append(episode_return)

        self.writer.add_scalar('eval/reward', np.mean(returns), i_iter)
        tqdm.write(f'Num steps: {i_iter:<5}, '
                   f'Return: {np.mean(returns):<5.1f}, '
                   f'Min/Max Return: {np.min(returns):<5.1f}/{np.max(returns):<5.1f}')
        self.env.train()
        return np.mean(returns)
