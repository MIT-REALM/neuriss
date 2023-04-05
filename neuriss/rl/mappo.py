import torch
import torch.nn as nn
import numpy as np
import os

from torch.optim import Adam
from typing import Tuple

from .ppo import PPO
from ..env.base import Env
from ..network.policy import StateIndependentPolicy


class MAPPO(PPO):
    """
    Multi-agent PPO

    References
    ----------
    [1] Chao Yu, Akash Velu, Eugene Vinitsky, Yu Wang, Alexandre Bayen, and Yi Wu.
    The surprising effectiveness of ppo in cooperative, multi-agent games.
    arXiv preprint arXiv:2103.01955, 2021.
    """

    def __init__(
            self,
            state_dims: tuple,
            action_dims: tuple,
            device: torch.device,
            goal_point: torch.Tensor,
            u_eq: torch.Tensor,
            state_std: torch.Tensor,
            ctrl_std: torch.Tensor,
            env: Env,
            gamma: float = 0.995,
            rollout_length: int = 4096,
            mix_buffer: int = 1,
            lr_actor: float = 3e-4,
            lr_critic: float = 3e-4,
            units_actor: tuple = (128, 128),
            units_critic: tuple = (128, 128),
            epoch_ppo: int = 20,
            clip_eps: float = 0.2,
            lambd: float = 0.97,
            coef_ent: float = 0.0,
            max_grad_norm: float = 10.0
    ):
        super().__init__(
            sum(state_dims), sum(action_dims), device, goal_point, u_eq, state_std, ctrl_std, gamma, rollout_length,
            mix_buffer, lr_actor, lr_critic, units_actor, units_critic, epoch_ppo, clip_eps, lambd, coef_ent,
            max_grad_norm
        )
        self.n_systems = len(state_dims)
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.n_actor = len(env.share_ctrl)

        # actor
        self.optim_actor = None
        self.actor = None
        self.input_dims = torch.zeros(self.n_actor, dtype=torch.int, device=device)
        self.output_dims = torch.zeros(self.n_actor, dtype=torch.int, device=device)
        self.actors = []
        self.optim_actor = []
        for i in range(self.n_actor):
            state_dims_torch = torch.tensor(self.state_dims, device=device)
            assert len(set(state_dims_torch[env.share_ctrl[i]].cpu().numpy())) == 1
            self.input_dims[i] = int(state_dims_torch[env.share_ctrl[i]][0])
            ctrl_dims_torch = torch.tensor(self.action_dims, device=device)
            assert len(set(ctrl_dims_torch[env.share_ctrl[i]].cpu().numpy())) == 1
            self.output_dims[i] = int(ctrl_dims_torch[env.share_ctrl[i]][0])
            self.actors.append(StateIndependentPolicy(
                state_dim=int(self.input_dims[i]),
                action_dim=int(self.output_dims[i]),
                goal_point=torch.split(goal_point, list(state_dims))[0],
                u_eq=torch.split(u_eq, list(action_dims))[0],
                state_std=torch.split(state_std, list(state_dims))[0],
                ctrl_std=torch.split(ctrl_std, list(action_dims))[0],
                hidden_units=units_actor,
                hidden_activation=nn.Tanh(),
            ).to(device))
            self.optim_actor.append(Adam(self.actors[i].parameters(), lr=lr_actor))

        # establish the mapping from subsystem idx to controller idx
        self._mapping_ctrl = torch.zeros(self.n_systems, dtype=torch.int, device=device)
        for i, share_id in enumerate(env.share_ctrl):
            self._mapping_ctrl[share_id] = i

    def mapping_ctrl(self, idx: int) -> int:
        return self._mapping_ctrl[idx]

    def act(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        system_x = torch.split(x, list(self.state_dims), dim=1)
        actions = []
        for i in range(self.n_systems):
            with torch.no_grad():
                actions.append(self.actors[self.mapping_ctrl(i)].forward(system_x[i]))
        return torch.cat(actions, dim=1)

    def save(self, path: str):
        for i in range(self.n_actor):
            torch.save(self.actors[i].state_dict(), os.path.join(path, f'actor_{i}.pkl'))

    def load(self, path: str, device: torch.device):
        for i in range(self.n_actor):
            self.actors[i].load_state_dict(torch.load(os.path.join(path, f'actor_{i}.pkl'), map_location=device))

    def step(self, env: Env, state: torch.Tensor, t: int, step: int) -> Tuple[torch.Tensor, int]:
        t += 1

        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float).to(self.device)

        system_state = torch.split(state, list(self.state_dims))

        action, log_pi = [], []
        for i in range(self.n_systems):
            with torch.no_grad():
                action_i, log_pi_i = self.actors[self.mapping_ctrl(i)].sample(system_state[i].to(self.device))
            action.append(action_i)
            log_pi.append(log_pi_i)
        action = torch.cat(action, dim=1)
        log_pi = torch.sum(torch.cat(log_pi, dim=1), dim=1)

        next_state, reward, done, _ = env.step(action.cpu())
        mask = True if t == env.max_episode_steps else done

        self.buffer.append(state, action, reward, mask, log_pi, next_state)

        if done or t == env.max_episode_steps:
            t = 0
            next_state = env.reset()

        return next_state, t

    def update_actor(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            log_pis_old: torch.Tensor,
            gaes: torch.Tensor
    ):
        """
        Update the actor for one step.
        """
        system_states = torch.split(states, list(self.state_dims), dim=1)
        system_actions = torch.split(actions, list(self.action_dims), dim=1)

        log_pis = []
        for i in range(self.n_systems):
            log_pis.append(self.actors[self.mapping_ctrl(i)].evaluate_log_pi(system_states[i], system_actions[i]))
        log_pis = torch.sum(torch.cat(log_pis, dim=1), dim=1).unsqueeze(1)

        entropy = -log_pis.mean()

        ratios = (log_pis - log_pis_old).exp_()
        loss_actor1 = -ratios * gaes
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * gaes
        loss_actor = torch.max(loss_actor1, loss_actor2).mean()

        for i in range(self.n_actor):
            self.optim_actor[i].zero_grad()
        (loss_actor - self.coef_ent * entropy).backward(retain_graph=False)
        for i in range(self.n_actor):
            nn.utils.clip_grad_norm_(self.actors[i].parameters(), self.max_grad_norm)
            self.optim_actor[i].step()
