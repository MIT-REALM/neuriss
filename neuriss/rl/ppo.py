import torch
import torch.nn as nn
import numpy as np
import os

from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple

from .base import Agent
from .utils import calculate_gae
from ..env.base import Env
from .buffer import RolloutBuffer
from ..network.policy import StateIndependentPolicy
from ..network.mlp import NormalizedMLP


class PPO(Agent):
    """
    PPO: Proximal policy optimization

    References
    ----------
    [1]John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov.
    Proximal policy optimization algorithms.
    arXiv preprint arXiv:1707.06347, 2017.
    """

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            device: torch.device,
            goal_point: torch.Tensor,
            u_eq: torch.Tensor,
            state_std: torch.Tensor,
            ctrl_std: torch.Tensor,
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
        super().__init__(state_dim, action_dim, gamma, device)

        # rollout buffer
        self.buffer = RolloutBuffer(
            buffer_size=rollout_length,
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            mix=mix_buffer
        )

        # actor
        self.actor = StateIndependentPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            goal_point=goal_point,
            u_eq=u_eq,
            state_std=state_std,
            ctrl_std=ctrl_std,
            hidden_units=units_actor,
            hidden_activation=nn.Tanh(),
        ).to(device)

        # critic
        self.critic = NormalizedMLP(
            in_dim=state_dim,
            out_dim=1,
            input_mean=goal_point.squeeze(),
            input_std=state_std,
            hidden_layers=units_critic,
            hidden_activation=nn.Tanh(),
        ).to(device)

        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic)

        self.learning_steps_ppo = 0
        self.rollout_length = rollout_length
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps
        self.lambd = lambd
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm

    def act(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            return self.actor.forward(x)

    def save(self, path: str):
        torch.save(self.actor.state_dict(), os.path.join(path, 'actor.pkl'))

    def load(self, path: str, device: torch.device):
        self.actor.load_state_dict(torch.load(os.path.join(path, 'actor.pkl'), map_location=device))

    def set_controller(self, controller: nn.Module):
        self.actor.net.load_state_dict(controller.state_dict())

    def is_update(self, step: int) -> bool:
        return step % self.rollout_length == 0 and step >= self.rollout_length

    def step(self, env: Env, state: torch.Tensor, t: int, step: int) -> Tuple[torch.Tensor, int]:
        t += 1

        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float).to(self.device)

        with torch.no_grad():
            action, log_pi = self.actor.sample(state.to(self.device))
        next_state, reward, done, _ = env.step(action.cpu())
        mask = True if t == env.max_episode_steps else done

        self.buffer.append(state, action, reward, mask, log_pi[0], next_state)

        if done or t == env.max_episode_steps:
            t = 0
            next_state = env.reset()

        return next_state, t

    def update(self, writer: SummaryWriter = None):
        self.learning_steps += 1
        states, actions, rewards, dones, log_pis, next_states = \
            self.buffer.get()
        self.update_ppo(states, actions, rewards, dones, log_pis, next_states)

    def update_ppo(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            dones: torch.Tensor,
            log_pis: torch.Tensor,
            next_states: torch.Tensor
    ):
        """
        Update PPO's actor and critic for some steps.

        Parameters
        ----------
        states: torch.Tensor,
            sampled states
        actions: torch.Tensor,
            sampled actions according to the states
        rewards: torch.Tensor,
            rewards of the s-a pairs
        dones: torch.Tensor,
            whether is the end of the episode
        log_pis: torch.Tensor,
            log(\pi(a|s)) of the actions
        next_states: torch.Tensor,
            next states give s-a pairs
        """
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)

        targets, gaes = calculate_gae(
            values, rewards, dones, next_values, self.gamma, self.lambd)

        for _ in range(self.epoch_ppo):
            self.learning_steps_ppo += 1
            self.update_critic(states, targets)
            self.update_actor(states, actions, log_pis, gaes)

    def update_critic(self, states: torch.Tensor, targets: torch.Tensor):
        """
        Update the critic for one step.
        """
        loss_critic = (self.critic(states) - targets).pow_(2).mean()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

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
        log_pis = self.actor.evaluate_log_pi(states, actions)
        entropy = -log_pis.mean()

        ratios = (log_pis - log_pis_old).exp_()
        loss_actor1 = -ratios * gaes
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * gaes
        loss_actor = torch.max(loss_actor1, loss_actor2).mean()

        self.optim_actor.zero_grad()
        (loss_actor - self.coef_ent * entropy).backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()
