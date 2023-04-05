import torch

from .base import Agent
from .ppo import PPO
from .mappo import MAPPO
from .lyppo import LYPPO
from ..env.base import Env

AGENTS = {
    'ppo',
    'mappo',
    'lyppo'
}


def get_agent(
        algo: str,
        state_dims: tuple,
        action_dims: tuple,
        device: torch.device,
        goal_point: torch.Tensor,
        u_eq: torch.Tensor,
        state_std: torch.Tensor,
        ctrl_std: torch.Tensor,
        env: Env
) -> Agent:
    if algo == 'ppo':
        return PPO(sum(state_dims), sum(action_dims), device,
                   goal_point.to(device), u_eq.to(device), state_std.to(device), ctrl_std.to(device))
    elif algo == 'mappo':
        return MAPPO(state_dims, action_dims, device,
                     goal_point.to(device), u_eq.to(device), state_std.to(device), ctrl_std.to(device), env)
    elif algo == 'lyppo':
        return LYPPO(sum(state_dims), sum(action_dims), device,
                     goal_point.to(device), u_eq.to(device), state_std.to(device), ctrl_std.to(device), env)
    else:
        raise NotImplementedError(f'{algo} has not been implemented')
