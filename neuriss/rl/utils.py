import torch

from typing import Tuple


def calculate_gae(
        values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_values: torch.Tensor,
        gamma: float,
        lambd: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate generalized advantage estimator.

    Parameters
    ----------
    values: torch.Tensor,
        values of the states
    rewards: torch.Tensor,
        rewards given by the reward function
    dones: torch.Tensor,
        if this state is the end of the episode
    next_values: torch.Tensor,
        values of the next states
    gamma: float,
        discount factor
    lambd: float,
        lambd factor

    Returns
    -------
    advantages: torch.Tensor,
        advantages
    gaes: torch.Tensor,
        normalized gae
    """
    # calculate TD errors
    deltas = rewards + gamma * next_values * (1 - dones) - values
    # initialize gae
    gaes = torch.empty_like(rewards)

    # calculate gae recursively from behind
    gaes[-1] = deltas[-1]
    for t in reversed(range(rewards.shape[0] - 1)):
        gaes[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * gaes[t + 1]

    return gaes + values, (gaes - gaes.mean()) / (gaes.std(dim=0) + 1e-8)
