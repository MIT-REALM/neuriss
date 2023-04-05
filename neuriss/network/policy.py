import torch
import torch.nn as nn

from typing import Tuple

from .mlp import NormalizedMLP
from .utils import reparameterize, evaluate_log_pi


class StateIndependentPolicy(nn.Module):
    """
    Stochastic policy \pi(a|s)
    """

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            goal_point: torch.Tensor,
            u_eq: torch.Tensor,
            state_std: torch.Tensor,
            ctrl_std: torch.Tensor,
            hidden_units: tuple = (128, 128),
            hidden_activation: nn.Module = nn.Tanh()
    ):
        super().__init__()

        self.net = NormalizedMLP(
            in_dim=state_dim,
            out_dim=action_dim,
            input_mean=goal_point.squeeze(),
            input_std=state_std,
            hidden_layers=hidden_units,
            hidden_activation=hidden_activation,
        )
        self.log_stds = nn.Parameter(torch.zeros(1, action_dim))

        self.u_eq = u_eq
        self.ctrl_std = ctrl_std

    def de_normalize_action(self, u_trans: torch.Tensor) -> torch.Tensor:
        if u_trans.ndim == 1:
            u_trans_ = u_trans.unsqueeze(0)
        else:
            u_trans_ = u_trans
        return (u_trans_ * self.ctrl_std + self.u_eq).view(u_trans.shape)

    def normalize_action(self, u: torch.Tensor) -> torch.Tensor:
        if u.ndim == 1:
            u_ = u.unsqueeze(0)
        else:
            u_ = u
        nonzero_std_dim = torch.nonzero(self.ctrl_std)
        zero_mask = torch.ones(self.ctrl_std.shape[0]).type_as(self.ctrl_std)
        zero_mask[nonzero_std_dim] = 0
        u_trans = (u_ - self.u_eq) / (self.ctrl_std + zero_mask)
        return u_trans

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Get the mean of the stochastic policy
        Parameters
        ----------
        states: torch.Tensor
            input states
        Returns
        -------
        actions: torch.Tensor
            mean of the stochastic policy
        """
        return self.de_normalize_action(self.net(states))

    def sample(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample actions given states
        Parameters
        ----------
        states: torch.Tensor
            input states
        Returns
        -------
        actions: torch.Tensor
            actions to take
        log_pi: torch.Tensor
            log_pi of the actions
        """
        u = self.net(states)
        actions, log_pi = reparameterize(u, self.log_stds)
        return self.de_normalize_action(actions), log_pi

    def evaluate_log_pi(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the log(\pi(a|s)) of the given action
        Parameters
        ----------
        states: torch.Tensor
            states that the actions act in
        actions: torch.Tensor
            actions taken
        Returns
        -------
        log_pi: : torch.Tensor
            log(\pi(a|s))
        """
        return evaluate_log_pi(self.net(states), self.log_stds, self.normalize_action(actions))
