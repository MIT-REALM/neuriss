import torch
import numpy as np

from typing import Tuple


class RolloutBuffer:
    """
    Rollout buffer that often used in training RL agents.
    """

    def __init__(
            self,
            buffer_size: int,
            state_dim: int,
            action_dim: int,
            device: torch.device,
            mix: int = 1
    ):
        self._n = 0
        self._p = 0
        self.mix = mix
        self.device = device
        self.buffer_size = buffer_size
        self.total_size = mix * buffer_size

        self.states = torch.empty(
            (self.total_size, state_dim), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (self.total_size, action_dim), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.log_pis = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (self.total_size, state_dim), dtype=torch.float, device=device)

    def append(
            self,
            state: torch.Tensor,
            action: torch.Tensor,
            reward: float,
            done: bool,
            log_pi: float,
            next_state: torch.Tensor
    ):
        """
        Save a transition in the buffer.
        """
        if state.ndim == 2:
            state = state.squeeze(0)
        if action.ndim == 2:
            action = action.squeeze(0)

        self.states[self._p].copy_(state)
        self.actions[self._p].copy_(action)
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.log_pis[self._p] = float(log_pi)
        self.next_states[self._p].copy_(next_state)

        self._p = (self._p + 1) % self.total_size
        self._n = min(self._n + 1, self.total_size)

    def get(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get all data in the buffer.

        Returns
        -------
        states: torch.Tensor
        actions: torch.Tensor
        rewards: torch.Tensor
        dones: torch.Tensor
        log_pis: torch.Tensor
        next_states: torch.Tensor
        """
        assert self._p % self.buffer_size == 0
        start = (self._p - self.buffer_size) % self.total_size
        idxes = slice(start, start + self.buffer_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )

    def sample(
            self,
            batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample data from the buffer.

        Parameters
        ----------
        batch_size: int
            batch size

        Returns
        -------
        states: torch.Tensor
        actions: torch.Tensor
        rewards: torch.Tensor
        dones: torch.Tensor
        log_pis: torch.Tensor
        next_states: torch.Tensor
        """
        assert self._p % self.buffer_size == 0
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )
