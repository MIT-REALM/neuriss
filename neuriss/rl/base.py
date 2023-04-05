import torch
import torch.nn as nn
import numpy as np

from abc import ABC, abstractmethod
from typing import Tuple
from neuriss.env.base import Env
from torch.utils.tensorboard import SummaryWriter


class Agent(ABC):

    def __init__(self, state_dim: int, action_dim: int, gamma: float, device: torch.device):
        super().__init__()
        self.learning_steps = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma

    @abstractmethod
    def act(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate action.

        Parameters
        ----------
        x: torch.Tensor,
            states

        Returns
        -------
        action: torch.Tensor,
            actions
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """
        Save the model.

        Parameters
        ----------
        path: str,
            path to save the model
        """
        pass

    @abstractmethod
    def load(self, path: str, device: torch.device):
        """
        Load the pre-trained model.

        Parameters
        ----------
        path: str,
            path to load the model
        device: torch.device,
            device to use
        """
        pass

    @abstractmethod
    def set_controller(self, controller: nn.Module):
        """
        Set the weights of the controller to be a pre-trained weight.

        Parameters
        ----------
        controller: nn.Module,
            pre-trained weights of the controller
        """
        pass

    @abstractmethod
    def is_update(self, step: int):
        """
        Whether the time is for update.

        Parameters
        ----------
        step: int,
            current training step
        """
        pass

    @abstractmethod
    def update(self, writer: SummaryWriter = None):
        """
        Update the algorithm.

        Parameters
        ----------
        writer: SummaryWriter,
            the writer to record training logs
        """
        pass

    @abstractmethod
    def step(self, env: Env, state: torch.Tensor, t: int, step: int) -> Tuple[torch.Tensor, int]:
        """
        Sample one step in the environment.

        Parameters
        ----------
        env: Env,
            the training environment
        state: torch.Tensor,
            current states
        t: int,
            current simulation time step
        step: int,
            current training step
        """
        pass
