import torch
import torch.nn as nn
import os

from typing import Tuple

from .controller import NeuralController
from ..env.base import Env

from neuriss.network.pd_quadratic import PDQuadraticNet


class NeuralCLFController(nn.Module):
    """
    Neural CLF Controller.

    References
    ----------
    [1] Charles Dawson, Zengyi Qin, Sicun Gao, and Chuchu Fan.
    Safe nonlinear control using robust neural lyapunov-barrier functions.
    Conference on Robot Learning, 2021.
    """

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            device: torch.device,
            goal_point: torch.Tensor,
            u_eq: torch.Tensor,
            state_std: torch.Tensor,
            env: Env,
            clf_lambda: float = 1.0,
            residue: bool = False,
    ):
        super(NeuralCLFController, self).__init__()

        # set up controller
        self.controller = NeuralController(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_layers=(64, 64)
        )

        # set up Lyapunov network
        self.lyapunov = PDQuadraticNet(
            in_dim=state_dim,
            hidden_layers=(64, 64),
            hidden_activation=nn.Tanh(),
        )

        # save parameters
        self.clf_lambda = clf_lambda
        self.device = device
        self.goal_point = goal_point.to(device)
        self.state_std = state_std.to(device)
        self.u_eq = u_eq.to(device)
        self.residue = residue
        self.env = env

    def normalize_state(self, x: torch.Tensor) -> torch.Tensor:
        return (x.to(self.device) - self.goal_point) / self.state_std

    def set_controller(self, controller: NeuralController):
        self.controller.load_state_dict(controller.state_dict())

    def u(self, x: torch.Tensor) -> torch.Tensor:
        if self.residue:
            with torch.no_grad():
                u_nominal = self.env.u_nominal(x)
            return self.controller(self.normalize_state(x)) + u_nominal
        else:
            return self.controller(self.normalize_state(x)) + self.u_eq

    def act(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.residue:
                return self.controller(self.normalize_state(x)) + self.env.u_nominal(x)
            else:
                return self.controller(self.normalize_state(x)) + self.u_eq

    def V_with_Jacobian(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_trans = self.normalize_state(x)
        V, JV = self.lyapunov.forward_jacobian(x_trans)
        return V, JV

    def V(self, x: torch.Tensor) -> torch.Tensor:
        x_trans = self.normalize_state(x)
        return self.lyapunov(x_trans)

    def save(self, path: str):
        torch.save(self.state_dict(), os.path.join(path, 'lyapunov.pkl'))

    def load(self, path: str, device: torch.device):
        self.load_state_dict(torch.load(os.path.join(path, 'lyapunov.pkl'), map_location=device))

    def disable_grad_lyapunov(self):
        self.lyapunov.disable_grad()

    def disable_grad_ctrl(self):
        self.controller.disable_grad()
