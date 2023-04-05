import torch
import torch.nn as nn

from neuriss.network.mlp import MLP


class NeuralController(nn.Module):
    """
    Neural network controller.

    Parameters
    ----------
    state_dim: int
    action_dim: int
    hidden_layers: tuple
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_layers: tuple = (128, 128)):
        super(NeuralController, self).__init__()
        self.net = MLP(
            in_dim=state_dim,
            out_dim=action_dim,
            hidden_layers=hidden_layers,
            hidden_activation=nn.Tanh(),
            limit_lip=True
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def act(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.net(x)
