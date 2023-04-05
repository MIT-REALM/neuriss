import torch
import torch.nn as nn

from typing import Tuple

from .mlp import MLP


class PDQuadraticNet(nn.Module):
    """
    Positive definite network: y = x^T * Q * x + 0.5 * f_NN(x)^T * f_NN(x) + g_NN(x).
    Note that output dim is 1, and the output activation of g_NN(x) is ReLU().
    Q and NNs are trainable.
    """

    def __init__(
            self,
            in_dim: int,
            hidden_layers: tuple,
            hidden_activation: nn.Module = nn.Tanh(),
            init: bool = True, gain: float = 1.
    ):
        super(PDQuadraticNet, self).__init__()
        self.f = MLP(
            in_dim, in_dim, hidden_layers, hidden_activation, None, init, gain, limit_lip=True
        )
        self.Q = nn.Parameter(torch.eye(in_dim) + 0.1 * torch.ones((in_dim, in_dim)))
        self.g = MLP(
            in_dim, 1, hidden_layers, hidden_activation, nn.ReLU(), init, gain, limit_lip=True
        )

    def quadratic(self, x: torch.Tensor) -> torch.Tensor:
        return torch.bmm(torch.matmul(x, self.quadratic_matrix).unsqueeze(1), x.unsqueeze(2)).squeeze(2)

    @property
    def quadratic_matrix(self) -> torch.Tensor:
        return torch.matmul(self.Q.t(), self.Q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quadratic(x) + 0.5 * (self.f(x)**2).sum(dim=1, keepdim=True) + self.g(x)

    def forward_jacobian(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        eye = torch.eye(x.shape[1]).type_as(x)
        jacobian_f = eye.repeat(x.shape[0], 1, 1)
        jacobian_g = eye.repeat(x.shape[0], 1, 1)
        y_f = x
        y_g = x

        # compute layer by layer in f
        for layer in self.f.net:
            y_f = layer(y_f)

            if isinstance(layer, nn.Linear):
                jacobian_f = torch.matmul(layer.weight, jacobian_f)
            elif isinstance(layer, nn.Tanh):
                jacobian_f = torch.matmul(torch.diag_embed(1 - y_f ** 2), jacobian_f)
            elif isinstance(layer, nn.ReLU):
                jacobian_f = torch.matmul(torch.diag_embed(torch.sign(y_f)), jacobian_f)

        # add last layer in f
        jacobian_f = torch.bmm(y_f.unsqueeze(1), jacobian_f)
        y_f = 0.5 * (y_f * y_f).sum(dim=1, keepdim=True)

        # compute layer by layer in g
        for layer in self.g.net:
            y_g = layer(y_g)

            if isinstance(layer, nn.Linear):
                jacobian_g = torch.matmul(layer.weight, jacobian_g)
            elif isinstance(layer, nn.Tanh):
                jacobian_g = torch.matmul(torch.diag_embed(1 - y_g ** 2), jacobian_g)
            elif isinstance(layer, nn.ReLU):
                jacobian_g = torch.matmul(torch.diag_embed(torch.sign(y_g)), jacobian_g)

        # add the quadratic part
        y = y_f + y_g + self.quadratic(x)
        jacobian = jacobian_f + jacobian_g + 2 * torch.matmul(x, self.quadratic_matrix).unsqueeze(1)

        return y, jacobian

    def disable_grad(self):
        for i in self.parameters():
            i.requires_grad = False
