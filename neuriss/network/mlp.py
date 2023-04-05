import torch
import torch.nn as nn

from torch.nn.utils import spectral_norm

from .utils import init_param


class MLP(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, hidden_layers: tuple,
                 hidden_activation: nn.Module = nn.ReLU(), output_activation: nn.Module = None,
                 init: bool = True, gain: float = 1., limit_lip=False):
        super().__init__()

        # build MLP
        layers = []
        units = in_dim
        for next_units in hidden_layers:
            if init:
                if limit_lip:
                    layers.append(init_param(spectral_norm(nn.Linear(units, next_units)), gain=gain))
                else:
                    layers.append(init_param(nn.Linear(units, next_units), gain=gain))
            else:
                if limit_lip:
                    layers.append(spectral_norm(nn.Linear(units, next_units)))
                else:
                    layers.append(nn.Linear(units, next_units))
            layers.append(hidden_activation)
            units = next_units
        if init:
            if limit_lip:
                layers.append(init_param(spectral_norm(nn.Linear(units, out_dim)), gain=gain))
            else:
                layers.append(init_param(nn.Linear(units, out_dim), gain=gain))
        else:
            if limit_lip:
                layers.append(spectral_norm(nn.Linear(units, out_dim)))
            else:
                layers.append(nn.Linear(units, out_dim))
        if output_activation is not None:
            layers.append(output_activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NormalizedMLP(MLP):

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            input_mean: torch.Tensor,
            input_std: torch.Tensor,
            hidden_layers: tuple,
            hidden_activation: nn.Module = nn.ReLU(),
            output_activation: nn.Module = None,
            init: bool = True,
            gain: float = 1.
    ):
        super().__init__(in_dim, out_dim, hidden_layers, hidden_activation, output_activation, init, gain)

        assert input_mean.ndim == 1 and input_mean.shape[0] == in_dim
        assert input_std.ndim == 1 and input_std.shape[0] == in_dim
        self.input_mean = input_mean
        self.input_std = input_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x_trans = self.normalize_input(x)
        return self.net(x_trans)

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        nonzero_std_dim = torch.nonzero(self.input_std)
        zero_mask = torch.ones(self.input_std.shape[0]).type_as(self.input_std)
        zero_mask[nonzero_std_dim] = 0
        zero_mask = zero_mask.reshape(x[0].shape)
        x_trans = (x - self.input_mean.reshape(x[0].shape)) / (self.input_std.reshape(x[0].shape) + zero_mask)
        return x_trans
