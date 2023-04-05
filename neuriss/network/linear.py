import torch
import torch.nn as nn


class ChiFunctions(nn.Module):

    def __init__(self, n: int):
        super(ChiFunctions, self).__init__()
        self.n = n
        self.a = [nn.Parameter(torch.tensor(0.)) for _ in range(n)]

        # register params
        for i, param in enumerate(self.a):
            self.register_parameter(f'coef{i}', param)

    def forward(self, x: torch.Tensor) -> list:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        assert x.shape[1] == self.n
        y = []
        for i in range(self.n):
            y.append(torch.sigmoid(self.a[i]) * x[:, i].unsqueeze(1))
        return y

    def value(self, i: int, x: torch.Tensor) -> torch.Tensor:
        return self.a[i] * x
