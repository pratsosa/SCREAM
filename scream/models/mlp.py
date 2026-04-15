from __future__ import annotations

from typing import Union

import torch.nn as nn


def get_activation(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()           # also called Swish
    if name == "gelu":
        return nn.GELU()
    if name == "leakyrelu":
        return nn.LeakyReLU(0.1)
    raise ValueError(f"Unknown activation '{name}'")


class ResidualBlock(nn.Module):
    """Pre-LN residual MLP block."""

    def __init__(self, dim, activation="silu"):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = get_activation(activation)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        # Pre-LayerNorm → Linear → Activation → Linear → + skip
        out = self.norm(x)
        out = self.fc1(out)
        out = self.act(out)
        out = self.fc2(out)
        return x + out


class LinearModel(nn.Module):
    def __init__(
        self,
        input_dim,
        num_layers: int = 3,
        hidden_units: Union[int, list[int]] = 256,
        dropout: float = 0.0,
        layer_norm: bool = False,
        activation: str = "relu",
        residual: bool = False,
    ):
        """
        hidden_units can be an int (uniform hidden size, num_layers controls depth)
        or a list of ints (one entry per hidden layer; num_layers is ignored).

        If residual=True:
            - Uses Pre-LN ResidualBlocks for all hidden layers
            - All hidden dims must be equal (ResidualBlock requires fixed dim)
            - LayerNorm is applied inside blocks, independent of layer_norm flag
        """
        super().__init__()
        assert 0.0 <= dropout < 1.0

        # Resolve hidden layer widths
        if isinstance(hidden_units, list):
            hidden_dims = hidden_units
        else:
            assert num_layers >= 1
            hidden_dims = [hidden_units] * (num_layers - 1)

        self.residual = residual

        layers = []
        in_dim = input_dim

        if residual:
            if len(set(hidden_dims)) > 1:
                raise ValueError(
                    "residual=True requires all hidden dims to be equal; "
                    f"got {hidden_dims}"
                )
            h = hidden_dims[0] if hidden_dims else input_dim
            layers.append(nn.Linear(in_dim, h))
            for _ in range(len(hidden_dims) - 1):
                layers.append(ResidualBlock(h, activation=activation))
            layers.append(nn.Linear(h, 1))
            self.net = nn.Sequential(*layers)
            return

        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(get_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
