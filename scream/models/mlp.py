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
        hidden_units: int = 256,
        dropout: float = 0.0,
        layer_norm: bool = False,
        activation: str = "relu",
        residual: bool = False,
    ):
        """
        If residual=True:
            - Uses Pre-LN ResidualBlocks for all hidden layers
            - LayerNorm is applied inside blocks, independent of layer_norm flag
        """
        super().__init__()
        assert num_layers >= 1
        assert 0.0 <= dropout < 1.0

        self.residual = residual

        layers = []
        in_dim = input_dim

        if residual:
            layers.append(nn.Linear(in_dim, hidden_units))
            for _ in range(num_layers - 1):
                layers.append(ResidualBlock(hidden_units, activation=activation))
            layers.append(nn.Linear(hidden_units, 1))
            self.net = nn.Sequential(*layers)
            return

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_units))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_units))
            layers.append(get_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_units

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
