import torch
import pytest

from scream.models.mlp import LinearModel, get_activation
from scream.utils.metrics import count_parameters


# ---------------------------------------------------------------------------
# get_activation
# ---------------------------------------------------------------------------

def test_get_activation_unknown():
    with pytest.raises(ValueError):
        get_activation("tanh_custom")


# ---------------------------------------------------------------------------
# LinearModel — forward pass shapes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("B,input_dim", [(8, 9), (1, 4)])
def test_forward_shape_plain(B, input_dim):
    model = LinearModel(input_dim=input_dim, num_layers=2, hidden_units=16)
    x = torch.randn(B, input_dim)
    out = model(x)
    assert out.shape == (B, 1)


def test_forward_shape_residual():
    model = LinearModel(input_dim=9, num_layers=3, hidden_units=32, residual=True)
    x = torch.randn(8, 9)
    out = model(x)
    assert out.shape == (8, 1)


def test_forward_shape_dropout():
    model = LinearModel(input_dim=9, num_layers=2, hidden_units=16, dropout=0.5)
    model.eval()
    x = torch.randn(8, 9)
    out = model(x)
    assert out.shape == (8, 1)


def test_forward_shape_layer_norm():
    model = LinearModel(input_dim=9, num_layers=2, hidden_units=16, layer_norm=True)
    x = torch.randn(8, 9)
    out = model(x)
    assert out.shape == (8, 1)


# ---------------------------------------------------------------------------
# count_parameters
# ---------------------------------------------------------------------------

def test_count_parameters_single_layer():
    # num_layers=1: no hidden layers, just Linear(input_dim, 1)
    # params = input_dim * 1 + 1
    model = LinearModel(input_dim=4, num_layers=1, hidden_units=8)
    assert count_parameters(model) == 4 * 1 + 1  # = 5


def test_count_parameters_two_layers():
    # num_layers=2: Linear(4, 8) + Linear(8, 1) = 40 + 9 = 49
    model = LinearModel(input_dim=4, num_layers=2, hidden_units=8)
    assert count_parameters(model) == (4 * 8 + 8) + (8 * 1 + 1)  # = 49
