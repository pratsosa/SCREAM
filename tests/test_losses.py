import pytest
import torch

from scream.losses.mc_marginal import mc_marginal_bce_loss


def _make_inputs(N, B, seed=0):
    """Return (y_pred, y_true) with shapes (N, B) and (B,)."""
    torch.manual_seed(seed)
    y_pred = torch.randn(N, B)
    y_true = torch.randint(0, 2, (B,)).float()
    return y_pred, y_true


# ---------------------------------------------------------------------------
# Shape / basic
# ---------------------------------------------------------------------------

def test_returns_scalar():
    y_pred, y_true = _make_inputs(N=5, B=8)
    loss = mc_marginal_bce_loss(y_pred, y_true)
    assert loss.shape == ()


def test_loss_is_positive():
    y_pred, y_true = _make_inputs(N=5, B=8)
    loss = mc_marginal_bce_loss(y_pred, y_true)
    assert loss.item() > 0.0


def test_gradient_flows():
    y_pred, y_true = _make_inputs(N=3, B=4)
    y_pred = y_pred.requires_grad_(True)
    loss = mc_marginal_bce_loss(y_pred, y_true)
    loss.backward()
    assert y_pred.grad is not None
    assert not torch.isnan(y_pred.grad).any()


# ---------------------------------------------------------------------------
# pos_weight
# ---------------------------------------------------------------------------

def test_pos_weight_changes_loss():
    y_pred, y_true = _make_inputs(N=4, B=8)
    loss_no_weight = mc_marginal_bce_loss(y_pred, y_true)
    pw = torch.tensor([2.0])
    loss_weighted = mc_marginal_bce_loss(y_pred, y_true, pos_weight=pw)
    assert loss_no_weight.item() != pytest.approx(loss_weighted.item())


def test_pos_weight_gradient_flows():
    y_pred, y_true = _make_inputs(N=4, B=8)
    y_pred = y_pred.requires_grad_(True)
    pw = torch.tensor([3.0])
    loss = mc_marginal_bce_loss(y_pred, y_true, pos_weight=pw)
    loss.backward()
    assert y_pred.grad is not None
    assert not torch.isnan(y_pred.grad).any()


# ---------------------------------------------------------------------------
# Limiting behaviour
# ---------------------------------------------------------------------------

def test_zero_errors_matches_standard_bce():
    """With N=1 MC sample the marginal loss should equal standard BCE."""
    B = 16
    torch.manual_seed(7)
    y_pred_1d = torch.randn(B)
    y_true = torch.randint(0, 2, (B,)).float()

    mc_loss = mc_marginal_bce_loss(y_pred_1d.unsqueeze(0), y_true)

    standard_loss = torch.nn.BCEWithLogitsLoss()(y_pred_1d, y_true)
    assert mc_loss.item() == pytest.approx(standard_loss.item(), rel=1e-5)


def test_large_n_samples_stable():
    """Loss should not be NaN or Inf with many MC samples."""
    y_pred, y_true = _make_inputs(N=100, B=32)
    loss = mc_marginal_bce_loss(y_pred, y_true)
    assert torch.isfinite(loss)


def test_all_signal_labels():
    """All-positive batch should not crash."""
    y_pred = torch.randn(5, 8)
    y_true = torch.ones(8)
    loss = mc_marginal_bce_loss(y_pred, y_true)
    assert torch.isfinite(loss)


def test_all_background_labels():
    """All-negative batch should not crash."""
    y_pred = torch.randn(5, 8)
    y_true = torch.zeros(8)
    loss = mc_marginal_bce_loss(y_pred, y_true)
    assert torch.isfinite(loss)


def test_batch_size_one():
    y_pred = torch.randn(5, 1)
    y_true = torch.ones(1)
    loss = mc_marginal_bce_loss(y_pred, y_true)
    assert loss.shape == ()
    assert torch.isfinite(loss)
