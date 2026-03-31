import torch
from torch import nn


def mc_marginal_bce_loss(y_pred, y_true, pos_weight: torch.Tensor = None):
    """
    Monte Carlo marginal likelihood loss for binary classification.

    Parameters
    ----------
    y_pred : torch.Tensor
        Logits predicted by the model with shape (N, B)
        where:
            N = number of Monte Carlo samples per input
            B = batch size (number of data points)

    y_true : torch.Tensor
        Ground truth labels with shape (B,).

    Returns
    -------
    torch.Tensor
        Scalar loss = - log [ (1/N) * sum_j p(y|x_j) ] averaged over batch.
    """

    # BCEWithLogitsLoss computes -log p(y|x) directly and is numerically stable.
    # Using reduction='none' gives shape (N, B)
    if pos_weight is None:
        bce = nn.BCEWithLogitsLoss(reduction='none')
    else:
        # pos_weight must be a 1-element tensor on the correct device
        pw = pos_weight.to(dtype=y_pred.dtype, device=y_pred.device)
        bce = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pw)

    # Expand y_true to match y_pred shape.
    # y_true: (B,) → (1, B) → (N, B)
    y_true_expanded = y_true.unsqueeze(0).expand_as(y_pred)

    # Compute elementwise BCE = -log p(y|x_MC_sample)
    # Shape: (N, B)
    bce_losses = bce(y_pred, y_true_expanded)

    # Compute log(sum_j p_j) stably
    # logsumexp(log p_j) = logsumexp(-bce_losses)
    log_total_likelihood = torch.logsumexp(-bce_losses, dim=0)  # sum over MC samples

    # Subtract log(N) → gives log( mean_j p_j )
    N = y_pred.size(0)
    log_marginal_likelihood = log_total_likelihood - torch.log(torch.tensor(float(N), device=y_pred.device))

    # Loss is negative log marginal likelihood, averaged over batch
    loss = -log_marginal_likelihood.mean()

    return loss
