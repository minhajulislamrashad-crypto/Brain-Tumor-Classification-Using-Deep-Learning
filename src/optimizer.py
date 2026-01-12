# ============================================================
# Optimization: loss function + optimizer factory
# ============================================================

from torch import nn, optim


def get_optimizer_and_loss(model, lr: float = 1e-4):
    """
    Creates the loss function and optimizer for training.

    Args:
        model (torch.nn.Module): The neural network to optimize.
        lr (float): Learning rate for AdamW.

    Returns:
        loss_fn (nn.Module): CrossEntropyLoss for multi-class classification.
        optimizer (optim.Optimizer): AdamW optimizer configured for the model.
    """
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    return loss_fn, optimizer
