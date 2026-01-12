# ============================================================
# Model: Lightweight CNN for 4-class MRI tumor classification
# ============================================================

import torch
from torch import nn


def create_model(num_classes: int = 4):
    """
    Creates the lightweight CNN model used in the report.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: CNN model (not moved to device).
    """
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, 1, 1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(32, 64, 3, 1, 1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(64, 128, 3, 1, 1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Flatten(),
        nn.Linear(128 * 16 * 16, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
    )
    return model
