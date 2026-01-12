# ============================================================
# Training: trains the CNN model and saves weights
# ============================================================

import os
import torch


def train_model(
    model,
    train_loader,
    loss_fn,
    optimizer,
    device,
    epochs: int = 5,
    save_path: str = "outputs/model.pt",
):
    """
    Trains a PyTorch model on the training dataset.

    Args:
        model (torch.nn.Module): Model to train.
        train_loader (torch.utils.data.DataLoader): Training loader yielding (images, labels).
        loss_fn (torch.nn.Module): Loss function (e.g., CrossEntropyLoss).
        optimizer (torch.optim.Optimizer): Optimizer (e.g., AdamW).
        device (torch.device): CPU or CUDA device.
        epochs (int): Number of epochs.
        save_path (str): File path to save trained weights (.pt).

    Returns:
        list[float]: Average training loss per epoch.
    """
    model.train()
    history = []

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):
        total_loss = 0.0
        total_samples = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)

            loss.backward()
            optimizer.step()

            bs = y.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

        avg_loss = total_loss / max(total_samples, 1)
        history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Saved model to: {save_path}")

    return history
