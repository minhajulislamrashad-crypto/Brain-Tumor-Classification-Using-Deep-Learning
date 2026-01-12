# ============================================================
# Evaluation: computes test loss and accuracy
# ============================================================

import torch


def evaluate_model(model, test_loader, loss_fn, device):
    """
    Evaluates the model on a test dataset.

    Args:
        model (torch.nn.Module): Trained CNN model.
        test_loader (torch.utils.data.DataLoader): Test loader yielding (images, labels).
        loss_fn (torch.nn.Module): Loss function (e.g., CrossEntropyLoss).
        device (torch.device): CPU or CUDA device.

    Returns:
        tuple[float, float]: (test_loss, accuracy_percent)
    """
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = loss_fn(logits, y)

            bs = y.size(0)
            total_loss += loss.item() * bs
            total += bs

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()

    test_loss = total_loss / max(total, 1)
    accuracy = 100.0 * correct / max(total, 1)

    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {accuracy:.2f}%")
    return test_loss, accuracy
