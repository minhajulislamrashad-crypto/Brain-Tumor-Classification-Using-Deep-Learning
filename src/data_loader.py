# ============================================================
# Data Loader: builds PyTorch DataLoaders for training/testing
# ============================================================

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders(train_dir, test_dir, batch_size=32, num_workers=4):
    """
    Creates training and testing DataLoaders.

    Args:
        train_dir (str): Path to training dataset folder (ImageFolder format).
        test_dir (str): Path to testing dataset folder (ImageFolder format).
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        train_dl (DataLoader): DataLoader for the training set.
        test_dl (DataLoader): DataLoader for the test set.
    """

    tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=tf)
    test_ds = datasets.ImageFolder(test_dir, transform=tf)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_dl, test_dl
