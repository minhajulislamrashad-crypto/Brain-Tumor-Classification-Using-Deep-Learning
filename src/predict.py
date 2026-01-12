import torch
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

def show_image(img_tensor):
    """Displays a normalized tensor image (expects range [-1,1] normalization)."""
    img = img_tensor * 0.5 + 0.5  # unnormalize from [-1,1] -> [0,1]
    plt.imshow(to_pil_image(img))
    plt.axis("off")
    plt.show()

def predict_one(model, img_tensor, class_names, device):
    """
    Predicts class for a single image tensor.

    Args:
        model (torch.nn.Module): Trained model.
        img_tensor (torch.Tensor): Single image tensor (C,H,W).
        class_names (list[str]): Class labels.
        device (torch.device): CPU/CUDA device.

    Returns:
        pred_class (str): Predicted class name.
        confidence (float): Softmax confidence.
        pred_idx (int): Predicted class index.
    """
    model.eval()

    x = img_tensor.unsqueeze(0).to(device)  # (1,C,H,W)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()

    return class_names[pred_idx], confidence, pred_idx
