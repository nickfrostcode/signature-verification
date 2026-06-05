import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm


def _normalize_tensor(tensor: torch.Tensor) -> np.ndarray:
    values = tensor.cpu().numpy()
    values = values - values.min()
    if values.max() > 0:
        values = values / values.max()
    return values


def compute_input_saliency(model: torch.nn.Module, image_tensor: torch.Tensor) -> np.ndarray:
    """Compute saliency map for a single image tensor."""
    image = image_tensor.clone().detach().unsqueeze(0)
    image.requires_grad_(True)

    output = model(image)
    if output.ndim == 2 and output.shape[1] == 1:
        score = output.squeeze(1).squeeze(0)
    else:
        score = output.squeeze()

    model.zero_grad()
    score.backward(retain_graph=False)

    grads = image.grad.abs().max(dim=1)[0]
    saliency = _normalize_tensor(grads[0])
    return saliency


from typing import Tuple

def compute_siamese_saliency(model: torch.nn.Module, image_a: torch.Tensor, image_b: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Compute saliency maps for a siamese image pair."""
    image_a = image_a.clone().detach().unsqueeze(0)
    image_b = image_b.clone().detach().unsqueeze(0)
    image_a.requires_grad_(True)
    image_b.requires_grad_(True)

    output = model(image_a, image_b)
    if output.ndim == 2 and output.shape[1] == 1:
        score = output.squeeze(1).squeeze(0)
    else:
        score = output.squeeze()

    model.zero_grad()
    score.backward(retain_graph=False)

    saliency_a = _normalize_tensor(image_a.grad.abs().max(dim=1)[0].squeeze(0))
    saliency_b = _normalize_tensor(image_b.grad.abs().max(dim=1)[0].squeeze(0))
    return saliency_a, saliency_b


def _build_heatmap(saliency: np.ndarray) -> np.ndarray:
    colormap = cm.get_cmap('jet')
    heatmap = colormap(saliency)[:, :, :3]
    heatmap = np.uint8(heatmap * 255)
    return heatmap


def save_saliency_overlay(original_image_path: str, saliency_map: np.ndarray, output_path: str):
    original = Image.open(original_image_path).convert('RGB')
    heatmap = _build_heatmap(saliency_map)
    overlay = Image.fromarray(heatmap).resize(original.size, resample=Image.BILINEAR)
    blended = Image.blend(original, overlay, alpha=0.45)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    blended.save(output_path)


def save_raw_saliency(saliency_map: np.ndarray, output_path: str):
    plt.figure(figsize=(4, 4), dpi=100)
    plt.axis('off')
    plt.imshow(saliency_map, cmap='hot')
    plt.tight_layout(pad=0)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
