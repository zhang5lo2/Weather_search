from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

from .model import load_checkpoint


class Predictor:
    def __init__(self, checkpoint_path: str | Path, device: str = "auto") -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {self.checkpoint_path}. "
                "Run train.py first to create results/model_best.pth."
            )

        self.device = _resolve_device(device)
        self.model, metadata = load_checkpoint(self.checkpoint_path, self.device)
        self.classes = metadata["classes"]
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((metadata["image_size"], metadata["image_size"])),
                transforms.ToTensor(),
                transforms.Normalize(mean=metadata["mean"], std=metadata["std"]),
            ]
        )

    def predict(self, image_bgr: np.ndarray) -> str:
        if image_bgr is None:
            raise ValueError("predict expected an image array, got None")
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError(f"predict expected an HxWx3 image, got shape {image_bgr.shape}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        x = self.transform(image_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            pred_idx = int(logits.argmax(dim=1).item())
        return self.classes[pred_idx]


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
