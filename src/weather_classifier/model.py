from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision import models

DEFAULT_CLASSES = ["cloudy", "rainy", "snowy", "sunny"]
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


class WeatherCNN(nn.Module):
    """Small baseline CNN that trains quickly on CPU."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 32),
            nn.MaxPool2d(2),
            _conv_block(32, 64),
            nn.MaxPool2d(2),
            _conv_block(64, 128),
            nn.MaxPool2d(2),
            _conv_block(128, 192),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(192, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = False,
) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "simple_cnn":
        return WeatherCNN(num_classes=num_classes)
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model: {model_name}")


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    model_name: str,
    classes: list[str],
    class_to_idx: dict[str, int],
    image_size: int,
    mean: list[float] | None = None,
    std: list[float] | None = None,
    metrics: dict[str, float] | None = None,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "model_name": model_name,
        "classes": classes,
        "class_to_idx": class_to_idx,
        "image_size": image_size,
        "mean": mean or DEFAULT_MEAN,
        "std": std or DEFAULT_STD,
        "metrics": metrics or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    path = Path(path)
    checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        classes = checkpoint.get("classes", DEFAULT_CLASSES)
        model_name = checkpoint.get("model_name", "simple_cnn")
        model = build_model(model_name, num_classes=len(classes), pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        metadata = {
            "model_name": model_name,
            "classes": classes,
            "image_size": int(checkpoint.get("image_size", 224)),
            "mean": checkpoint.get("mean", DEFAULT_MEAN),
            "std": checkpoint.get("std", DEFAULT_STD),
            "metrics": checkpoint.get("metrics", {}),
        }
    else:
        classes = DEFAULT_CLASSES
        model = build_model("simple_cnn", num_classes=len(classes), pretrained=False)
        model.load_state_dict(checkpoint)
        metadata = {
            "model_name": "simple_cnn",
            "classes": classes,
            "image_size": 224,
            "mean": DEFAULT_MEAN,
            "std": DEFAULT_STD,
            "metrics": {},
        }

    model.to(device)
    model.eval()
    return model, metadata
