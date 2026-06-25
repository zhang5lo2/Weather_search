from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm

from src.weather_classifier.model import (
    DEFAULT_MEAN,
    DEFAULT_STD,
    build_model,
    save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a weather image classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--model", choices=["simple_cnn", "resnet18"], default="simple_cnn")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet weights for resnet18.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--random-resized-crop", action="store_true")
    parser.add_argument("--crop-scale-min", type=float, default=0.80)
    parser.add_argument("--brightness", type=float, default=0.15)
    parser.add_argument("--contrast", type=float, default=0.15)
    parser.add_argument("--saturation", type=float, default=0.12)
    parser.add_argument("--hue", type=float, default=0.0)
    parser.add_argument("--random-rotation", type=float, default=0.0)
    parser.add_argument("--no-balanced-sampler", action="store_true")
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=0,
        help="Optional quick-smoke limit per class; 0 uses all images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"device: {device}")

    train_tf, val_tf = build_transforms(
        args.image_size,
        random_resized_crop=args.random_resized_crop,
        crop_scale_min=args.crop_scale_min,
        brightness=args.brightness,
        contrast=args.contrast,
        saturation=args.saturation,
        hue=args.hue,
        random_rotation=args.random_rotation,
    )
    base_dataset = datasets.ImageFolder(args.data_dir)
    if args.limit_per_class > 0:
        indices = limited_indices(base_dataset.targets, args.limit_per_class, args.seed)
    else:
        indices = list(range(len(base_dataset)))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=[base_dataset.targets[i] for i in indices],
    )

    train_dataset = datasets.ImageFolder(args.data_dir, transform=train_tf)
    val_dataset = datasets.ImageFolder(args.data_dir, transform=val_tf)
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)

    classes = train_dataset.classes
    class_to_idx = train_dataset.class_to_idx
    print(f"class_to_idx: {class_to_idx}")
    print(f"train size: {len(train_subset)}  val size: {len(val_subset)}")
    print_class_counts("train", train_dataset.targets, train_idx, classes)
    print_class_counts("val", val_dataset.targets, val_idx, classes)

    class_weights = compute_class_weights(train_dataset.targets, train_idx, len(classes)).to(device)
    sampler = None
    shuffle = True
    if not args.no_balanced_sampler:
        sample_weights = [float(class_weights[train_dataset.targets[i]].cpu()) for i in train_idx]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args.model, num_classes=len(classes), pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macro_f1={val_f1:.4f}"
        )

        metrics = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        save_checkpoint(
            args.output_dir / "model_latest.pth",
            model,
            model_name=args.model,
            classes=classes,
            class_to_idx=class_to_idx,
            image_size=args.image_size,
            mean=DEFAULT_MEAN,
            std=DEFAULT_STD,
            metrics=metrics,
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_checkpoint(
                args.output_dir / "model_best.pth",
                model,
                model_name=args.model,
                classes=classes,
                class_to_idx=class_to_idx,
                image_size=args.image_size,
                mean=DEFAULT_MEAN,
                std=DEFAULT_STD,
                metrics=metrics,
            )
            print(f"saved best checkpoint: {args.output_dir / 'model_best.pth'}")


def build_transforms(
    image_size: int,
    *,
    random_resized_crop: bool = False,
    crop_scale_min: float = 0.80,
    brightness: float = 0.15,
    contrast: float = 0.15,
    saturation: float = 0.12,
    hue: float = 0.0,
    random_rotation: float = 0.0,
) -> tuple[transforms.Compose, transforms.Compose]:
    train_steps: list[torch.nn.Module] = []
    if random_resized_crop:
        train_steps.append(
            transforms.RandomResizedCrop(
                (image_size, image_size),
                scale=(crop_scale_min, 1.0),
                ratio=(0.90, 1.10),
            )
        )
    else:
        train_steps.append(transforms.Resize((image_size, image_size)))
    train_steps.extend(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            ),
        ]
    )
    if random_rotation > 0:
        train_steps.append(transforms.RandomRotation(degrees=random_rotation))
    train_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD),
        ]
    )
    train_tf = transforms.Compose(train_steps)
    val_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD),
        ]
    )
    return train_tf, val_tf


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    loss_sum = 0.0
    total = 0
    correct = 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        loss_sum += float(loss.item()) * batch_size
        total += batch_size
        correct += int((logits.argmax(dim=1) == y).sum().item())
    return loss_sum / total, correct / total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    loss_sum = 0.0
    total = 0
    correct = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="val", leave=False):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            pred = logits.argmax(dim=1)
            batch_size = x.size(0)
            loss_sum += float(loss.item()) * batch_size
            total += batch_size
            correct += int((pred == y).sum().item())
            y_true.extend(y.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())

    return loss_sum / total, correct / total, f1_score(y_true, y_pred, average="macro")


def compute_class_weights(targets: list[int], indices: list[int], num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for idx in indices:
        counts[targets[idx]] += 1
    counts = counts.clamp_min(1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights


def limited_indices(targets: list[int], limit_per_class: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {}
    for idx, target in enumerate(targets):
        by_class.setdefault(target, []).append(idx)
    selected: list[int] = []
    for class_indices in by_class.values():
        rng.shuffle(class_indices)
        selected.extend(class_indices[:limit_per_class])
    selected.sort()
    return selected


def print_class_counts(name: str, targets: list[int], indices: list[int], classes: list[str]) -> None:
    counts = {label: 0 for label in classes}
    for idx in indices:
        counts[classes[targets[idx]]] += 1
    summary = ", ".join(f"{label}={count}" for label, count in counts.items())
    print(f"{name} counts: {summary}")


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
