from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.weather_classifier.model import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained weather classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--checkpoint", type=Path, default=Path("results/model_best.pth"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--predictions-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model, metadata = load_checkpoint(args.checkpoint, device)
    classes = list(metadata["classes"])

    transform = transforms.Compose(
        [
            transforms.Resize((metadata["image_size"], metadata["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=metadata["mean"], std=metadata["std"]),
        ]
    )
    base_dataset = datasets.ImageFolder(args.data_dir)
    eval_dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    indices = list(range(len(base_dataset)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=base_dataset.targets,
    )
    val_subset = Subset(eval_dataset, val_idx)
    val_paths = [eval_dataset.samples[idx][0] for idx in val_idx]
    loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    weighted_criterion = nn.CrossEntropyLoss(
        weight=compute_class_weights(base_dataset.targets, train_idx, len(classes)).to(device)
    )
    unweighted_criterion = nn.CrossEntropyLoss()
    y_true: list[int] = []
    y_pred: list[int] = []
    weighted_loss_sum = 0.0
    unweighted_loss_sum = 0.0
    total = 0

    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            weighted_loss = weighted_criterion(logits, y)
            unweighted_loss = unweighted_criterion(logits, y)
            pred = logits.argmax(dim=1)

            batch_size = x.size(0)
            weighted_loss_sum += float(weighted_loss.item()) * batch_size
            unweighted_loss_sum += float(unweighted_loss.item()) * batch_size
            total += batch_size
            y_true.extend(y.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
    elapsed = time.perf_counter() - started

    print(f"checkpoint: {args.checkpoint}")
    print(f"model: {metadata['model_name']}")
    print(f"classes: {classes}")
    print(f"image_size: {metadata['image_size']}")
    print(f"device: {device}")
    print(f"val_size: {total}")
    print(f"val_loss_weighted: {weighted_loss_sum / total:.4f}")
    print(f"val_loss_unweighted: {unweighted_loss_sum / total:.4f}")
    print(f"val_acc: {accuracy_score(y_true, y_pred):.4f}")
    print(f"val_macro_f1: {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"val_weighted_f1: {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"elapsed_seconds: {elapsed:.2f}")
    print(f"images_per_second: {total / elapsed:.2f}")
    print("classification_report:")
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))
    print("confusion_matrix:")
    print(confusion_matrix(y_true, y_pred).tolist())
    if metadata.get("metrics"):
        print(f"checkpoint_metrics: {metadata['metrics']}")
    if args.predictions_csv is not None:
        write_predictions(args.predictions_csv, val_paths, y_true, y_pred, classes)
        print(f"predictions_csv: {args.predictions_csv}")


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def compute_class_weights(targets: list[int], indices: list[int], num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for idx in indices:
        counts[targets[idx]] += 1
    counts = counts.clamp_min(1.0)
    return counts.sum() / (num_classes * counts)


def write_predictions(
    path: Path,
    image_paths: list[str],
    y_true: list[int],
    y_pred: list[int],
    classes: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "path", "true_label", "pred_label", "is_correct"],
        )
        writer.writeheader()
        for i, (image_path, true_idx, pred_idx) in enumerate(zip(image_paths, y_true, y_pred)):
            writer.writerow(
                {
                    "index": i,
                    "path": image_path,
                    "true_label": classes[true_idx],
                    "pred_label": classes[pred_idx],
                    "is_correct": true_idx == pred_idx,
                }
            )


if __name__ == "__main__":
    main()
