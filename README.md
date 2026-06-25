# RAICOM Weather Image Classification

This project is organized for the RAICOM 2026 "智海算法调优赛题" / Mo platform image-classification task. The local dataset is a four-class weather image dataset:

- `cloudy`
- `rainy`
- `snowy`
- `sunny`

The competition scoring described in the downloaded rules is based on F1, so training and validation report macro F1 rather than only accuracy.

Competition FAQ constraints for final inference: CPU only, 2 cores, 8 GiB RAM, and torch `2.1.7` on the platform. The local Conda environment targets the public PyTorch `2.1.x` / torchvision `0.16.x` compatibility line.

## Project Layout

```text
.
+-- main.py                         # Mo/competition inference entry, exposes predict(X)
+-- train.py                        # Local training entry
+-- evaluate.py                     # Fixed validation-set evaluation entry
+-- environment.yml                 # Conda environment definition
+-- requirements.txt                # Minimal pip dependency list
+-- data/train/                     # Local training images, not committed
+-- docs/
|   +-- baselines.md                # Baselines, experiments, metrics
|   +-- competition.md              # Competition constraints summary
|   +-- raicom_2026_algorithm_tuning_rules.pdf
+-- results/
|   +-- model_best.pth              # Default tracked inference checkpoint
+-- src/weather_classifier/         # Shared model and prediction code
+-- AGENTS.md                       # Project working rules
+-- archive/original_submission/    # Original notebook/scripts/dependency freeze
```

The repository tracks source code, documentation, environment files, and the default `results/model_best.pth` checkpoint. Local training images, duplicate experiment checkpoints, generated submission packages, and Python caches are intentionally ignored.

## Environment

Create the conda environment:

```powershell
conda env create -f environment.yml
conda activate raicom-weather
```

If the environment already exists:

```powershell
conda env update -n raicom-weather -f environment.yml --prune
conda activate raicom-weather
```

## Train

Fast baseline:

```powershell
conda run -n raicom-weather python train.py --data-dir data/train --epochs 10 --batch-size 32 --image-size 128 --model simple_cnn --num-workers 0
```

Transfer-learning baseline, usually better when downloading pretrained weights is allowed:

```powershell
conda run -n raicom-weather python train.py --data-dir data/train --epochs 15 --model resnet18 --pretrained --num-workers 0
```

The best checkpoint is written to:

```text
results/model_best.pth
```

## Current Default Model

The current default checkpoint is `exp04_resnet18_224_2026-06-25`, stored at `results/model_best.pth`.

```text
model: resnet18 pretrained
image_size: 224
val_acc: 0.9280
val_macro_f1: 0.9128
val_weighted_f1: 0.9281
CPU speed: about 88 images/s
```

Evaluate it on the fixed local validation split:

```powershell
conda run -n raicom-weather python evaluate.py --data-dir data/train --checkpoint results/model_best.pth --batch-size 32 --num-workers 0 --device cpu
```

## Predict Entry

`main.py` exposes the function required by the competition-style evaluator:

```python
def predict(X):
    ...
```

`X` is expected to be a `numpy.ndarray` image loaded by `cv2.imread`, so the predictor converts BGR to RGB before applying the same normalization used during training.

Quick import and single-image check:

```powershell
conda run -n raicom-weather python -c "import main; print(callable(main.predict))"
conda run -n raicom-weather python -c "import cv2, main; img=cv2.imread('data/train/cloudy/cloudy_00001.jpg'); print(main.predict(img))"
```

## Dataset Snapshot

Current local class counts:

```text
cloudy: 2184
rainy : 446
snowy : 403
sunny : 1966
```

Because the dataset is imbalanced, `train.py` uses class-weighted loss by default and can use a balanced sampler.
