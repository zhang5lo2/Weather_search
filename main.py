from __future__ import annotations

from pathlib import Path

import numpy as np

from src.weather_classifier import Predictor

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_CANDIDATES = [
    PROJECT_ROOT / "results" / "model_best.pth",
    PROJECT_ROOT / "results" / "model_sample.pth",
]

_predictor: Predictor | None = None


def _get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        checkpoint = next((path for path in CHECKPOINT_CANDIDATES if path.exists()), None)
        if checkpoint is None:
            expected = ", ".join(str(path) for path in CHECKPOINT_CANDIDATES)
            raise FileNotFoundError(
                "No model checkpoint found. Train a model first. "
                f"Expected one of: {expected}"
            )
        _predictor = Predictor(checkpoint)
    return _predictor


def predict(X: np.ndarray) -> str:
    """
    Competition inference entry.

    Args:
        X: Image array loaded by cv2.imread, shape HxWx3 in BGR order.

    Returns:
        One of: cloudy, rainy, snowy, sunny.
    """
    return _get_predictor().predict(X)
