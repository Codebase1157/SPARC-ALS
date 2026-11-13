"""Speaker-level detection metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(balanced_accuracy_score(y_true, y_pred))


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_score))


def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (p > bins[i]) & (p <= bins[i + 1]) if i > 0 else (p <= bins[1])
        if mask.sum() == 0:
            continue
        conf = p[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / len(p)) * abs(conf - acc)
    return float(ece)


def brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    return float(brier_score_loss(y_true, p))


def speaker_decision(p: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (p >= threshold).astype(int)
