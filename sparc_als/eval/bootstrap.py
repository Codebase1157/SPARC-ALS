"""Class-stratified speaker-level cluster bootstrap (Field & Welsh 2007)."""
from __future__ import annotations

from typing import Callable

import numpy as np


def cluster_bootstrap_ci(
    stat_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (point, low, high) 100(1-alpha)% percentile interval."""
    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    stats = np.empty(n_resamples, dtype=float)
    for r in range(n_resamples):
        sp = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        sn = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([sp, sn])
        stats[r] = stat_fn(y_true[idx], y_score[idx])
    point = stat_fn(y_true, y_score)
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return point, lo, hi
