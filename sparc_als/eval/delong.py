"""DeLong test for paired AUCs (DeLong et al. 1988)."""
from __future__ import annotations

import numpy as np
from scipy import stats


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sx = x[order]
    n = len(x)
    T = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and sx[j] == sx[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(n)
    T2[order] = T
    return T2


def _structural_components(y_true: np.ndarray, y_score: np.ndarray):
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    m, n = len(pos), len(neg)
    tx = _compute_midrank(pos)
    ty = _compute_midrank(neg)
    tz = _compute_midrank(np.concatenate([pos, neg]))
    auc = (tz[:m].sum() / (m * n)) - (m + 1) / (2 * n)
    v10 = (tz[:m] - tx) / n
    v01 = 1.0 - (tz[m:] - ty) / m
    return auc, v10, v01


def delong_test(y_true: np.ndarray,
                score_a: np.ndarray, score_b: np.ndarray) -> dict[str, float]:
    """Returns (auc_a, auc_b, delta, z, p_value)."""
    auc_a, v10_a, v01_a = _structural_components(y_true, score_a)
    auc_b, v10_b, v01_b = _structural_components(y_true, score_b)
    m = (y_true == 1).sum()
    n = (y_true == 0).sum()
    var_a = (np.var(v10_a, ddof=1) / m) + (np.var(v01_a, ddof=1) / n)
    var_b = (np.var(v10_b, ddof=1) / m) + (np.var(v01_b, ddof=1) / n)
    cov = (np.cov(v10_a, v10_b, ddof=1)[0, 1] / m
           + np.cov(v01_a, v01_b, ddof=1)[0, 1] / n)
    var_diff = var_a + var_b - 2 * cov
    delta = auc_a - auc_b
    z = delta / np.sqrt(max(var_diff, 1e-12))
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"auc_a": float(auc_a), "auc_b": float(auc_b),
            "delta": float(delta), "z": float(z), "p_value": float(p)}
