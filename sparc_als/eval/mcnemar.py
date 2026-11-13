"""Exact McNemar test on speaker decisions (Dietterich 1998)."""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest


def mcnemar_exact_test(y_true: np.ndarray,
                       decision_a: np.ndarray, decision_b: np.ndarray) -> dict[str, float]:
    a_correct = decision_a == y_true
    b_correct = decision_b == y_true
    b01 = int(np.sum(a_correct & ~b_correct))  # A right, B wrong
    b10 = int(np.sum(~a_correct & b_correct))  # A wrong, B right
    n = b01 + b10
    if n == 0:
        return {"b01": 0, "b10": 0, "p_value": 1.0}
    test = binomtest(b01, n, p=0.5, alternative="two-sided")
    return {"b01": b01, "b10": b10, "p_value": float(test.pvalue)}


def holm_correction(p_values: list[float]) -> list[float]:
    """Holm's step-down adjustment (Holm 1979)."""
    n = len(p_values)
    order = np.argsort(p_values)
    adj = np.empty(n)
    running_max = 0.0
    for rank, i in enumerate(order):
        val = (n - rank) * p_values[i]
        running_max = max(running_max, val)
        adj[i] = min(running_max, 1.0)
    return adj.tolist()
