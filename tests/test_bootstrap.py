"""Class-stratified cluster bootstrap on a toy case."""
import numpy as np

from sparc_als.eval.bootstrap import cluster_bootstrap_ci
from sparc_als.eval.metrics import roc_auc


def test_bootstrap_auc_interval_contains_point():
    rng = np.random.default_rng(0)
    y = np.concatenate([np.ones(50), np.zeros(50)])
    s = np.concatenate([rng.normal(1, 1, 50), rng.normal(0, 1, 50)])
    point, lo, hi = cluster_bootstrap_ci(roc_auc, y, s, n_resamples=500, seed=42)
    assert lo <= point <= hi
    assert 0.0 < lo < hi < 1.0
