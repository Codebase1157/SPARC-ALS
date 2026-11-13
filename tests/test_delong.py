"""DeLong test on a toy case."""
import numpy as np

from sparc_als.eval.delong import delong_test


def test_delong_identical_scores_pvalue_one():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=200)
    s = rng.random(200)
    out = delong_test(y, s, s.copy())
    assert abs(out["delta"]) < 1e-12
    assert out["p_value"] > 0.99


def test_delong_separated_scores_pvalue_small():
    rng = np.random.default_rng(0)
    y = np.concatenate([np.ones(100), np.zeros(100)])
    s_a = np.concatenate([rng.normal(1, 1, 100), rng.normal(0, 1, 100)])
    s_b = np.concatenate([rng.normal(0.1, 1, 100), rng.normal(0, 1, 100)])
    out = delong_test(y, s_a, s_b)
    assert out["p_value"] < 0.05
