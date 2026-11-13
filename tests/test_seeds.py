"""Seeds yield fixture predictions bit-for-bit."""
import numpy as np
import torch

from sparc_als.utils.seeds import RELEASE_SEEDS, set_all_seeds


def test_release_seeds_present():
    assert RELEASE_SEEDS == (11, 23, 42, 77, 2025)


def test_seed_is_deterministic():
    set_all_seeds(11)
    a = np.random.rand(16)
    t_a = torch.randn(16)
    set_all_seeds(11)
    b = np.random.rand(16)
    t_b = torch.randn(16)
    np.testing.assert_array_equal(a, b)
    torch.testing.assert_close(t_a, t_b)
