"""Deterministic seeding for SPARC-ALS.

Every stochastic source — numpy, torch, torch.cuda, python random,
dataloader workers, dropout and counterfactual sampling — is seeded
from one of the five release seeds {11, 23, 42, 77, 2025}.

All released per-speaker / per-utterance predictions carry one of
these values as `seed_id`, so paired tests on the seed-averaged
logit l̄_i = (1/5) Σ_s l_{i,s} are exactly reproducible.
"""
from __future__ import annotations

import os
import random
from typing import Iterable

import numpy as np

RELEASE_SEEDS: tuple[int, ...] = (11, 23, 42, 77, 2025)


def set_all_seeds(seed: int, *, deterministic: bool = True) -> None:
    """Seed every stochastic source."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # PyTorch >= 1.11
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
    except ImportError:
        pass


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn ensuring reproducibility across workers."""
    import torch

    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def validate_seed(seed: int) -> int:
    """Reject seeds not in the release set."""
    if seed not in RELEASE_SEEDS:
        raise ValueError(
            f"Seed {seed} is not a release seed; expected one of {RELEASE_SEEDS}."
        )
    return seed


def iter_release_seeds() -> Iterable[int]:
    return iter(RELEASE_SEEDS)
