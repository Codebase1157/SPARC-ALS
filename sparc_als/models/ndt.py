"""Neuromotor Degradation Tensor (NDT) head and reference statistics.

Section 4.3 of the manuscript: 256 -> 128 -> 4 perceptron predicting
four acoustic proxy axes. The axis mask m is applied before losses.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class ReferenceStats:
    """Per-sex healthy-reference means and SDs for z-scoring (Eq. 5)."""
    male_mean: np.ndarray
    male_sd: np.ndarray
    female_mean: np.ndarray
    female_sd: np.ndarray


class NDTHead(nn.Module):
    """A 256-128-4 perceptron that predicts the four NDT axes."""

    def __init__(self, in_dim: int = 256, hidden: int = 128, out_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, out_dim, bias=True),
        )
        # Uncertainty weighting (Kendall et al. 2018)
        self.log_sigma = nn.Parameter(torch.zeros(out_dim))

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        return self.net(g)


def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    diff = pred - target
    abs_diff = diff.abs()
    quadratic = 0.5 * diff ** 2
    linear = delta * (abs_diff - 0.5 * delta)
    return torch.where(abs_diff <= delta, quadratic, linear)


def uncertainty_weighted(pred: torch.Tensor, target: torch.Tensor,
                         mask: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    """Equation (6)."""
    loss = huber_loss(pred, target)
    scaled = loss / (2 * torch.exp(2 * log_sigma)) + log_sigma
    return (scaled * mask).sum(dim=-1).mean()
