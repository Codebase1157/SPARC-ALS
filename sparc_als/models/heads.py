"""Detection head, cumulative-logit severity head and FiLM modulation."""
from __future__ import annotations

import torch
import torch.nn as nn


class DetectionHead(nn.Module):
    """Linear logit on [g; y_hat] (Section 4.3).

    l = w_d^T [g; y_hat] + b_d.
    """

    def __init__(self, g_dim: int = 256, aux_dim: int = 4):
        super().__init__()
        self.linear = nn.Linear(g_dim + aux_dim, 1, bias=True)

    def forward(self, g: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.cat([g, y_hat], dim=-1)).squeeze(-1)


class SeverityHead(nn.Module):
    """Cumulative-logit model over R severity thresholds (Section 4.3).

    P(s > r) = sigma(w_s^T g - kappa_r).
    """

    def __init__(self, g_dim: int = 256, n_thresholds: int = 4):
        super().__init__()
        self.w = nn.Linear(g_dim, 1, bias=False)
        # Enforce monotonic thresholds through cumsum of softplus
        self.raw_kappa = nn.Parameter(torch.zeros(n_thresholds))

    def thresholds(self) -> torch.Tensor:
        return torch.cumsum(torch.nn.functional.softplus(self.raw_kappa), dim=0)

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        score = self.w(g)                               # [B, 1]
        return score - self.thresholds().unsqueeze(0)   # [B, R]


class FiLMModulator(nn.Module):
    """[gamma; beta] = W_f (mu_1 - mu_0). Identity in ordinary training."""

    def __init__(self, g_dim: int = 256):
        super().__init__()
        self.W = nn.Linear(g_dim, 2 * g_dim, bias=False)
        self._identity = True

    def set_prototypes(self, mu_pos: torch.Tensor, mu_neg: torch.Tensor):
        delta = mu_pos - mu_neg
        gamma_beta = self.W(delta)
        d = delta.shape[-1]
        self._gamma = 1.0 + gamma_beta[..., :d]         # initialised at identity
        self._beta = gamma_beta[..., d:]
        self._identity = False

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        if self._identity:
            return g
        return self._gamma * g + self._beta
