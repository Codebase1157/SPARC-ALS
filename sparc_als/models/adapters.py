"""Adapter and attentive statistics pooling (Section 4.1)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Adapter(nn.Module):
    """A(H) = H + GELU( DWConv( H W_down ) ) W_up.

    DWConv width 15, GELU activation, bias enabled.
    """

    def __init__(self, in_dim: int = 1024, down_dim: int = 64, conv_width: int = 15):
        super().__init__()
        self.W_down = nn.Linear(in_dim, down_dim, bias=True)
        # Depthwise temporal convolution over F
        self.dwconv = nn.Conv1d(
            in_channels=down_dim, out_channels=down_dim,
            kernel_size=conv_width, padding=conv_width // 2,
            groups=down_dim, bias=True,
        )
        self.W_up = nn.Linear(down_dim, in_dim, bias=True)
        self.activation = nn.GELU()

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        # H: [B, F, D]
        h = self.W_down(H)                      # [B, F, 64]
        h = h.transpose(1, 2)                   # [B, 64, F]
        h = self.dwconv(h)
        h = h.transpose(1, 2)                   # [B, F, 64]
        h = self.activation(h)
        h = self.W_up(h)                        # [B, F, D]
        return H + h


class AttentiveStatsPooling(nn.Module):
    """Attentive statistics pooling (Okabe et al. 2018) with explicit
    projection to 256 and a token-type embedding."""

    def __init__(self, in_dim: int = 1024, hidden_dim: int = 128,
                 out_dim: int = 256, n_token_types: int = 8):
        super().__init__()
        self.W_a = nn.Linear(in_dim, hidden_dim, bias=True)
        self.v = nn.Parameter(torch.zeros(hidden_dim))
        self.W_o = nn.Linear(2 * in_dim, out_dim, bias=True)
        self.layernorm = nn.LayerNorm(out_dim)
        self.token_emb = nn.Embedding(n_token_types, out_dim)
        nn.init.normal_(self.v, std=0.02)
        nn.init.normal_(self.token_emb.weight, std=0.02)

    def forward(self, H: torch.Tensor, token_type_id: torch.Tensor) -> torch.Tensor:
        # H: [B, F, D]; token_type_id: [B]
        attn = torch.tanh(self.W_a(H))                 # [B, F, hidden]
        logits = (attn @ self.v).unsqueeze(-1)         # [B, F, 1]
        beta = F.softmax(logits, dim=1)                # [B, F, 1]
        mu = (beta * H).sum(dim=1)                     # [B, D]
        sigma = ((beta * (H - mu.unsqueeze(1)) ** 2).sum(dim=1) + 1e-6).sqrt()
        stats = torch.cat([mu, sigma], dim=-1)         # [B, 2D]
        z = self.W_o(stats)                            # [B, out]
        z = self.layernorm(z)
        z = z + self.token_emb(token_type_id)
        return z
