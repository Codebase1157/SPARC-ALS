"""Vowel-Space Graph Attention (VSGA).

Section 4.2 of the manuscript. Uses canonical Bark-scaled F1/F2 positions
(Peterson & Barney 1952) for vowel edges, unit-weight /pa/-/ta/-/ka/
syllable edges, 0.5-weight syllable-to-/a/ edges, and a self-loop on
every node. A learnable severity edge G ∈ R^{8x8} is added before the
softmax. Multi-head attention follows Veličković et al. (2018).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sparc_als.utils.io import load_yaml

TOKEN_ORDER = ("a", "e", "i", "o", "u", "pa", "ta", "ka")
TOKEN_INDEX = {t: i for i, t in enumerate(TOKEN_ORDER)}


def load_graph_priors(path: str | Path) -> dict[str, Any]:
    return load_yaml(path)


def build_edge_matrix(priors: dict[str, Any]) -> torch.Tensor:
    """Returns the fixed log-edge-weight matrix log e_{vu} of shape [8, 8]."""
    e = np.zeros((8, 8), dtype=np.float32)
    # Vowel edges
    positions = {k: np.array(v) for k, v in priors["formant_positions_bark"].items()}
    h = priors["bandwidth_h"]
    vowels = ("a", "e", "i", "o", "u")
    for v in vowels:
        for u in vowels:
            d = np.linalg.norm(positions[v] - positions[u])
            e[TOKEN_INDEX[v], TOKEN_INDEX[u]] = np.exp(-(d ** 2) / (2 * h ** 2))
    # Syllable edges
    for edge in priors["syllable_edges"]:
        i, j, w = TOKEN_INDEX[edge["u"]], TOKEN_INDEX[edge["v"]], edge["weight"]
        e[i, j] = e[j, i] = w
    # Self-loops
    for i in range(8):
        e[i, i] = priors["self_loop_weight"]
    log_e = np.where(e > 0, np.log(e), -1e9)
    return torch.from_numpy(log_e)


class _VSGALayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int,
                 negative_slope: float, edge_matrix: torch.Tensor):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.W = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.a = nn.Parameter(torch.zeros(num_heads, 2 * head_dim))
        nn.init.xavier_uniform_(self.a)
        self.leaky = nn.LeakyReLU(negative_slope)
        self.register_buffer("log_e", edge_matrix)       # [8, 8]
        # Learnable severity edge G, shared across heads of this layer.
        self.G = nn.Parameter(torch.zeros(8, 8))

    def forward(self, Z: torch.Tensor, node_ids: torch.Tensor) -> torch.Tensor:
        # Z: [B, N, dim]; node_ids: [B, N] token indices in {0..7}
        B, N, _ = Z.shape
        H = self.num_heads
        WZ = self.W(Z).view(B, N, H, self.head_dim)         # [B, N, H, Dh]
        # Pair concatenation
        src = WZ.unsqueeze(2).expand(B, N, N, H, self.head_dim)
        dst = WZ.unsqueeze(1).expand(B, N, N, H, self.head_dim)
        pair = torch.cat([src, dst], dim=-1)                # [B, N, N, H, 2Dh]
        chi = self.leaky((pair * self.a).sum(dim=-1))       # [B, N, N, H]
        # Prior and severity edges
        ids_v = node_ids.unsqueeze(2).expand(B, N, N)
        ids_u = node_ids.unsqueeze(1).expand(B, N, N)
        log_e = self.log_e[ids_v, ids_u].unsqueeze(-1)      # [B, N, N, 1]
        G = self.G[ids_v, ids_u].unsqueeze(-1)              # [B, N, N, 1]
        alpha = F.softmax(chi + log_e + G, dim=2)           # softmax over u
        out = (alpha.unsqueeze(-1) * dst).sum(dim=2)        # [B, N, H, Dh]
        out = out.reshape(B, N, H * self.head_dim)          # [B, N, dim]
        return F.elu(out)


class VowelSpaceGraphAttention(nn.Module):
    """Two layers, no parameter sharing between them. Residual via identity
    because K * head_dim == 256 == z dimension."""

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        edge_matrix = build_edge_matrix(cfg)
        dim = cfg["num_heads"] * cfg["head_dim"]
        self.layers = nn.ModuleList([
            _VSGALayer(dim, cfg["num_heads"], cfg["head_dim"],
                       cfg["negative_slope"], edge_matrix)
            for _ in range(cfg["num_layers"])
        ])
        self.layernorms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(cfg["num_layers"])])
        # Readout
        self.W_r = nn.Linear(dim, cfg["readout"]["q_dim"], bias=True)
        self.q = nn.Parameter(torch.zeros(cfg["readout"]["q_dim"]))
        nn.init.normal_(self.q, std=0.02)

    def forward(self, Z: torch.Tensor, node_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (speaker embedding g, attention weights pi)."""
        H = Z
        for layer, ln in zip(self.layers, self.layernorms):
            H = ln(Z + layer(H, node_ids))
            Z = H
        # Attention readout
        logits = torch.tanh(self.W_r(H)) @ self.q           # [B, N]
        pi = F.softmax(logits, dim=-1)                      # [B, N]
        g = (pi.unsqueeze(-1) * H).sum(dim=1)                # [B, dim]
        return g, pi
