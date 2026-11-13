"""B6 — Prototypical network, after Bharathi Sri & Somasundaram (2026)."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from sparc_als.models.adapters import Adapter, AttentiveStatsPooling


class B6(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        from transformers import WavLMModel
        self.enc = WavLMModel.from_pretrained(cfg["encoder"]["hf_id"])
        for p in self.enc.parameters():
            p.requires_grad_(False)
        self.adapter = Adapter(1024, cfg["adapter"]["hidden_down"],
                               cfg["adapter"]["depthwise_conv_width"])
        self.pool = AttentiveStatsPooling(1024, 128, cfg["embedding_dim"], n_token_types=8)

    def embed(self, wav: torch.Tensor, token_id: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h = self.enc(wav).last_hidden_state
        h = self.adapter(h)
        return self.pool(h, token_id)


def prototypical_loss(z: torch.Tensor, labels: torch.Tensor, n_support: int,
                       temperature: float = 1.0) -> torch.Tensor:
    pos = z[labels == 1]
    neg = z[labels == 0]
    sup_p, qry_p = pos[:n_support], pos[n_support:]
    sup_n, qry_n = neg[:n_support], neg[n_support:]
    proto = torch.stack([sup_n.mean(0), sup_p.mean(0)], dim=0)   # [2, D]
    queries = torch.cat([qry_n, qry_p], dim=0)
    labels_q = torch.cat([torch.zeros(len(qry_n)), torch.ones(len(qry_p))]).long().to(z.device)
    dists = torch.cdist(queries, proto)
    logits = -dists / temperature
    return F.cross_entropy(logits, labels_q)


def train_B6(train_loader, val_loader, *, cfg: dict[str, Any], seed: int,
             device: str = "cuda", checkpoint_path: str | None = None):
    from sparc_als.utils.seeds import set_all_seeds
    set_all_seeds(seed)
    model = B6(cfg).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                             lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"])
    for _ in range(cfg["training"]["max_epochs"]):
        model.train()
        for episode in train_loader:
            z = model.embed(episode["wav"].to(device), episode["token_id"].to(device))
            loss = prototypical_loss(z, episode["label"].to(device),
                                     n_support=cfg["episodes"]["n_support_per_class"],
                                     temperature=cfg["temperature"])
            opt.zero_grad(); loss.backward(); opt.step()
    return model
