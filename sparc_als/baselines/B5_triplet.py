"""B5 — Triplet regularizer on WavLM, after Kadirvelu et al. (2025).

Positive: word of DIFFERENT speaker at the SAME severity.
Negative: word of the SAME speaker.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from sparc_als.models.adapters import Adapter, AttentiveStatsPooling


class B5(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        from transformers import WavLMModel
        self.enc = WavLMModel.from_pretrained(cfg["encoder"]["hf_id"])
        for p in self.enc.parameters():
            p.requires_grad_(False)
        self.adapter = Adapter(1024, cfg["adapter"]["hidden_down"],
                               cfg["adapter"]["depthwise_conv_width"])
        self.pool = AttentiveStatsPooling(1024, 128, cfg["embedding_dim"], n_token_types=8)
        self.head = nn.Linear(cfg["embedding_dim"], 1)

    def embed(self, wav: torch.Tensor, token_id: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h = self.enc(wav).last_hidden_state
        h = self.adapter(h)
        return self.pool(h, token_id)

    def forward(self, wav: torch.Tensor, token_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.embed(wav, token_id)
        return z, self.head(z).squeeze(-1)


def triplet_loss(z: torch.Tensor, speaker_ids: torch.Tensor, severity: torch.Tensor,
                 margin: float = 0.2) -> torch.Tensor:
    """Semi-hard triplet: positive is different speaker at same severity,
    negative is same speaker."""
    d = torch.cdist(z, z)                              # [B, B]
    pos = (speaker_ids.unsqueeze(0) != speaker_ids.unsqueeze(1)) \
         & (severity.unsqueeze(0) == severity.unsqueeze(1))
    neg = speaker_ids.unsqueeze(0) == speaker_ids.unsqueeze(1)
    neg = neg & ~torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    loss = 0.0
    n = 0
    for i in range(z.shape[0]):
        pi = pos[i].nonzero(as_tuple=True)[0]
        ni = neg[i].nonzero(as_tuple=True)[0]
        if len(pi) == 0 or len(ni) == 0:
            continue
        d_ap = d[i, pi].min()
        d_an = d[i, ni].max()
        loss = loss + F.relu(d_ap - d_an + margin)
        n += 1
    return loss / max(1, n)


def train_B5(train_loader, val_loader, *, cfg: dict[str, Any], seed: int,
             device: str = "cuda", checkpoint_path: str | None = None):
    from sparc_als.utils.seeds import set_all_seeds
    set_all_seeds(seed)
    model = B5(cfg).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                             lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"])
    for _ in range(cfg["training"]["max_epochs"]):
        model.train()
        for batch in train_loader:
            z, logit = model(batch["wav"].to(device), batch["token_id"].to(device))
            y = batch["label"].to(device).float()
            L_det = F.binary_cross_entropy_with_logits(logit, y)
            L_tri = triplet_loss(z, batch["speaker_id_int"].to(device),
                                 batch["severity"].to(device), cfg["triplet"]["margin"])
            loss = cfg["loss_weights"]["detection"] * L_det + cfg["loss_weights"]["triplet"] * L_tri
            opt.zero_grad(); loss.backward(); opt.step()
    return model
