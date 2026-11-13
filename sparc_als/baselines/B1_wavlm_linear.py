"""B1 — frozen WavLM-Large + linear probe (~1 k parameters)."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class B1(nn.Module):
    def __init__(self, hf_id: str = "microsoft/wavlm-large"):
        super().__init__()
        from transformers import WavLMModel
        self.enc = WavLMModel.from_pretrained(hf_id)
        for p in self.enc.parameters():
            p.requires_grad_(False)
        self.head = nn.Linear(1024, 1)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h = self.enc(wav).last_hidden_state  # [B, F, 1024]
        g = h.mean(dim=1)                        # [B, 1024]
        return self.head(g).squeeze(-1)


def train_B1(train_loader, val_loader, *, cfg: dict[str, Any], seed: int,
             device: str = "cuda", checkpoint_path: str | None = None):
    from sparc_als.utils.seeds import set_all_seeds
    set_all_seeds(seed)
    model = B1(hf_id=cfg["encoder"]["hf_id"]).to(device)
    opt = torch.optim.AdamW(model.head.parameters(),
                            lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"])
    best = float("inf")
    for _ in range(cfg["training"]["max_epochs"]):
        model.train()
        for batch in train_loader:
            wav = batch["wav"].to(device)
            y = batch["label"].to(device).float()
            logit = model(wav)
            loss = F.binary_cross_entropy_with_logits(logit, y)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            losses = [F.binary_cross_entropy_with_logits(model(b["wav"].to(device)),
                                                          b["label"].to(device).float()).item()
                      for b in val_loader]
            val = sum(losses) / max(1, len(losses))
        if val < best:
            best = val
            if checkpoint_path:
                torch.save(model.state_dict(), checkpoint_path)
    return model
