"""B4 — CNN-BiLSTM with additive attention, after Louja et al. (2025)."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class B4(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        chans = cfg["cnn"]["channels"]
        layers = []
        in_ch = 1
        for i, c in enumerate(chans):
            layers.append(nn.Conv2d(in_ch, c, kernel_size=tuple(cfg["cnn"]["kernels"][i]),
                                     stride=tuple(cfg["cnn"]["strides"][i]), padding=1))
            layers.append(nn.BatchNorm2d(c))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout2d(cfg["cnn"]["dropout"]))
            in_ch = c
        self.cnn = nn.Sequential(*layers)
        self.bilstm = nn.LSTM(input_size=chans[-1] * (cfg["input"]["n_mels"] // (2 ** len(chans))),
                              hidden_size=cfg["bilstm"]["hidden"],
                              num_layers=cfg["bilstm"]["num_layers"],
                              batch_first=True, bidirectional=True,
                              dropout=cfg["bilstm"]["dropout"])
        attn_in = 2 * cfg["bilstm"]["hidden"]
        self.attn = nn.Linear(attn_in, cfg["attention"]["hidden"])
        self.attn_v = nn.Parameter(torch.zeros(cfg["attention"]["hidden"]))
        nn.init.normal_(self.attn_v, std=0.02)
        self.head = nn.Linear(attn_in, 1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = mel.unsqueeze(1)                        # [B, 1, M, T]
        x = self.cnn(x)                             # [B, C, M', T']
        B, C, M, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * M)
        h, _ = self.bilstm(x)                       # [B, T, 2H]
        logits = torch.tanh(self.attn(h)) @ self.attn_v
        beta = F.softmax(logits, dim=1).unsqueeze(-1)
        g = (beta * h).sum(dim=1)
        return self.head(g).squeeze(-1)


def train_B4(train_loader, val_loader, *, cfg: dict[str, Any], seed: int,
             device: str = "cuda", checkpoint_path: str | None = None):
    from sparc_als.utils.seeds import set_all_seeds
    set_all_seeds(seed)
    model = B4(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["optim"]["lr"])
    best = float("inf")
    for _ in range(cfg["training"]["max_epochs"]):
        model.train()
        for batch in train_loader:
            mel = batch["log_mel"].to(device)
            y = batch["label"].to(device).float()
            logit = model(mel)
            loss = F.binary_cross_entropy_with_logits(logit, y)
            opt.zero_grad(); loss.backward(); opt.step()
    return model
