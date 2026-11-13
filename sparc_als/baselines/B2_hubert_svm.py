"""B2 — HuBERT-Large features + SVM, after Javanmardi et al. (2024)."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.svm import SVC


@torch.no_grad()
def extract_features(hf_id: str, loader, layer: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    from transformers import HubertModel
    enc = HubertModel.from_pretrained(hf_id).to(device).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    X, y = [], []
    for batch in loader:
        wav = batch["wav"].to(device)
        out = enc(wav, output_hidden_states=True)
        h = out.hidden_states[layer]             # [B, F, 1024]
        X.append(h.mean(dim=1).cpu().numpy())
        y.append(batch["label"].numpy())
    return np.concatenate(X), np.concatenate(y)


def train_B2(train_loader, val_loader, *, cfg: dict[str, Any], seed: int, device: str = "cuda"):
    from sparc_als.utils.seeds import set_all_seeds
    set_all_seeds(seed)
    X_tr, y_tr = extract_features(cfg["encoder"]["hf_id"], train_loader,
                                   cfg["feature_layer"], device)
    X_va, y_va = extract_features(cfg["encoder"]["hf_id"], val_loader,
                                   cfg["feature_layer"], device)
    best, best_C, best_gamma = None, None, None
    for C in cfg["hp_search"]["C"]:
        for gamma in cfg["hp_search"]["gamma"]:
            clf = SVC(C=C, gamma=gamma, kernel=cfg["classifier"]["kernel"],
                      class_weight=cfg["classifier"]["class_weight"],
                      probability=True, random_state=seed)
            clf.fit(X_tr, y_tr)
            score = clf.score(X_va, y_va)
            if best is None or score > best:
                best, best_C, best_gamma = score, C, gamma
    final = SVC(C=best_C, gamma=best_gamma,
                kernel=cfg["classifier"]["kernel"],
                class_weight=cfg["classifier"]["class_weight"],
                probability=True, random_state=seed)
    X_all = np.concatenate([X_tr, X_va]); y_all = np.concatenate([y_tr, y_va])
    final.fit(X_all, y_all)
    return final
