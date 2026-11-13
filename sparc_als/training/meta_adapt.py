"""Prototype-conditioned meta-adaptation (Stage 6, Section 4.5).

Meta-training optimises
   theta_meta = arg min E_{S, Q} [ L_Q(theta - eta * grad L_S(theta)) ]
with three first-order inner steps on the support loss, prototype FiLM
modulation and a prototype-based detection correction.

At deployment, k support speakers per class adapt theta with the same
three steps on the DETECTION loss only, using support labels only,
never axis targets, and without refitting the temperature.
"""
from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_prototypes(model, support_batch, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    z_p, _ = model.forward_tokens(support_batch["wav"].to(device),
                                  support_batch["token_id"].to(device))
    out = model.speaker_forward(
        z_p_per_token=support_batch["z_pack"].to(device),
        token_ids_per_speaker=support_batch["token_pack"].to(device),
    )
    g = out["g"]
    labels = support_batch["label"].to(device)
    mu_pos = g[labels == 1].mean(dim=0)
    mu_neg = g[labels == 0].mean(dim=0)
    return mu_pos, mu_neg


def meta_train(model, episodes, *, inner_steps: int = 3, inner_lr: float = 0.01,
               tau_p: float = 1.0, device: str = "cuda") -> None:
    """One meta-epoch over `episodes`."""
    model.train()
    # First-order MAML: theta_meta = arg min E_{S,Q}[ L_Q(theta - eta * grad L_S(theta)) ].
    for episode in episodes:
        # Inner loop on S
        fast_weights = {n: p for n, p in model.named_parameters() if p.requires_grad}
        for _ in range(inner_steps):
            loss_s = episode_loss(model, episode["support"], device, tau_p=tau_p)
            grads = torch.autograd.grad(loss_s, list(fast_weights.values()),
                                        create_graph=False, allow_unused=True)
            fast_weights = {
                n: (p - inner_lr * (g if g is not None else torch.zeros_like(p)))
                for (n, p), g in zip(fast_weights.items(), grads)
            }
        # Outer step on Q using fast_weights
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in fast_weights:
                    p.copy_(fast_weights[n])
        loss_q = episode_loss(model, episode["query"], device, tau_p=tau_p)
        loss_q.backward()


def episode_loss(model, batch, device: str, *, tau_p: float = 1.0) -> torch.Tensor:
    wav = batch["wav"].to(device)
    token_ids = batch["token_id"].to(device)
    labels = batch["label"].to(device).float()
    z_p, _ = model.forward_tokens(wav, token_ids)
    out = model.speaker_forward(
        z_p_per_token=batch["z_pack"].to(device),
        token_ids_per_speaker=batch["token_pack"].to(device),
    )
    g = out["g"]
    mu_pos = batch["mu_pos"].to(device)
    mu_neg = batch["mu_neg"].to(device)
    dist_pos = (g - mu_pos).pow(2).sum(dim=-1)
    dist_neg = (g - mu_neg).pow(2).sum(dim=-1)
    logit = out["logit"] + (dist_neg - dist_pos) / tau_p
    return F.binary_cross_entropy_with_logits(logit, labels)


def adapt_support(model, support_batch, *, inner_steps: int = 3,
                  inner_lr: float = 0.01, device: str = "cuda"):
    """Deployment adaptation: three inner steps on detection loss only."""
    model = copy.deepcopy(model)
    model.train()
    opt = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=inner_lr
    )
    for _ in range(inner_steps):
        wav = support_batch["wav"].to(device)
        token_ids = support_batch["token_id"].to(device)
        labels = support_batch["label"].to(device).float()
        z_p, _ = model.forward_tokens(wav, token_ids)
        out = model.speaker_forward(
            z_p_per_token=support_batch["z_pack"].to(device),
            token_ids_per_speaker=support_batch["token_pack"].to(device),
        )
        loss = F.binary_cross_entropy_with_logits(out["logit"], labels)
        opt.zero_grad(); loss.backward(); opt.step()
    return model
