"""Losses used by SPARC-ALS (Section 4)."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def sup_con_loss(z: torch.Tensor, labels: torch.Tensor, severity: torch.Tensor,
                 speaker_ids: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """Supervised contrastive loss (Equation 2).

    An NA-severity patient (severity == -1) is a positive of every patient;
    anchors without positives are excluded from I.
    """
    z = F.normalize(z, dim=-1)
    sim = (z @ z.t()) / tau                                    # [B, B]
    B = z.shape[0]
    device = z.device
    eye = torch.eye(B, dtype=torch.bool, device=device)
    same_speaker = speaker_ids.unsqueeze(0) == speaker_ids.unsqueeze(1)
    mask_other = (~eye) & (~same_speaker)
    sim = sim.masked_fill(eye | same_speaker, -1e9)
    # Positives: same label and matching severity (NA matches any severity of same label)
    lab_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
    sev_eq = (severity.unsqueeze(0) == severity.unsqueeze(1)) \
            | (severity.unsqueeze(0) == -1) \
            | (severity.unsqueeze(1) == -1)
    pos = lab_eq & sev_eq & mask_other
    # Rows without positives are excluded
    row_has_pos = pos.any(dim=1)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    mean_log_prob_pos = (pos.float() * log_prob).sum(dim=1) \
                       / pos.sum(dim=1).clamp_min(1)
    loss = -mean_log_prob_pos[row_has_pos].mean()
    return loss


def grl_lambda(progress: float) -> float:
    """lambda_rev = 2 / (1 + exp(-10 * progress)) - 1 (Ganin et al. 2016)."""
    return 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


def gradient_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return _GradReverse.apply(x, lambd)


def adversarial_loss(classifier: torch.nn.Module, z: torch.Tensor,
                     labels: torch.Tensor, lambd: float) -> torch.Tensor:
    return F.cross_entropy(classifier(gradient_reverse(z, lambd)), labels)


def cross_covariance_penalty(z_p: torch.Tensor, z_n: torch.Tensor) -> torch.Tensor:
    """||Cov(z_p, z_n)||_F^2."""
    z_p = z_p - z_p.mean(dim=0, keepdim=True)
    z_n = z_n - z_n.mean(dim=0, keepdim=True)
    cov = z_p.t() @ z_n / max(1, z_p.shape[0] - 1)
    return (cov ** 2).sum()


def cf_consistency_loss(y_hat_cf: torch.Tensor, y_hat_orig: torch.Tensor,
                        delta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """L_cf = ||m ⊙ (y_hat(x_tilde) - y_hat(x) - delta)||_2^2."""
    err = mask * (y_hat_cf - y_hat_orig - delta)
    return (err ** 2).sum(dim=-1).mean()


def cumulative_logit_loss(logits: torch.Tensor, severity: torch.Tensor,
                          ignore_index: int = -1) -> torch.Tensor:
    """Ordinal cumulative-logit loss over R thresholds."""
    valid = severity != ignore_index
    if not valid.any():
        return logits.new_zeros(())
    logits = logits[valid]                                 # [n, R]
    severity = severity[valid]
    # Threshold-wise binary cross-entropy: y > r
    R = logits.shape[1]
    targets = (severity.unsqueeze(1) > torch.arange(R, device=logits.device).unsqueeze(0)).float()
    return F.binary_cross_entropy_with_logits(logits, targets)
