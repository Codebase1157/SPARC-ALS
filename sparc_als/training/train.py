"""Per-fold training entry point for SPARC-ALS.

This is the function called by scripts/E1_within_corpus.py once per
LOSO fold and seed. The full objective is

  L = L_det + lambda_1 * L_NDT + lambda_2 * L_dis + lambda_3 * L_cf + lambda_4 * L_ord,

with L_dis = L_sup + psi * (L_adv + L_nuis) + zeta * || Cov(z_p, z_n) ||_F^2.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from sparc_als.models.ndt import uncertainty_weighted
from sparc_als.models.sparc_als import SparcALS
from sparc_als.training.counterfactuals import CounterfactualGenerator
from sparc_als.training.losses import (
    adversarial_loss,
    cf_consistency_loss,
    cross_covariance_penalty,
    cumulative_logit_loss,
    grl_lambda,
    sup_con_loss,
)
from sparc_als.utils.seeds import set_all_seeds


def build_adversaries(embed_dim: int, n_speakers: int, n_corpora: int = 3) -> dict[str, nn.Module]:
    mk = lambda out: nn.Sequential(
        nn.Linear(embed_dim, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, out)
    )
    return {
        "spk_p":  mk(n_speakers),
        "dom_p":  mk(n_corpora),
        "spk_n":  mk(n_speakers),
        "dom_n":  mk(n_corpora),
    }


def train_fold(
    model: SparcALS,
    train_loader,
    inner_val_loader,
    *,
    cfg_training: dict[str, Any],
    cfg_cf: dict[str, Any] | None,
    seed: int,
    device: str = "cuda",
    experiment: str = "E1",
    log_dir: str = "runs",
    checkpoint_path: str,
) -> dict[str, Any]:
    """Train one fold with the full objective. Returns the best-checkpoint state."""
    set_all_seeds(seed)
    model.to(device)

    opt_cfg = cfg_training["optim"]
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"],
        betas=tuple(opt_cfg["betas"]), eps=opt_cfg["eps"],
    )

    cf_gen = CounterfactualGenerator("configs/counterfactuals.yaml", rng_seed=seed) if cfg_cf else None
    adversaries = build_adversaries(
        embed_dim=cfg_training["asp"]["output_dim"],
        n_speakers=cfg_training.get("_n_speakers_in_fold", 32),
        n_corpora=3,
    )
    for m in adversaries.values():
        m.to(device)

    best_val = float("inf")
    patience_ctr = 0
    full = cfg_training["full_objective"]
    psi = cfg_training["disentangle"]["psi"]
    zeta = cfg_training["disentangle"]["zeta"]
    tau = cfg_training["sup_con"]["temperature_tau"]

    for epoch in range(1, cfg_training["training_loop"]["max_epochs"] + 1):
        model.train()
        progress = (epoch - 1) / cfg_training["training_loop"]["max_epochs"]
        lambda_rev = grl_lambda(progress)

        for batch in train_loader:
            wav = batch["wav"].to(device)
            token_ids = batch["token_id"].to(device)
            labels = batch["label"].to(device)
            severity = batch["severity"].to(device)
            speaker_ids = batch["speaker_id_int"].to(device)
            corpus_ids = batch["corpus_id_int"].to(device)
            ndt_targets = batch["ndt_target"].to(device)
            ndt_mask = batch["ndt_mask"].to(device)

            z_p, z_n = model.forward_tokens(wav, token_ids)

            # Speaker-level graph forward (expects per-speaker token packs).
            # For brevity assume loader provides the per-speaker tensor:
            out = model.speaker_forward(
                z_p_per_token=batch["z_pack"].to(device),
                token_ids_per_speaker=batch["token_pack"].to(device),
            )
            logit = out["logit"]
            severity_logits = out["severity"]
            y_hat = out["y_hat"]

            # --- Detection ---
            L_det = F.binary_cross_entropy_with_logits(logit, labels.float())

            # --- NDT (Eq. 6) ---
            L_ndt = uncertainty_weighted(y_hat, ndt_targets, ndt_mask,
                                         model.ndt_head.log_sigma)

            # --- Disentanglement ---
            L_sup = sup_con_loss(z_p, labels, severity, speaker_ids, tau=tau)
            L_adv = adversarial_loss(adversaries["spk_p"], z_p, speaker_ids, lambda_rev)
            if (corpus_ids != corpus_ids[0]).any():
                L_adv = L_adv + adversarial_loss(adversaries["dom_p"], z_p, corpus_ids, lambda_rev)
            L_nuis = F.cross_entropy(adversaries["spk_n"](z_n), speaker_ids)
            if (corpus_ids != corpus_ids[0]).any():
                L_nuis = L_nuis + F.cross_entropy(adversaries["dom_n"](z_n), corpus_ids)
            L_cov = cross_covariance_penalty(z_p, z_n)
            L_dis = L_sup + psi * (L_adv + L_nuis) + zeta * L_cov

            # --- Counterfactual consistency ---
            L_cf = logit.new_zeros(())
            if cf_gen is not None and "z_pack_cf" in batch:
                out_cf = model.speaker_forward(
                    z_p_per_token=batch["z_pack_cf"].to(device),
                    token_ids_per_speaker=batch["token_pack"].to(device),
                )
                delta = batch["delta"].to(device)
                L_cf = cf_consistency_loss(out_cf["y_hat"], y_hat, delta, ndt_mask)

            # --- Ordinal severity ---
            L_ord = cumulative_logit_loss(severity_logits, severity)

            loss = (L_det
                    + full["lambda_1_ndt"] * L_ndt
                    + full["lambda_2_dis"] * L_dis
                    + full["lambda_3_cf"] * L_cf
                    + full["lambda_4_ord"] * L_ord)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt_cfg["grad_clip_norm"])
            optim.step()

        # Inner-validation loss
        model.eval()
        with torch.no_grad():
            val_losses = []
            for batch in inner_val_loader:
                wav = batch["wav"].to(device)
                token_ids = batch["token_id"].to(device)
                labels = batch["label"].to(device).float()
                z_p, _ = model.forward_tokens(wav, token_ids)
                out = model.speaker_forward(
                    z_p_per_token=batch["z_pack"].to(device),
                    token_ids_per_speaker=batch["token_pack"].to(device),
                )
                val_losses.append(F.binary_cross_entropy_with_logits(out["logit"], labels).item())
            val_loss = sum(val_losses) / max(1, len(val_losses))

        if val_loss < best_val:
            best_val = val_loss
            patience_ctr = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": best_val},
                       checkpoint_path)
        else:
            patience_ctr += 1
            if patience_ctr >= cfg_training["training_loop"]["patience"]:
                break

    return {"best_val": best_val, "checkpoint_path": checkpoint_path}
