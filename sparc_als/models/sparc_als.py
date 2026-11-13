"""Top-level SPARC-ALS model (Stages 1 to 6 combined)."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .adapters import Adapter, AttentiveStatsPooling
from .encoder import LayerMixedWavLM
from .heads import DetectionHead, FiLMModulator, SeverityHead
from .ndt import NDTHead
from .vsga import VowelSpaceGraphAttention


class SparcALS(nn.Module):
    """Full SPARC-ALS model."""

    def __init__(self, cfg_training: dict[str, Any], cfg_graph: dict[str, Any]):
        super().__init__()
        # Stage 1: Encoder
        enc_cfg = cfg_training["encoder"]
        self.encoder = LayerMixedWavLM(
            hf_id=enc_cfg["hf_id"],
            n_layers=enc_cfg["layers"],
            hidden_dim=enc_cfg["hidden_dim"],
            per_path=enc_cfg["layer_mixing_per_path"],
            n_paths=2,
        )

        # Stage 2: Dual adapters
        ad = cfg_training["adapter"]
        self.pathology_adapter = Adapter(
            in_dim=enc_cfg["hidden_dim"],
            down_dim=ad["hidden_down"],
            conv_width=ad["depthwise_conv_width"],
        )
        self.nuisance_adapter = Adapter(
            in_dim=enc_cfg["hidden_dim"],
            down_dim=ad["hidden_down"],
            conv_width=ad["depthwise_conv_width"],
        )

        # Attentive statistics pooling (shared across paths' configs,
        # independent weights)
        asp = cfg_training["asp"]
        self.pool_p = AttentiveStatsPooling(
            in_dim=enc_cfg["hidden_dim"],
            hidden_dim=asp["hidden_dim"],
            out_dim=asp["output_dim"],
            n_token_types=8,
        )
        self.pool_n = AttentiveStatsPooling(
            in_dim=enc_cfg["hidden_dim"],
            hidden_dim=asp["hidden_dim"],
            out_dim=asp["output_dim"],
            n_token_types=8,
        )

        # Stage 3: VSGA
        self.vsga = VowelSpaceGraphAttention(cfg_graph)

        # Stage 4: NDT head
        self.ndt_head = NDTHead(in_dim=asp["output_dim"], hidden=128, out_dim=4)

        # FiLM for meta-adaptation (Stage 6)
        self.film = FiLMModulator(g_dim=asp["output_dim"])

        # Heads
        self.detection_head = DetectionHead(g_dim=asp["output_dim"], aux_dim=4)
        self.severity_head = SeverityHead(g_dim=asp["output_dim"], n_thresholds=4)

        # Training-only adversarial / nuisance classifiers are created dynamically
        # in sparc_als.training.train, because the speaker class count depends
        # on the fold.

    def forward_tokens(self, wav: torch.Tensor, token_ids: torch.Tensor):
        """Encode a batch of tokens through both paths.

        Returns speaker-level embeddings z_p, z_n of shape [B, 256].
        """
        H_p, H_n = self.encoder(wav)
        H_p = self.pathology_adapter(H_p)
        H_n = self.nuisance_adapter(H_n)
        z_p = self.pool_p(H_p, token_ids)
        z_n = self.pool_n(H_n, token_ids)
        return z_p, z_n

    def speaker_forward(self, z_p_per_token: torch.Tensor,
                        token_ids_per_speaker: torch.Tensor,
                        node_mask: torch.Tensor | None = None):
        """Combine per-token z_p into a speaker embedding g via VSGA."""
        g, pi = self.vsga(z_p_per_token, token_ids_per_speaker)
        g = self.film(g)
        y_hat = self.ndt_head(g)
        logit = self.detection_head(g, y_hat)
        severity = self.severity_head(g)
        return {
            "g": g, "pi": pi, "y_hat": y_hat,
            "logit": logit, "severity": severity,
        }
