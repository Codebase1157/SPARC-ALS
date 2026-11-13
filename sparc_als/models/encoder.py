"""WavLM-Large encoder, frozen, with learnable per-path layer mixing."""
from __future__ import annotations

import torch
import torch.nn as nn


class LayerMixedWavLM(nn.Module):
    """Frozen WavLM-Large; a softmaxed learnable weight mixes the 24 hidden
    layers. Separate weights are kept for the pathology and nuisance paths
    (Section 4.1)."""

    def __init__(self, hf_id: str = "microsoft/wavlm-large", n_layers: int = 24,
                 hidden_dim: int = 1024, per_path: bool = True,
                 n_paths: int = 2):
        super().__init__()
        from transformers import WavLMModel
        self.wavlm = WavLMModel.from_pretrained(hf_id, output_hidden_states=True)
        for p in self.wavlm.parameters():
            p.requires_grad_(False)
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.per_path = per_path
        # 48 parameters total (2 x 24) in the per-path case
        self.layer_logits = nn.Parameter(torch.zeros(n_paths, n_layers))

    def forward(self, wav: torch.Tensor) -> list[torch.Tensor]:
        """Returns one [B, F, 1024] tensor per path."""
        with torch.no_grad():
            out = self.wavlm(wav)
            # hidden_states: tuple of (n_layers + 1,) [B, F, 1024]
            hs = torch.stack(out.hidden_states[1:], dim=0)  # [L, B, F, D]
        outs = []
        w = torch.softmax(self.layer_logits, dim=-1)        # [n_paths, L]
        for p in range(w.shape[0]):
            mixed = (w[p].view(-1, 1, 1, 1) * hs).sum(dim=0)
            outs.append(mixed)
        return outs
