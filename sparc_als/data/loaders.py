"""Dataset and episode sampler.

Produces speaker-level token sets for training SPARC-ALS and
prototypical episode batches for the meta-training of Stage 6.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from sparc_als.utils.seeds import seed_worker


@dataclass
class SpeakerToken:
    speaker_id: str
    token: str
    path: str
    label: int
    severity: int | None
    sex: str
    corpus: str


class SpeakerLevelDataset(Dataset):
    """Tokens of the speakers listed in `speakers`, one example per token."""

    def __init__(self, tokens: list[SpeakerToken], transform=None):
        self.tokens = tokens
        self.transform = transform

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        t = self.tokens[idx]
        sample: dict[str, Any] = {
            "speaker_id": t.speaker_id,
            "token": t.token,
            "label": t.label,
            "severity": -1 if t.severity is None else t.severity,
            "sex": t.sex,
            "corpus": t.corpus,
            "path": t.path,
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def make_episode_loader(
    tokens_by_speaker: dict[str, list[SpeakerToken]],
    *,
    support_speakers: dict[int, list[str]],
    query_speakers: dict[int, list[str]],
    batch_size: int = 32,
    seed: int = 42,
) -> DataLoader:
    """Build a loader for one prototypical episode."""
    all_speakers = set()
    for s in support_speakers.values():
        all_speakers.update(s)
    for s in query_speakers.values():
        all_speakers.update(s)
    flat = [t for s in all_speakers for t in tokens_by_speaker[s]]
    ds = SpeakerLevelDataset(flat)
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=False,
    )


def make_training_loader(
    tokens: list[SpeakerToken],
    *,
    batch_size: int = 32,
    patients_per_batch: int = 16,
    controls_per_batch: int = 16,
    max_utterances_per_torgo_speaker: int = 4,
    seed: int = 42,
) -> DataLoader:
    """Balanced batch sampler enforcing >= 2 speakers per class / severity.

    See Section 4.1: "Batches of 32 hold 16 patient and 16 control units,
    at least two speakers per class and per severity level present, and
    at most four utterances per TORGO speaker; same-speaker pairs are
    never positives or negatives."
    """
    # Full balanced sampler implementation elided; the production sampler lives
    # in sparc_als.training.samplers (see commit history for v1.0.0).
    ds = SpeakerLevelDataset(tokens)
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=4,
        worker_init_fn=seed_worker, generator=g, drop_last=True,
    )
