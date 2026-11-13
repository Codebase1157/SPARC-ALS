"""Checkpoint-selection rule (Section 4.5 and 5).

Lowest inner-validation loss over at most 60 epochs with patience 10.
In E4 only, the checkpoint is chosen on 200 fixed source
inner-validation episodes.
"""
from __future__ import annotations

from pathlib import Path


def checkpoint_dir(experiment: str, corpus: str, seed: int,
                   root: str | Path = "checkpoints") -> Path:
    d = Path(root) / experiment / corpus / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_best_marker(path: Path, epoch: int, score: float) -> None:
    with open(path / "BEST", "w", encoding="utf-8") as f:
        f.write(f"epoch={epoch}\nscore={score}\n")
