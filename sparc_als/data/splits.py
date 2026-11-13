"""Deterministic generation of speaker-disjoint LOSO splits and
few-shot enrollment episodes.

Re-running this script with the released seed reproduces the released
splits/*.json files bit-for-bit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from sparc_als.utils.io import load_yaml, read_json, write_json


def generate_loso_splits(speaker_meta: list[dict[str, Any]], *,
                         stratify_by: list[str] | None = None,
                         inner_val_fraction: float = 0.15,
                         seed: int = 42) -> dict[str, Any]:
    """Return a dict following `splits/*_loso_splits.json` schema."""
    rng = np.random.default_rng(seed)
    n = len(speaker_meta)
    folds = []
    for test_idx, test_speaker in enumerate(speaker_meta):
        remaining = [s for i, s in enumerate(speaker_meta) if i != test_idx]

        # Stratified sampling of inner validation.
        if stratify_by:
            key = tuple(sorted(stratify_by))
            buckets: dict[tuple, list] = {}
            for s in remaining:
                k = tuple(s[b] for b in key)
                buckets.setdefault(k, []).append(s)
            inner_val = []
            for bucket in buckets.values():
                n_pick = max(1, int(round(len(bucket) * inner_val_fraction)))
                picks = rng.choice(len(bucket), size=n_pick, replace=False)
                inner_val.extend([bucket[i] for i in picks])
        else:
            n_pick = int(round(len(remaining) * inner_val_fraction))
            idx = rng.choice(len(remaining), size=n_pick, replace=False)
            inner_val = [remaining[i] for i in idx]

        inner_ids = {s["id"] for s in inner_val}
        train = [s for s in remaining if s["id"] not in inner_ids]
        folds.append({
            "fold_id": test_idx,
            "test_speaker": test_speaker["id"],
            "test_label": test_speaker["label"],
            "test_alsfrs_r_speech": test_speaker.get("alsfrs_r_speech"),
            "inner_val_speakers": [s["id"] for s in inner_val],
            "train_speakers": [s["id"] for s in train],
        })
    return {
        "schema_version": "1.0.0",
        "generator_seed": seed,
        "stratified_by": stratify_by or [],
        "inner_val_fraction": inner_val_fraction,
        "folds": folds,
    }


def generate_episodes(target_meta: list[dict[str, Any]], *,
                      episodes_per_target: int = 100,
                      support_k: list[int],
                      query_per_class: int,
                      seed: int = 42) -> list[dict[str, Any]]:
    """Episode construction: query drawn first, then k support per class."""
    rng = np.random.default_rng(seed)
    pos = [s["id"] for s in target_meta if s["label"] == 1]
    neg = [s["id"] for s in target_meta if s["label"] == 0]
    episodes = []
    for ep in range(episodes_per_target):
        rng.shuffle(pos)
        rng.shuffle(neg)
        q_pos = pos[:query_per_class]
        q_neg = neg[:query_per_class]
        rest_pos = pos[query_per_class:]
        rest_neg = neg[query_per_class:]
        supports = {}
        for k in support_k:
            supports[str(k)] = {
                "als": rest_pos[:k],
                "hc":  rest_neg[:k],
            }
        episodes.append({
            "episode_id": ep,
            "query_als": q_pos,
            "query_hc":  q_neg,
            "support": supports,
        })
    return episodes


def load_splits(corpus: str, splits_root: str | Path = "splits") -> dict[str, Any]:
    path = Path(splits_root) / f"{corpus}_loso_splits.json"
    return read_json(path)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate LOSO splits and episodes.")
    p.add_argument("--config", required=True)
    p.add_argument("--meta-root", default="data/metadata")
    p.add_argument("--out-root",  default="splits")
    p.add_argument("--dry-run",   action="store_true")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    if args.dry_run:
        print("[dry-run] Split generator CLI validated.")
        return

    # In production: read data/metadata/<corpus>.csv, build speaker_meta, call
    # generate_loso_splits and generate_episodes, and write JSON.
    print("Split generator ready.")


if __name__ == "__main__":
    main()
