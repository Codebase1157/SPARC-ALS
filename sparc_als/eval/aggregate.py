"""Aggregate per-seed CSVs into seed-averaged speaker logits for paired tests."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sparc_als.utils.seeds import RELEASE_SEEDS


def aggregate_corpus(pred_root: Path, corpus: str, experiment: str,
                     seeds: list[int]) -> pd.DataFrame:
    frames = []
    for seed in seeds:
        path = pred_root / corpus / experiment / f"speaker_logits_seed{seed}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["seed_id"] = seed
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    big = pd.concat(frames, ignore_index=True)
    agg = big.groupby("speaker_id").agg(
        logit_mean=("logit", "mean"),
        logit_std=("logit", "std"),
        label=("label", "first"),
    ).reset_index()
    return agg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=list(RELEASE_SEEDS))
    p.add_argument("--pred-root", default="predictions")
    p.add_argument("--out-root", default="results")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    for corpus in ("voc_als", "minsk2020", "torgo"):
        for experiment in ("E1", "E3", "E4"):
            agg = aggregate_corpus(Path(args.pred_root), corpus, experiment, args.seeds)
            if agg.empty:
                continue
            out_path = out_root / f"{corpus}_{experiment}_seed_avg.csv"
            if not args.dry_run:
                agg.to_csv(out_path, index=False)
            print(f"  wrote {out_path}  ({len(agg)} speakers)")


if __name__ == "__main__":
    main()
