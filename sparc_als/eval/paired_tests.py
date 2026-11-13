"""Run paired McNemar + DeLong tests on seed-averaged speaker scores,
report Holm-adjusted p-values.
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from sparc_als.eval.delong import delong_test
from sparc_als.eval.mcnemar import holm_correction, mcnemar_exact_test
from sparc_als.utils.seeds import RELEASE_SEEDS


def run_paired_tests(corpus_csv: Path, methods: list[str]) -> pd.DataFrame:
    """corpus_csv: a long-form CSV with columns [speaker_id, method, logit_mean, label]."""
    df = pd.read_csv(corpus_csv)
    rows, pvals = [], []
    for m_a, m_b in combinations(methods, 2):
        a = df[df.method == m_a].set_index("speaker_id").sort_index()
        b = df[df.method == m_b].set_index("speaker_id").sort_index()
        assert (a.label.values == b.label.values).all()
        y_true = a.label.values.astype(int)
        mc = mcnemar_exact_test(y_true,
                                (a.logit_mean >= 0).astype(int).values,
                                (b.logit_mean >= 0).astype(int).values)
        dl = delong_test(y_true, a.logit_mean.values, b.logit_mean.values)
        rows.append({"a": m_a, "b": m_b, **{f"mcnemar_{k}": v for k, v in mc.items()},
                     **{f"delong_{k}": v for k, v in dl.items()}})
        pvals.extend([mc["p_value"], dl["p_value"]])
    adj = holm_correction(pvals)
    out = pd.DataFrame(rows)
    out["mcnemar_p_adj"] = adj[0::2]
    out["delong_p_adj"] = adj[1::2]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=list(RELEASE_SEEDS))
    p.add_argument("--inputs-root", default="results")
    p.add_argument("--out-root", default="results/paired")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    out = Path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)

    for corpus in ("voc_als", "minsk2020", "torgo"):
        path = Path(args.inputs_root) / f"{corpus}_paired_long.csv"
        if not path.exists():
            continue
        methods = pd.read_csv(path)["method"].unique().tolist()
        result = run_paired_tests(path, methods)
        if not args.dry_run:
            result.to_csv(out / f"{corpus}_paired.csv", index=False)
        print(f"  {corpus}: {len(result)} pairwise tests")


if __name__ == "__main__":
    main()
