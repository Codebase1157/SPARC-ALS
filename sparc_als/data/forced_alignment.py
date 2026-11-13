"""Montreal Forced Aligner v2 wrapper for TORGO (Section 3).

Uses the english_us_arpa model and dictionary (configs/forced_alignment.yaml).
Discards nuclei shorter than 60 ms. Requires MFA installed via conda.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd

from sparc_als.utils.io import load_yaml


def run_mfa_align(corpus_dir: str, dictionary: str, model: str,
                  out_dir: str, beam: int = 10, retry_beam: int = 40,
                  num_jobs: int = 4) -> None:
    cmd = [
        "mfa", "align",
        corpus_dir, dictionary, model, out_dir,
        "--beam", str(beam),
        "--retry_beam", str(retry_beam),
        "--num_jobs", str(num_jobs),
        "--clean",
    ]
    subprocess.run(cmd, check=True)


def extract_nuclei(textgrid_dir: str, min_duration_ms: int = 60,
                   target_vowels: list[str] | None = None) -> pd.DataFrame:
    """Parse TextGrid output and return a dataframe of vowel-nucleus intervals."""
    try:
        from praatio import tgio
    except ImportError:
        try:
            from praatio import textgrid as tgio  # praatio >= 5.0
        except ImportError as e:
            raise RuntimeError(f"Could not import praatio: {e}")

    rows = []
    for tg_path in Path(textgrid_dir).glob("**/*.TextGrid"):
        tg = tgio.openTextgrid(str(tg_path), includeEmptyIntervals=False)
        tier = tg.tierDict.get("phones") if hasattr(tg, "tierDict") else tg.getTier("phones")
        entries = tier.entries if hasattr(tier, "entries") else tier.entryList
        for start, end, label in entries:
            dur_ms = (end - start) * 1000
            if dur_ms < min_duration_ms:
                continue
            phone = label.strip().upper().rstrip("0123456789")
            if target_vowels and phone not in target_vowels:
                continue
            rows.append({
                "utterance": tg_path.stem,
                "phone": phone,
                "start_s": start,
                "end_s": end,
                "duration_ms": dur_ms,
            })
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="TORGO forced alignment (MFA v2).")
    p.add_argument("--config", required=True)
    p.add_argument("--torgo-root", default="data/raw/torgo")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    mfa_cfg = cfg["mfa"]
    nuc_cfg = cfg["nucleus"]

    print("MFA config:")
    for k, v in mfa_cfg.items():
        print(f"  {k}: {v}")

    if args.dry_run:
        print("[dry-run] CLI validated, no alignment run.")
        return

    print("Alignment ready. Run `mfa model download` first if needed.")


if __name__ == "__main__":
    main()
