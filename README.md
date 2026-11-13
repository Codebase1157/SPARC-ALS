# SPARC-ALS v1.0.0

**Speaker-Invariant Pathology-Aware Representation with Vowel-Space Graph Attention for Cross-Corpus Detection of Amyotrophic Lateral Sclerosis**

Reference implementation for the manuscript by Komal Ahuja and Amanpreet Kaur
(Chitkara University Institute of Engineering and Technology, India).

This release contains the versioned implementation, exact preprocessing and
forced-alignment settings, graph priors, acoustic proxy target configurations
and masks, counterfactual-generation operators, speaker-disjoint split and
enrollment indices, baseline (B1–B6) and control (C1–C6) configurations,
training seeds, checkpoint rules, calibration temperatures, and per-speaker /
per-utterance predictions tagged by seed, so every result reported in the
paper can be audited and rerun.

---

## 1. Headline results reproduced by this release

| Corpus      | n   | BAcc (%)       | AUC            | ECE / Brier    |
| ----------- | --- | -------------- | -------------- | -------------- |
| VOC-ALS     | 153 | 88.7 ± 0.9     | 0.942 ± 0.008  | 0.041 / 0.082  |
| Minsk2020   | 64  | 85.9 ± 1.2     | 0.912 ± 0.013  | 0.057 / 0.108  |
| TORGO       | 15  | 79.5 ± 2.9     | 0.839 ± 0.031  | 0.094 / 0.147  |
| **Cross (mean)** |  |            | **0.826**      |                |

TORGO results reflect dysarthria of mixed etiology and are never reported as
ALS detection in isolation.

---

## 2. Quick start

```bash
# 1. Clone and enter
git clone <YOUR_GITHUB_URL> sparc-als
cd sparc-als

# 2. Create environment (conda/mamba or venv)
bash scripts/setup_env.sh

# 3. Place corpora (see docs/DATA.md for licensing) under data/raw/
#    data/raw/voc_als/
#    data/raw/minsk2020/
#    data/raw/torgo/

# 4. Reproduce every result reported in the paper
bash run.sh            # ≈ 30 h on one A100; see run.sh for sub-targets
```

Every script accepts `--dry-run` for a shape-only pass that validates
configs, splits and I/O without touching the GPU.

---

## 3. Repository layout

```
sparc-als/
├── run.sh                     # Top-level driver — reproduces everything
├── configs/                   # Versioned YAML configs
│   ├── preprocessing.yaml     # Resampling, VAD, trimming, bandwidth
│   ├── forced_alignment.yaml  # MFA v2 settings, nucleus filters
│   ├── graph_priors.yaml      # Vowel-space edges, formants, syllable links
│   ├── acoustic_targets.yaml  # NDT 4 axes, Praat settings, masks
│   ├── counterfactuals.yaml   # T1–T4 operators, strengths, limits
│   ├── training.yaml          # Seeds, optimizer, hyperparameters, temps
│   ├── baselines/             # B1 … B6
│   └── controls/              # C1 … C6
├── sparc_als/                 # Library
│   ├── data/                  # preprocess, forced_alignment, loaders, splits
│   ├── models/                # encoder, adapters, vsga, ndt, heads
│   ├── training/              # train, losses, counterfactuals, meta_adapt
│   ├── baselines/             # Reimplemented baselines
│   ├── eval/                  # metrics, bootstrap, DeLong, McNemar, calibration
│   └── utils/                 # seeds, acoustic_features, io
├── splits/                    # Frozen, released speaker-level indices
│   ├── voc_als_loso_splits.json       # 153 LOSO folds
│   ├── minsk2020_loso_splits.json     # 64 LOSO folds
│   ├── torgo_loso_splits.json         # 15 LOSO folds
│   ├── enrollment_indices.json        # Support / query indices per episode
│   └── README.md
├── scripts/                   # One script per experiment (E1 … E7)
├── predictions/               # Per-speaker and per-utterance logits/labels
│                              # tagged by seed_id ∈ {11, 23, 42, 77, 2025}
├── docs/
│   ├── DATA.md                # Access and license notes
│   ├── REPRODUCTION.md        # Full checklist
│   └── MANUSCRIPT_LINKS.md    # Section ↔ file map
└── tests/                     # Unit tests (seeds, masks, graph, metrics)
```

---

## 4. The seven experiments (E1 – E7)

| ID | Script                               | What it reports                                         |
| -- | ------------------------------------ | ------------------------------------------------------- |
| E1 | `scripts/E1_within_corpus.py`        | LOSO detection on VOC-ALS / Minsk2020 / TORGO           |
| E2 | `scripts/E2_severity.py`             | VOC-ALS five-level severity, QWK, Wilson intervals      |
| E3 | `scripts/E3_cross_corpus.py`         | Six cross-corpus directions on /a/–/i/ tier             |
| E4 | `scripts/E4_few_shot.py`             | Meta-training on two corpora, enrollment on the third   |
| E5 | `scripts/E5_ablations.py`            | Full ablations and the five control families            |
| E6 | `scripts/E6_factor_controlled.py`    | 4 kHz LPF, ±10 dB, SSN at 20 dB SNR, truncation         |
| E7 | `scripts/E7_probes.py`               | Linear probes, null-transform test, token deletion      |

---

## 5. Seeds, checkpoints, temperatures

* **Seeds:** `{11, 23, 42, 77, 2025}` — driven centrally by `sparc_als/utils/seeds.py`.
* **Checkpoint rule:** lowest inner-validation loss over at most 60 epochs with
  patience 10. In E4 the checkpoint is chosen on 200 fixed source
  inner-validation episodes.
* **Temperatures:** fit once on source inner-validation speakers and **never**
  refit on a target corpus; stored per fold in
  `predictions/<corpus>/temperature_seed<seed>.json`.

---

## 6. Baselines and controls on identical splits

All baselines (`B1` WavLM linear probe, `B2` HuBERT + SVM, `B3` LASSO + LDA,
`B4` CNN-BiLSTM, `B5` triplet regularizer, `B6` prototypical network) and
controls (`C1` matched multi-task, `C2` mean pooling, `C3` token transformer,
`C4` uniform / permuted priors, `C5` nuisance ablations, `C6` token reduction)
use the **same** splits, preprocessing, seeds, 20-config inner-validation
search, and inferential unit (speaker), so comparisons are paired at the
speaker level.

---

## 7. Per-speaker / per-utterance predictions

For every experiment the pipeline writes:

```
predictions/<corpus>/<experiment>/
    speaker_logits_seed<seed>.csv    # speaker_id, logit, prob, decision, label
    utterance_logits_seed<seed>.csv  # speaker_id, token, logit, prob, label
    temperature_seed<seed>.json      # fit once on source, frozen on target
```

Every row carries its `seed_id` so paired McNemar and DeLong tests on the
seed-averaged speaker score `l̄_i = (1/5) Σ_s l_{i,s}` are reproducible
line-for-line.

---

## 8. Citing

See `CITATION.cff` for the preferred citation. If you use this release, please
also cite the three corpora:

* VOC-ALS — Dubbioso et al., *Scientific Data* 11, 800 (2024).
* Minsk2020 — Vashkevich & Rushkevich, *BSPC* 65, 102350 (2021).
* TORGO — Rudzicz et al., *LREC* 46, 523–541 (2012).

---

## 9. License

Code is released under the MIT License (see `LICENSE`).

Each corpus retains its own license and must be obtained separately from the
upstream providers.
