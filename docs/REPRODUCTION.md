# Reproduction checklist

A full rerun of the pipeline reproduces every table and figure in the
manuscript. Steps 1–3 are one-time; step 4 can be scoped per
experiment.

## 1. Obtain the three corpora

See `docs/DATA.md`. Place them as:

```
data/raw/voc_als/
data/raw/minsk2020/
data/raw/torgo/
```

No preprocessing happens in `data/raw/`; the next step writes everywhere
else.

## 2. Build the environment

```bash
bash scripts/setup_env.sh
```

* Python 3.10, PyTorch 2.2.1, transformers 4.38.2, parselmouth 0.4.3,
  Montreal Forced Aligner 2.2.17. See `requirements.txt` for the
  complete pin list.
* The script also downloads MFA's `english_us_arpa` acoustic model and
  dictionary.

## 3. Build features + splits + reference statistics

```bash
python -m sparc_als.data.preprocess       --config configs/preprocessing.yaml
python -m sparc_als.data.forced_alignment --config configs/forced_alignment.yaml
python -m sparc_als.data.splits           --config configs/preprocessing.yaml
python -m sparc_als.utils.reference_stats --config configs/acoustic_targets.yaml
```

The split files are written **bit-for-bit identical** to those in
`splits/*.json`. The reference statistics NPZ files are per fold and
per seed.

## 4. Run the experiments

The full seven-experiment sweep:

```bash
bash run.sh                             # ≈ 30 h on one A100
```

A single experiment:

```bash
bash run.sh E1                          # within-corpus LOSO
bash run.sh E3 E4                       # transfer + enrollment
```

A single seed (all experiments):

```bash
for s in 11 23 42 77 2025; do python scripts/E1_within_corpus.py --seed "$s"; done
```

## 5. Aggregate + paired tests

```bash
python -m sparc_als.eval.aggregate     --seeds 11 23 42 77 2025
python -m sparc_als.eval.paired_tests  --seeds 11 23 42 77 2025
```

## 6. Spot-check numerical equality

```bash
pytest tests/
```

The tests check:

* `tests/test_seeds.py` — the five release seeds yield the committed
  fixture predictions bit-for-bit.
* `tests/test_graph_prior.py` — the edge-weight matrix built from
  `configs/graph_priors.yaml` matches a known-good checksum.
* `tests/test_ndt_mask.py` — axis masks match Table 3.
* `tests/test_bootstrap.py` — class-stratified cluster bootstrap matches
  a hand-computed example.
* `tests/test_delong.py` — DeLong p-value on a toy case matches
  pROC::roc.test to 1e-4.

## 7. What-to-compare table

| Paper object        | Produced by                                      |
| ------------------- | ------------------------------------------------ |
| Table 5             | `python -m sparc_als.eval.metrics_table E1`      |
| Table 6             | `python -m sparc_als.eval.metrics_table E1 E3`   |
| Table 7             | `python -m sparc_als.eval.metrics_table E5`      |
| Fig. 2              | `python scripts/plot_confusion_and_roc.py`       |
| Fig. 3              | `python scripts/plot_cross_corpus.py`            |
| Section 6.5 probes  | `python scripts/E7_probes.py --seed 42`          |
