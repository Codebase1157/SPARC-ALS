# Per-speaker / per-utterance predictions

Every experiment writes predictions here, tagged by seed. Running
`bash run.sh` reproduces every file. The fixture rows committed below
show the exact schema; running the full pipeline overwrites them with
the real values.

## Files

```
predictions/<corpus>/<experiment>/
    speaker_logits_seed<seed>.csv    # one row per test speaker, per seed
    utterance_logits_seed<seed>.csv  # one row per test token, per seed
    temperature_seed<seed>.json      # calibration temperature for the fold
```

* `<corpus>` ∈ `{voc_als, minsk2020, torgo}`
* `<experiment>` ∈ `{E1, E2, E3, E4, E5, E6, E7}`
* `<seed>` ∈ `{11, 23, 42, 77, 2025}`

## speaker_logits CSV schema

| column          | type   | description                                                   |
| --------------- | ------ | ------------------------------------------------------------- |
| speaker_id      | str    | Speaker ID as released by the upstream corpus                  |
| logit           | float  | Model's decision logit at speaker level                        |
| prob            | float  | sigmoid(logit / T)                                             |
| decision        | int    | 1 if prob ≥ threshold (default 0.5), else 0                    |
| label           | int    | Ground-truth binary label (ALS = 1, HC = 0)                    |
| severity        | int    | ALSFRS-R speech item 4..1 → 1..4, HC → 0, NA → -1              |
| seed_id         | int    | One of {11, 23, 42, 77, 2025}                                  |
| corpus          | str    | voc_als | minsk2020 | torgo                                  |
| fold_id         | int    | LOSO fold or episode index                                     |
| experiment      | str    | E1..E7                                                         |

## utterance_logits CSV schema

| column          | type   | description                                      |
| --------------- | ------ | ------------------------------------------------ |
| speaker_id      | str    |                                                  |
| token           | str    | one of {a,e,i,o,u,pa,ta,ka} or TORGO utterance id |
| logit           | float  | token-level logit                                 |
| prob            | float  | token-level probability                           |
| label           | int    |                                                  |
| seed_id         | int    |                                                  |

## Paired tests

Paired McNemar on decisions and DeLong on scores use the seed-averaged
logit `l̄_i = (1/5) Σ_s l_{i,s}` as the single observation per speaker.
The aggregation step is:

```bash
python -m sparc_als.eval.aggregate     --seeds 11 23 42 77 2025
python -m sparc_als.eval.paired_tests  --seeds 11 23 42 77 2025
```

The Holm-adjusted p-values reported in Tables 6 and 7 of the paper come
from this exact pipeline.
